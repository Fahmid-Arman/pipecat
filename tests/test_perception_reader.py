#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the processor that reads each user turn by asking a classifier."""

import asyncio

import pytest

from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    ScoreLevel,
    ScoreQuestion,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.classifiers.segmented_audio_classifier import ClassifierAudio, SegmentedAudioClassifier
from pipecat.expressive.perception import PerceptionSignal
from pipecat.expressive.processors.perception_reader import PerceptionQuestion, PerceptionReader
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    LLMContextAssistantTurnFrame,
    MetricsFrame,
    PerceptionMetadataFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test

QUESTIONS = {
    "tone": PerceptionQuestion(
        question=ChoiceQuestion(
            instructions="How does the user sound?",
            options={"calm": None, "frustrated": None, "confused": None},
        ),
        description="How the user sounded.",
    ),
    "hesitation": PerceptionQuestion(
        question=YesNoQuestion(instructions="Does the user sound unsure?"),
        description="The chance the user sounded unsure, from 0 to 1.",
    ),
    "energy": PerceptionQuestion(
        question=ScoreQuestion(instructions="How energetic is the user?", levels=["low", "high"]),
        description="From 0, low, to 1, high.",
    ),
}

RESULTS = {
    "tone": ChoiceResult(
        choice="frustrated",
        probabilities={"calm": 0.1, "frustrated": 0.7, "confused": 0.2},
        confidence=0.6,
    ),
    "hesitation": YesNoResult(probability=0.6432),
    "energy": ScoreResult(
        score=0.678,
        levels=[
            ScoreLevel(level="low", probability=0.32),
            ScoreLevel(level="high", probability=0.68),
        ],
        confidence=0.4,
    ),
}


class FakeClassifier(BaseClassifier):
    def __init__(
        self,
        *,
        error: Exception | None = None,
        setup_error: Exception | None = None,
        delay: float = 0.0,
    ):
        super().__init__()
        self.asked: list[tuple] = []
        self._error = error
        self._setup_error = setup_error
        self._delay = delay

    async def setup(self, task_manager):
        await super().setup(task_manager)
        if self._setup_error:
            raise self._setup_error

    async def _ask(self, state, questions, images=()):
        self.asked.append((state, dict(questions)))
        await asyncio.sleep(self._delay)
        if self._error:
            raise self._error
        return {name: RESULTS[name] for name in questions}, None


def _transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text=text, user_id="", timestamp="now")


@pytest.mark.asyncio
async def test_the_signals_are_declared_at_start():
    reader = PerceptionReader(classifier=FakeClassifier(), questions=QUESTIONS)

    down, _ = await run_test(reader, frames_to_send=[])

    metadata = [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert [(t.name, t.description, t.labels) for t in metadata[0].perception.signals] == [
        ("tone", "How the user sounded.", ["calm", "frustrated", "confused"]),
        ("hesitation", "The chance the user sounded unsure, from 0 to 1.", []),
        ("energy", "From 0, low, to 1, high.", []),
    ]


@pytest.mark.asyncio
async def test_each_answer_is_set_as_a_signal_on_its_transcription():
    down, _ = await run_test(
        PerceptionReader(classifier=FakeClassifier(), questions=QUESTIONS),
        frames_to_send=[_transcription("I already gave you my order number.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert down[1].perception == [
        PerceptionSignal("tone", labels=[("frustrated", 0.7), ("confused", 0.2)]),
        PerceptionSignal("hesitation", value=0.64),
        PerceptionSignal("energy", value=0.68),
    ]


@pytest.mark.asyncio
async def test_a_choice_reports_at_most_its_max_labels():
    question = QUESTIONS["tone"].model_copy(update={"max_labels": 1})

    down, _ = await run_test(
        PerceptionReader(classifier=FakeClassifier(), questions={"tone": question}),
        frames_to_send=[_transcription("Hello.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert down[1].perception == [PerceptionSignal("tone", labels=[("frustrated", 0.7)])]


@pytest.mark.asyncio
async def test_a_turn_is_asked_about_with_the_bots_words_before_it():
    classifier = FakeClassifier()

    await run_test(
        PerceptionReader(classifier=classifier, questions=QUESTIONS),
        frames_to_send=[
            _transcription("Hi."),
            LLMContextAssistantTurnFrame(text="What's your order number?", timestamp="now"),
            _transcription("I already gave it to you."),
        ],
    )

    assert [state for state, _ in classifier.asked] == [
        {"user": "Hi."},
        {"assistant": "What's your order number?", "user": "I already gave it to you."},
    ]


@pytest.mark.asyncio
async def test_a_late_reading_lets_the_transcription_pass_unread():
    down, up = await run_test(
        PerceptionReader(classifier=FakeClassifier(delay=1.0), questions=QUESTIONS, timeout=0.05),
        frames_to_send=[_transcription("Hello.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert down[1].perception is None
    assert not [f for f in up if isinstance(f, ErrorFrame)]


@pytest.mark.asyncio
async def test_a_failed_reading_is_reported_and_the_transcription_still_passes():
    down, up = await run_test(
        PerceptionReader(
            classifier=FakeClassifier(error=ClassifierError("down")), questions=QUESTIONS
        ),
        frames_to_send=[_transcription("Hello.")],
        expected_down_frames=[PerceptionMetadataFrame, TranscriptionFrame],
    )

    assert down[1].perception is None
    assert [f for f in up if isinstance(f, ErrorFrame)]


@pytest.mark.asyncio
async def test_a_classifier_that_cant_be_set_up_leaves_the_reader_unusable():
    classifier = FakeClassifier(setup_error=ClassifierError("no model"))
    reader = PerceptionReader(classifier=classifier, questions=QUESTIONS)

    down, up = await run_test(reader, frames_to_send=[_transcription("Hello.")])

    assert not reader.is_usable
    assert [f for f in up if isinstance(f, ErrorFrame)]
    assert not [f for f in down if isinstance(f, PerceptionMetadataFrame)]
    assert [f.perception for f in down if isinstance(f, TranscriptionFrame)] == [None]
    assert classifier.asked == []


def test_a_name_that_cant_be_written_in_a_reading_is_refused():
    with pytest.raises(ValueError):
        PerceptionReader(
            classifier=FakeClassifier(), questions={"speech rate": QUESTIONS["energy"]}
        )


@pytest.mark.asyncio
async def test_the_classifiers_metrics_are_pushed_when_metrics_are_on():
    frames = [_transcription("Hello."), SleepFrame(sleep=0.2)]

    with_metrics, _ = await run_test(
        PerceptionReader(classifier=FakeClassifier(), questions=QUESTIONS),
        frames_to_send=frames,
        pipeline_params=PipelineParams(enable_metrics=True),
    )
    without_metrics, _ = await run_test(
        PerceptionReader(classifier=FakeClassifier(), questions=QUESTIONS),
        frames_to_send=frames,
    )

    assert [f for f in with_metrics if isinstance(f, MetricsFrame)]
    assert not [f for f in without_metrics if isinstance(f, MetricsFrame)]


#: The sample rate of the pipeline's input audio in these tests.
SAMPLE_RATE = 16000

AUDIO_QUESTIONS = {"hesitation": QUESTIONS["hesitation"]}


class FakeAudioClassifier(SegmentedAudioClassifier):
    def __init__(self, *, min_secs: float = 1.0, delay: float = 0.0):
        super().__init__()
        self.clips: list[ClassifierAudio] = []
        self.states: list = []
        self._min_secs = min_secs
        self._delay = delay

    @property
    def min_audio_secs(self) -> float:
        return self._min_secs

    async def _ask_segment(self, state, questions, audio, images):
        self.clips.append(audio)
        self.states.append(state)
        await asyncio.sleep(self._delay)
        return {name: RESULTS[name] for name in questions}, None


def _audio(value: int, secs: float) -> InputAudioRawFrame:
    """``secs`` seconds of input audio whose every byte is ``value``."""
    audio = bytes([value]) * (int(SAMPLE_RATE * secs) * 2)
    return InputAudioRawFrame(audio=audio, sample_rate=SAMPLE_RATE, num_channels=1)


def _pcm(value: int, secs: float) -> bytes:
    return _audio(value, secs).audio


def _perception(frames) -> list:
    return [f.perception for f in frames if isinstance(f, TranscriptionFrame)]


@pytest.mark.asyncio
async def test_an_audio_classifier_hears_the_turn_from_a_second_before_its_speech():
    classifier = FakeAudioClassifier()

    down, _ = await run_test(
        PerceptionReader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            LLMContextAssistantTurnFrame(text="What's your order number?", timestamp="now"),
            SleepFrame(sleep=0.05),
            _audio(1, 2),
            VADUserStartedSpeakingFrame(),
            _audio(2, 2),
            VADUserStoppedSpeakingFrame(),
            _transcription("I already gave it to you."),
        ],
    )

    assert classifier.clips == [
        ClassifierAudio(audio=_pcm(1, 1) + _pcm(2, 2), sample_rate=SAMPLE_RATE)
    ]
    assert classifier.states == [{"assistant": "What's your order number?"}]
    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.64)]]


@pytest.mark.asyncio
async def test_a_turn_with_a_pause_is_heard_as_one_clip():
    classifier = FakeAudioClassifier(min_secs=1.5)

    down, _ = await run_test(
        PerceptionReader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 0.5),
            VADUserStoppedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _audio(2, 1),
            VADUserStoppedSpeakingFrame(),
            _transcription("I want to... cancel my order."),
        ],
    )

    assert classifier.clips == [
        ClassifierAudio(audio=_pcm(1, 0.5) + _pcm(2, 1), sample_rate=SAMPLE_RATE)
    ]
    assert _perception(down) == [[PerceptionSignal("hesitation", value=0.64)]]


@pytest.mark.asyncio
async def test_a_turn_shorter_than_the_classifier_takes_passes_unread():
    classifier = FakeAudioClassifier(min_secs=5.0)

    down, _ = await run_test(
        PerceptionReader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 1),
            VADUserStoppedSpeakingFrame(),
            _transcription("Fine."),
        ],
    )

    assert classifier.clips == []
    assert _perception(down) == [None]


@pytest.mark.asyncio
async def test_a_new_turn_doesnt_get_the_last_turns_reading():
    classifier = FakeAudioClassifier()

    down, _ = await run_test(
        PerceptionReader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(),
            SleepFrame(sleep=0.1),
            UserStoppedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            _audio(2, 0.2),
            VADUserStoppedSpeakingFrame(),
            _transcription("Yes."),
        ],
    )

    assert _perception(down) == [None]


@pytest.mark.asyncio
async def test_audio_heard_while_the_user_is_muted_isnt_kept():
    classifier = FakeAudioClassifier()

    await run_test(
        PerceptionReader(classifier=classifier, questions=AUDIO_QUESTIONS),
        frames_to_send=[
            UserMuteStartedFrame(),
            _audio(1, 2),
            UserMuteStoppedFrame(),
            _audio(2, 0.5),
            VADUserStartedSpeakingFrame(),
            _audio(3, 1),
            VADUserStoppedSpeakingFrame(),
            _transcription("Hello?"),
        ],
    )

    assert classifier.clips == [
        ClassifierAudio(audio=_pcm(2, 0.5) + _pcm(3, 1), sample_rate=SAMPLE_RATE)
    ]


@pytest.mark.asyncio
async def test_a_late_audio_reading_lets_the_transcription_pass_unread():
    down, up = await run_test(
        PerceptionReader(
            classifier=FakeAudioClassifier(delay=1.0), questions=AUDIO_QUESTIONS, timeout=0.05
        ),
        frames_to_send=[
            VADUserStartedSpeakingFrame(),
            _audio(1, 2),
            VADUserStoppedSpeakingFrame(),
            _transcription("Hello."),
        ],
    )

    assert _perception(down) == [None]
    assert not [f for f in up if isinstance(f, ErrorFrame)]
