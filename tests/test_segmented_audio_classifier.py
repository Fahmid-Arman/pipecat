#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for classifiers that answer questions about a clip of audio."""

import pytest

from pipecat.classifiers.base_classifier import ClassifierError, YesNoQuestion, YesNoResult
from pipecat.classifiers.segmented_audio_classifier import ClassifierAudio, SegmentedAudioClassifier

QUESTIONS = {"frustrated": YesNoQuestion(instructions="Does the speaker sound frustrated?")}


class FakeAudioClassifier(SegmentedAudioClassifier):
    def __init__(self, *, sample_rate: int | None = None):
        super().__init__()
        self._rate = sample_rate
        self.clips: list[ClassifierAudio] = []
        self.states: list = []

    @property
    def min_audio_secs(self) -> float:
        return 1.0

    @property
    def max_audio_secs(self) -> float:
        return 3.0

    @property
    def sample_rate(self) -> int | None:
        return self._rate

    async def _ask_segment(self, state, questions, audio, images):
        self.clips.append(audio)
        self.states.append(state)
        return {name: YesNoResult(probability=0.7) for name in questions}, None


def _clip(secs: float, sample_rate: int = 16000) -> ClassifierAudio:
    return ClassifierAudio(audio=bytes(int(secs * sample_rate) * 2), sample_rate=sample_rate)


def test_a_clip_knows_how_long_it_is():
    assert _clip(2.5).duration_secs == 2.5


@pytest.mark.asyncio
async def test_questions_are_answered_about_the_clip_and_its_state():
    classifier = FakeAudioClassifier()
    clip = _clip(2)

    results = await classifier.yes_no("A caller.", QUESTIONS, audio=clip)

    assert results == {"frustrated": YesNoResult(probability=0.7)}
    assert classifier.clips == [clip]
    assert classifier.states == ["A caller."]


@pytest.mark.asyncio
async def test_questions_without_a_clip_are_refused():
    classifier = FakeAudioClassifier()

    with pytest.raises(ClassifierError):
        await classifier.ask("A caller.", QUESTIONS)
    with pytest.raises(ClassifierError):
        await classifier.yes_no("A caller.", QUESTIONS)


@pytest.mark.asyncio
@pytest.mark.parametrize("secs", [0.5, 3.5])
async def test_a_clip_of_a_length_the_classifier_doesnt_accept_is_refused(secs):
    classifier = FakeAudioClassifier()

    with pytest.raises(ClassifierError):
        await classifier.ask("A caller.", QUESTIONS, audio=_clip(secs))

    assert classifier.clips == []


@pytest.mark.asyncio
async def test_a_clip_is_resampled_to_the_classifiers_sample_rate():
    classifier = FakeAudioClassifier(sample_rate=16000)

    await classifier.ask("A caller.", QUESTIONS, audio=_clip(2, sample_rate=8000))

    [clip] = classifier.clips
    assert clip.sample_rate == 16000
    assert clip.duration_secs == pytest.approx(2, abs=0.05)


@pytest.mark.asyncio
async def test_every_call_reports_its_metrics():
    classifier = FakeAudioClassifier()
    reported = []

    @classifier.event_handler("on_metrics")
    async def on_metrics(classifier, data):
        reported.append(data)

    await classifier.ask("A caller.", QUESTIONS, audio=_clip(2))
    await classifier.cleanup()

    assert len(reported) == 1
