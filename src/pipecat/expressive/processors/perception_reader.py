#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""The processor that reads how each user turn was said by asking a classifier."""

import asyncio
import re
from collections.abc import Mapping
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.audio.speech_segment_buffer import SpeechSegmentBuffer
from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    ClassifierQuestion,
    ClassifierResult,
    ScoreResult,
)
from pipecat.classifiers.segmented_audio_classifier import ClassifierAudio, SegmentedAudioClassifier
from pipecat.expressive.perception import PerceptionCapabilities, PerceptionSignal, PerceptionTerm
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    LLMContextAssistantTurnFrame,
    MetricsFrame,
    PerceptionMetadataFrame,
    StartFrame,
    TranscriptionFrame,
    UserMuteStartedFrame,
    UserMuteStoppedFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import MetricsData
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
    FrameProcessorSetup,
)

#: A signal's name, as written in a reading: no whitespace, "=", ",", "[", "]", "(" or ")".
_SIGNAL_NAME = re.compile(r"[^\s=,\[\]()]+")


class PerceptionQuestion(BaseModel):
    """One signal a :class:`PerceptionReader` reports, and the question it asks for it.

    Parameters:
        question: What the classifier is asked. A choice is reported as its
            likeliest options, a yes/no as its probability and a score as its
            position on the scale.
        description: A short line telling the LLM how to read the signal,
            such as its range, or empty.
        max_labels: The most options a choice reports, likeliest first.
    """

    question: ClassifierQuestion
    description: str = ""
    max_labels: int = Field(default=2, ge=1)


class PerceptionReader(FrameProcessor):
    """Reads how each user turn was said by asking a classifier.

    Goes between the STT service and the user aggregator. The answers are set
    as each final transcription's ``perception``: one signal per question,
    named after it. The signals are declared at start in a
    :class:`~pipecat.frames.frames.PerceptionMetadataFrame`.

    A text classifier is asked about each final transcription, with the bot's
    reply before it. A
    :class:`~pipecat.classifiers.segmented_audio_classifier.SegmentedAudioClassifier`
    is asked about the audio of the user's turn instead, with the bot's reply
    before it as the state. Each time the VAD reports silence, it hears the
    turn so far, from a second before its speech started. A turn shorter than
    the classifier's ``min_audio_secs`` isn't asked about, a clip isn't sent
    while the turn's last one is still being read, and no audio is kept while
    the user is muted.

    The transcription waits for its reading up to ``timeout`` seconds, and
    passes unread when the reading is late or the classifier fails. A
    classifier that can't be set up is reported as an error that leaves the
    reader unusable, and transcriptions pass unread.

    Part of expressive conversation (alpha): turn it on with
    ``PipelineParams(enable_expressive_conversation=True)`` for the LLM to read
    the turns.

    Example::

        reader = PerceptionReader(
            classifier=JevClassifier(api_key=os.getenv("TYPESAFE_API_KEY")),
            questions={
                "frustration": PerceptionQuestion(
                    question=YesNoQuestion(instructions="Is the user frustrated?"),
                    description="The chance the user is frustrated, from 0 to 1.",
                ),
            },
        )

        pipeline = Pipeline([transport.input(), stt, reader, user_aggregator, llm, ...])
    """

    def __init__(
        self,
        *,
        classifier: BaseClassifier,
        questions: Mapping[str, PerceptionQuestion],
        timeout: float = 0.5,
        **kwargs,
    ):
        """Initialize the processor.

        Args:
            classifier: What answers the questions about each turn.
            questions: The questions, by the name of the signal each reports.
            timeout: Seconds a transcription waits for its reading.
            **kwargs: Additional arguments passed to the parent.

        Raises:
            ValueError: If a name can't be written in a reading.
        """
        super().__init__(**kwargs)
        for name in questions:
            if not _SIGNAL_NAME.fullmatch(name):
                raise ValueError(f"{name!r} can't name a signal: it has a space or one of =,[]()")
        self._classifier = classifier
        self._questions = dict(questions)
        self._timeout = timeout
        self._previous: str | None = None
        # The user's turn, kept and read for a classifier that hears audio.
        self._audio_classifier = (
            classifier if isinstance(classifier, SegmentedAudioClassifier) else None
        )
        self._speech_buffer = SpeechSegmentBuffer(
            max_secs=self._audio_classifier.max_audio_secs if self._audio_classifier else None
        )
        self._sample_rate = 0
        self._muted = False
        self._audio_read: asyncio.Task | None = None
        self._classifier.add_event_handler("on_metrics", self._on_classifier_metrics)

    @property
    def classifier(self) -> BaseClassifier:
        """The classifier that answers the questions."""
        return self._classifier

    @property
    def questions(self) -> dict[str, PerceptionQuestion]:
        """The questions asked about each turn, by the name of the signal each reports."""
        return self._questions

    async def setup(self, setup: FrameProcessorSetup):
        """Set up the processor and its classifier.

        Args:
            setup: Configuration object containing setup parameters.
        """
        await super().setup(setup)
        self._sample_rate = setup.audio_in_sample_rate
        self._speech_buffer.set_sample_rate(self._sample_rate)
        try:
            await self._classifier.setup(self.task_manager)
        except ClassifierError as e:
            await self.push_error(f"{self}: {e}", e, force_treat_as_permanent=True)

    async def cleanup(self):
        """Clean up the processor and its classifier."""
        await super().cleanup()
        await self._end_turn()
        await self._classifier.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Read the user's turns, and keep the bot's replies to read them against.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            self._handle_audio(frame)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._handle_vad_user_started_speaking(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._handle_vad_user_stopped_speaking(frame)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)
        elif isinstance(frame, (UserMuteStartedFrame, UserMuteStoppedFrame)):
            await self._handle_mute(frame)
        elif isinstance(frame, LLMContextAssistantTurnFrame):
            self._handle_assistant_turn(frame)

        await self.push_frame(frame, direction)

        # The declaration follows the StartFrame, which must reach every
        # processor before anything else.
        if isinstance(frame, StartFrame):
            await self._declare_signals()

    def _handle_audio(self, frame: InputAudioRawFrame):
        if self._audio_classifier and not self._muted:
            self._speech_buffer.append(frame.audio)

    async def _handle_transcription(self, frame: TranscriptionFrame):
        if frame.text.strip() and self.is_usable:
            if self._audio_classifier:
                frame.perception = await self._audio_reading()
            else:
                frame.perception = await self._read(frame.text)

    def _handle_vad_user_started_speaking(self, frame: VADUserStartedSpeakingFrame):
        self._speech_buffer.start()

    def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        if self._audio_classifier and self.is_usable:
            self._read_turn_audio(self._audio_classifier)

    async def _handle_user_stopped_speaking(self, frame: UserStoppedSpeakingFrame):
        await self._end_turn()

    async def _handle_mute(self, frame: UserMuteStartedFrame | UserMuteStoppedFrame):
        self._muted = isinstance(frame, UserMuteStartedFrame)
        if self._muted:
            await self._end_turn()

    def _handle_assistant_turn(self, frame: LLMContextAssistantTurnFrame):
        self._previous = frame.text

    async def _declare_signals(self):
        """Push the signals the reader reports, if it can report them."""
        if self.is_usable:
            await self.push_frame(
                PerceptionMetadataFrame(service_name=self.name, perception=self._capabilities())
            )

    def _capabilities(self) -> PerceptionCapabilities:
        """The signals the reader reports, from its questions."""
        return PerceptionCapabilities(
            signals=[
                PerceptionTerm(name, question.description, labels=self._labels(question.question))
                for name, question in self.questions.items()
            ]
        )

    def _labels(self, question: ClassifierQuestion) -> list[str]:
        """The labels a signal can report: a choice's options, or none."""
        return list(question.options) if isinstance(question, ChoiceQuestion) else []

    async def _read(self, text: str) -> list[PerceptionSignal] | None:
        """A transcription's reading, or None when it's late or the classifier fails."""
        questions = self.questions
        state = {"assistant": self._previous, "user": text} if self._previous else {"user": text}
        signals = None
        try:
            results = await asyncio.wait_for(
                self._classifier.ask(state, self._classifier_questions(questions)),
                self._timeout,
            )
            signals = self._signals(results, questions)
        except TimeoutError:
            logger.warning(f"{self}: no reading after {self._timeout}s, the turn passes unread")
        except ClassifierError as e:
            await self.push_error(f"{self}: {e}", e)
        return signals

    def _read_turn_audio(self, classifier: SegmentedAudioClassifier):
        """Start reading the turn's audio so far, if it's long enough and no read is waiting."""
        if self._audio_read and not self._audio_read.done():
            return
        clip = ClassifierAudio(audio=self._speech_buffer.audio, sample_rate=self._sample_rate)
        if clip.duration_secs >= classifier.min_audio_secs:
            state = {"assistant": self._previous} if self._previous else {}
            self._audio_read = self.create_task(self._read_audio(classifier, state, clip))

    async def _read_audio(
        self, classifier: SegmentedAudioClassifier, state: dict[str, Any], clip: ClassifierAudio
    ) -> list[PerceptionSignal] | None:
        """A clip's reading, or None when the classifier fails."""
        questions = self.questions
        signals = None
        try:
            results = await classifier.ask(state, self._classifier_questions(questions), audio=clip)
            signals = self._signals(results, questions)
        except ClassifierError as e:
            await self.push_error(f"{self}: {e}", e)
        return signals

    async def _audio_reading(self) -> list[PerceptionSignal] | None:
        """The turn's latest audio reading, waited for up to the timeout, or None."""
        read = self._audio_read
        if not read:
            return None
        # A wait that neither cancels the read when it's late nor fails when
        # the turn's end cancels it.
        done, _ = await asyncio.wait({read}, timeout=self._timeout)
        if not done:
            logger.warning(f"{self}: no reading after {self._timeout}s, the turn passes unread")
        return read.result() if done and not read.cancelled() else None

    async def _end_turn(self):
        """Drop the turn's audio and stop reading it."""
        self._speech_buffer.stop()
        if self._audio_read and not self._audio_read.done():
            await self.cancel_task(self._audio_read)
        self._audio_read = None

    def _classifier_questions(
        self, questions: Mapping[str, PerceptionQuestion]
    ) -> dict[str, ClassifierQuestion]:
        """The questions as the classifier is asked them."""
        return {name: question.question for name, question in questions.items()}

    def _signals(
        self, results: Mapping[str, ClassifierResult], questions: Mapping[str, PerceptionQuestion]
    ) -> list[PerceptionSignal]:
        """The answers as the signals they report, in the questions' order."""
        return [self._signal(name, results[name], questions[name]) for name in questions]

    def _signal(
        self, name: str, result: ClassifierResult, question: PerceptionQuestion
    ) -> PerceptionSignal:
        """One answer as the signal it reports."""
        if isinstance(result, ChoiceResult):
            likeliest = sorted(result.probabilities.items(), key=lambda p: -p[1])
            signal = PerceptionSignal(name, labels=likeliest[: question.max_labels])
        elif isinstance(result, ScoreResult):
            signal = PerceptionSignal(name, value=round(result.score, 2))
        else:
            signal = PerceptionSignal(name, value=round(result.probability, 2))
        return signal

    async def _on_classifier_metrics(self, classifier: BaseClassifier, data: list[MetricsData]):
        """Push the classifier's metrics into the pipeline, when metrics are on."""
        if self.metrics_enabled:
            await self.push_frame(MetricsFrame(data=data))
