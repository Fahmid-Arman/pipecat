#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Classifiers that answer typed questions about a clip of audio.

A :class:`SegmentedAudioClassifier` takes one clip, a :class:`ClassifierAudio`,
with each set of questions, and the state as what is known about it, such as a
transcript. It declares the clips it accepts, so whoever cuts them knows how
long they must be.
"""

import time
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from pipecat.audio.utils import create_file_resampler
from pipecat.classifiers.base_classifier import (
    BaseClassifier,
    ChoiceQuestion,
    ChoiceResult,
    ClassifierError,
    ClassifierImage,
    ClassifierQuestion,
    ClassifierResult,
    ScoreQuestion,
    ScoreResult,
    YesNoQuestion,
    YesNoResult,
)
from pipecat.metrics.metrics import LLMTokenUsage


class ClassifierAudio(BaseModel):
    """A clip of audio the questions are about.

    Parameters:
        audio: The clip, as 16-bit mono PCM.
        sample_rate: The clip's sample rate in Hz.
    """

    audio: bytes = Field(repr=False)
    sample_rate: int

    @property
    def duration_secs(self) -> float:
        """How long the clip is, in seconds."""
        return len(self.audio) / (2 * self.sample_rate)


class SegmentedAudioClassifier(BaseClassifier):
    """Answers typed questions about one clip of audio per call.

    Every call takes a clip as ``audio``, and the state as what is known about
    it. The classifier declares the clips it accepts: from
    :attr:`min_audio_secs` to :attr:`max_audio_secs` long, resampled to its
    :attr:`sample_rate` before it's sent. A clip of another length raises a
    :class:`~pipecat.classifiers.base_classifier.ClassifierError`, as does a
    call without one. Subclasses implement :meth:`_ask_segment`.

    Example::

        clip = ClassifierAudio(audio=pcm, sample_rate=16000)
        results = await classifier.yes_no(
            "A caller talking to a voice agent.",
            {"frustrated": YesNoQuestion(instructions="Does the caller sound frustrated?")},
            audio=clip,
        )
    """

    def __init__(self, **kwargs):
        """Initialize the classifier.

        Args:
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self._resampler = create_file_resampler()

    @property
    def min_audio_secs(self) -> float:
        """The shortest clip the classifier accepts, in seconds."""
        return 0.0

    @property
    def max_audio_secs(self) -> float | None:
        """The longest clip the classifier accepts, in seconds, or None for no limit."""
        return None

    @property
    def sample_rate(self) -> int | None:
        """The sample rate clips are sent at, or None to send them at their own."""
        return None

    async def ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        *,
        images: Sequence[ClassifierImage] | None = None,
        audio: ClassifierAudio | None = None,
    ) -> dict[str, ClassifierResult]:
        """Answer several questions about one clip of audio.

        Args:
            state: What is known about the clip, such as a transcript.
            questions: The questions, by name.
            images: Images the questions can look at, for a classifier that
                supports them.
            audio: The clip the questions are about.

        Returns:
            One result per question, by the same names, each of the type
            its question calls for.

        Raises:
            ClassifierError: If the answers could not be produced, or the clip
                is missing or of a length the classifier doesn't accept.
        """
        if audio is None:
            raise ClassifierError(f"{self} answers questions about a clip of audio")
        if images and not self.supports_images:
            raise ClassifierError(f"{self} cannot see images")
        clip = await self._clip(audio)
        started = time.perf_counter()
        results, usage = await self._ask_segment(state, questions, clip, list(images or []))
        await self._call_event_handler(
            "on_metrics", self._metrics(time.perf_counter() - started, usage)
        )
        return results

    async def yes_no(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, YesNoQuestion],
        *,
        images: Sequence[ClassifierImage] | None = None,
        audio: ClassifierAudio | None = None,
    ) -> dict[str, YesNoResult]:
        """Ask whether the clip meets each condition.

        Args:
            state: What is known about the clip.
            questions: The questions, by name.
            images: Images the questions can look at, for a classifier that
                supports them.
            audio: The clip the questions are about.

        Returns:
            How likely each answer is yes, by the same names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        results = await self.ask(state, questions, images=images, audio=audio)
        return self._typed(results, YesNoResult)

    async def choice(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ChoiceQuestion],
        *,
        images: Sequence[ClassifierImage] | None = None,
        audio: ClassifierAudio | None = None,
    ) -> dict[str, ChoiceResult]:
        """Ask which option fits the clip, for each question.

        Args:
            state: What is known about the clip.
            questions: The questions, by name.
            images: Images the questions can look at, for a classifier that
                supports them.
            audio: The clip the questions are about.

        Returns:
            The option that fits and how likely each one is, by the same
            names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        results = await self.ask(state, questions, images=images, audio=audio)
        return self._typed(results, ChoiceResult)

    async def score(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ScoreQuestion],
        *,
        images: Sequence[ClassifierImage] | None = None,
        audio: ClassifierAudio | None = None,
    ) -> dict[str, ScoreResult]:
        """Ask where the clip falls on each scale.

        Args:
            state: What is known about the clip.
            questions: The questions, by name.
            images: Images the questions can look at, for a classifier that
                supports them.
            audio: The clip the questions are about.

        Returns:
            The position on each scale and how likely each level is, by the
            same names.

        Raises:
            ClassifierError: If the answers could not be produced.
        """
        results = await self.ask(state, questions, images=images, audio=audio)
        return self._typed(results, ScoreResult)

    async def _ask(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        images: Sequence[ClassifierImage] = (),
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage | None]:
        """Refuse questions without a clip: audio classifiers answer in :meth:`_ask_segment`."""
        raise ClassifierError(f"{self} answers questions about a clip of audio")

    @abstractmethod
    async def _ask_segment(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, ClassifierQuestion],
        audio: ClassifierAudio,
        images: Sequence[ClassifierImage],
    ) -> tuple[dict[str, ClassifierResult], LLMTokenUsage | None]:
        """Answer the questions about a clip of an accepted length, at :attr:`sample_rate`."""
        pass

    async def _clip(self, audio: ClassifierAudio) -> ClassifierAudio:
        """The clip at :attr:`sample_rate`, or a ClassifierError if its length isn't accepted."""
        duration = audio.duration_secs
        if duration < self.min_audio_secs:
            raise ClassifierError(
                f"{self} needs at least {self.min_audio_secs:g}s of audio, got {duration:.2f}s"
            )
        if self.max_audio_secs is not None and duration > self.max_audio_secs:
            raise ClassifierError(
                f"{self} takes at most {self.max_audio_secs:g}s of audio, got {duration:.2f}s"
            )
        if self.sample_rate and audio.sample_rate != self.sample_rate:
            resampled = await self._resampler.resample(
                audio.audio, audio.sample_rate, self.sample_rate
            )
            audio = ClassifierAudio(audio=resampled, sample_rate=self.sample_rate)
        return audio
