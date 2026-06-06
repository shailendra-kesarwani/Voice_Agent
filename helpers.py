from pipecat.processors.frameworks.langchain import LangchainProcessor
import time
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables import Runnable
import math, struct
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.frames.frames import (
    Frame,
    AudioRawFrame,
    # InterimTranscriptionFrame,
    TranscriptionFrame,
    TextFrame,
    # StartInterruptionFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    # LLMResponseEndFrame,
    # LLMResponseStartFrame,
    EndFrame,
    StartFrame,
    # TTSStoppedFrame,
    MetricsFrame
)
# from pipecat.processors.aggregators.llm_response import (
#     LLMAssistantResponseAggregator,
#     LLMUserResponseAggregator,
# )
from loguru import logger
from typing import List, Union

class LangchainRAGProcessor(LangchainProcessor):
    def __init__(self, chain: Runnable, transcript_key: str = "input"):
        super().__init__(chain, transcript_key)  
        self._chain = chain
        self._transcript_key = transcript_key

    @staticmethod
    def __get_token_value(text: Union[str, AIMessageChunk]) -> str:
        match text:
            case str():
                return text
            case AIMessageChunk():
                return text.content
            case dict() as d if 'answer' in d:
                return d['answer']
            case _:
                return ""
            
    async def _ainvoke(self, text: str):
        logger.debug(f"Invoking chain with {text}")
        targetPhrases = [
          "you can continue with the lecture",
          "continue with the lecture",
          "you can continue with lecture",
          "continue with lecture",
          "play the video",
          "continue with the video"
        ]

        ##Simple fuzzy matching by checking if the target phrase is included in the transcript text
        matchFound = any(phrase in text for phrase in targetPhrases)
        if matchFound:
            print("Fuzzy match found for the phrase: 'You can continue with the lecture'")
            return
        
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            async for token in self._chain.astream(
                {self._transcript_key: text},
                config={"configurable": {"session_id": self._participant_id}},
            ):
                await self.push_frame(StartFrame())
                await self.push_frame(TextFrame(self.__get_token_value(token)))
                await self.push_frame(EndFrame())
        except GeneratorExit:
            logger.warning(f"{self} generator was closed prematurely")
        except Exception as e:
            logger.exception(f"{self} an unknown error occurred: {e}")
        finally:
            await self.push_frame(LLMFullResponseEndFrame())

class TranscriptionTimingLogger(FrameProcessor):
    def __init__(self, avt):
        super().__init__()
        self.name = "Transcription"
        self._avt = avt

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, value):
        """The setter for the name property."""
        # You can add validation or other logic here before setting the value
        self._name = value

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        try:
            await super().process_frame(frame, direction)
            if isinstance(frame, TranscriptionFrame):
                elapsed = time.time() - self._avt.last_transition_ts
                logger.debug(f"Transcription TTF: {elapsed}")
                await self.push_frame(MetricsFrame(ttfb={self.name: elapsed}))

            await self.push_frame(frame, direction)
        except Exception as e:
            logger.debug(f"Exception {e}")

class AudioVolumeTimer(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.last_transition_ts = 0
        self._prev_volume = -80
        self._speech_volume_threshold = -50

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            volume = self.calculate_volume(frame)
            # print(f"Audio volume: {volume:.2f} dB")
            if (
                volume >= self._speech_volume_threshold
                and self._prev_volume < self._speech_volume_threshold
            ):
                # logger.debug("transition above speech volume threshold")
                self.last_transition_ts = time.time()
            elif (
                volume < self._speech_volume_threshold
                and self._prev_volume >= self._speech_volume_threshold
            ):
                # logger.debug("transition below non-speech volume threshold")
                self.last_transition_ts = time.time()
            self._prev_volume = volume

        await self.push_frame(frame, direction)

    def calculate_volume(self, frame: AudioRawFrame) -> float:
        if frame.num_channels != 1:
            raise ValueError(f"Expected 1 channel, got {frame.num_channels}")

        # Unpack audio data into 16-bit integers
        fmt = f"{len(frame.audio) // 2}h"
        audio_samples = struct.unpack(fmt, frame.audio)

        # Calculate RMS
        sum_squares = sum(sample**2 for sample in audio_samples)
        rms = math.sqrt(sum_squares / len(audio_samples))

        # Convert RMS to decibels (dB)
        # Reference: maximum value for 16-bit audio is 32767
        if rms > 0:
            db = 20 * math.log10(rms / 32767)
        else:
            db = -96  # Minimum value (almost silent)

        return db