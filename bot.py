#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os
import sys

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.serializers.exotel import ExotelFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import Frame
from pipecat.frames.frames import (TranscriptionFrame,TransportMessageUrgentFrame, TextFrame)
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


async def run_bot(
    websocket_client,
    stream_sid: str,
    call_sid: str,
):
    serializer = ExotelFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=SileroVADAnalyzer(),
            serializer=serializer,
        ),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.2)

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    # )
    # tts = DeepgramTTSService(
    #     api_key=os.getenv("DEEPGRAM_API_KEY"),
    #     model='nova-2',
    #     # voice="aura-2-andromeda-en",
    #     # sample_rate=24000,
    #     # encoding="linear16"
    # )
    tts= ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="ODq5zmih8GrVes37Dizd",
        # voice_id = "21m00Tcm4TlvDq8ikWAM",
        # model="eleven_flash_v2_5",
        # output_format="pcm_16000",
        # params=ElevenLabsTTSService.InputParams(
        #         language=Language.EN # Specify the language
        #     )
    )
    # tts = DeepgramTTSService(
    #     api_key=os.getenv("DEEPGRAM_API_KEY"),
    #     model="aura-asteria-en",
    # )
    # tts = OpenAITTSService(
    #     api_key=os.getenv("OPENAI_API_KEY"),
        # voice="nova",
        # model="gpt-4o-mini-tts",
        # instructions="Speak enthusiastically about any topics, but use a calm tone for sensitive subjects",
    # )
    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in an audio call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Websocket input from client
            stt,  # Speech-To-Text
            # user_send,
            context_aggregator.user(),
            llm,  # LLM
            # robot_send,
            tts,  # Text-To-Speech
            transport.output(),  # Websocket output to client
            context_aggregator.assistant(),
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            # audio_out_sample_rate=8000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,  # Default: True (strongly recommended)
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)

    await runner.run(task)
