# import asyncio
import os
import sys
# import time

import aiohttp
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from loguru import logger
from pipecat.frames.frames import EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.audio.vad.silero import SileroVADAnalyzer
# from pipecat.vad.silero import SileroVADAnalyzer
# from pipecat.vad.vad_analyzer import VADParams
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.serializers.exotel import ExotelFrameSerializer
from deepgram import LiveOptions
from pipecat.transcriptions.language import Language
from helpers import (
    AudioVolumeTimer,
    TranscriptionTimingLogger,
    LangchainRAGProcessor,
    # ElevenLabsTurbo,
)

os.environ["SSL_CERT"] = ""
os.environ["SSL_KEY"] = ""
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")
load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

message_store = {}

embeddings = OpenAIEmbeddings()
persist_directory = "./car-service-vectorestore"

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in message_store:
        message_store[session_id] = ChatMessageHistory()
    return message_store[session_id]


async def run_bot(websocket_client,
    stream_sid: str,
    call_sid: str,
):
    serializer = ExotelFrameSerializer(
        stream_sid=stream_sid,
        call_sid=call_sid,
    )

    async with aiohttp.ClientSession() as session:
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

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))


        tts= ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id="ODq5zmih8GrVes37Dizd",
            # voice_id = "21m00Tcm4TlvDq8ikWAM", #default voice rachel
            # model="eleven_multilingual_v2",
            model="eleven_flash_v2",
        )


        # llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.2)
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        retriever = vectorstore.as_retriever()

        # context = OpenAILLMContext(messages)
        # context_aggregator = llm.create_context_aggregator(context)

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful LLM in an audio call. 
                        Your goal is to demonstrate your capabilities in a succinct way. 
                        Your output will be converted to audio so don't include special characters in your answers. 
                        Respond to what the user said in a creative and helpful way.{context}
                    """,
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, answer_prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        # chain = prompt | llm

        history_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            history_messages_key="chat_history",
            input_messages_key="input",
            output_messages_key="answer",
        )
        lc = LangchainRAGProcessor(chain=history_chain)

        avt = AudioVolumeTimer()
        tl = TranscriptionTimingLogger(avt)

        tma_in = LLMUserResponseAggregator()
        tma_out = LLMAssistantResponseAggregator()

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                avt,  # Audio volume timer
                stt,  # Speech-to-text
                tl,  # Transcription timing logger
                tma_in,  # User responses
                lc,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                tma_out,  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                # audio_in_sample_rate=8000,
                # audio_out_sample_rate=8000,
                # audio_out_sample_rate=24000,
                enable_metrics=True,
                report_only_initial_ttfb=True,
            )
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            # Kick off the conversation.
            # messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames(EndFrame())

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            await task.cancel()

        runner = PipelineRunner(handle_sigint=False)

        await runner.run(task)

