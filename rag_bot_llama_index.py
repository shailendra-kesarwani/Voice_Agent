import asyncio, uvicorn
import os, json, chromadb

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.frames.frames import Frame, StartInterruptionFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.services.ai_service import AIService
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from pipecat.serializers.exotel import ExotelFrameSerializer
from fastapi import FastAPI, WebSocket
from pipecat.serializers.protobuf import ProtobufFrameSerializer
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport
from llama_index.core import VectorStoreIndex
# from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv(override=True)

app = FastAPI()
persist_directory = "./car-service-vectorestore"

class RAGService(AIService):
    def __init__(self, context: OpenAILLMContext, **kwargs):
        super().__init__(**kwargs)
        self.context = context
        self.current_rag_task: asyncio.Task[None] | None = None

        # Done in a hacky way via llama_index, can be implemented with native embedding model provider client
        # Additional dependencies:
        #   "llama-index>=0.12.38",
        #   "llama-index-embeddings-openai>=0.3.1"
        # dummy_docs = [
        #     Document(text="Weather in San Francisco is sunny"),
        #     Document(text="Weather in New York is rainy"),
        #     Document(text="Weather in London is cloudy"),
        #     Document(text="Weather in Paris is sunny"),
        #     Document(text="Weather in Tokyo is rainy"),
        #     Document(text="Weather in Sydney is sunny"),
        #     Document(text="Weather in Berlin is cloudy"),
        #     Document(text="Weather in Rome is sunny"),
        # ]

        db2 = chromadb.PersistentClient(path="./car-service-vectorestore-llama-index")
        chroma_collection = db2.get_or_create_collection("quickstart")
        # set up ChromaVectorStore and load in data
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        embedding_model = OpenAIEmbedding(model="text-embedding-3-small")
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=embedding_model,
        )

        self._similarity_threshold = 0.7
        self.retriver = vector_index.as_retriever(similarity_top_k=1, embedding_model=embedding_model)

    def _cancel_current_rag_task(self) -> None:
        """Cancel the RAG task if it is running."""
        if self.current_rag_task:
            try:
                self.current_rag_task.cancel()
                self.current_rag_task = None
                logger.info("RAG task canceled successfully")
            except Exception as e:
                logger.error(f"Error canceling RAG task: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process the frame and add the relevant documents to the context as a system message.
        Args:
            frame (Frame): The frame to process.
            direction (FrameDirection): The direction of the frame.
        """
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            # if a new transcription is detected, cancel the RAG task and start a new one
            logger.info(f"User transcription: {frame.text=}")
            self._cancel_current_rag_task()
            self.current_rag_task = asyncio.create_task(self._rag_task(frame))
            await self.current_rag_task
            await self.push_frame(frame, direction)
        elif isinstance(frame, StartInterruptionFrame):
            # if the user starts speaking again, cancel the RAG task
            self._cancel_current_rag_task()
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

    async def _rag_task(self, frame: TranscriptionFrame) -> None:
        """Query the Vector DB for the transcription and add the relevant documents to the context as a system message.
        Args:
            frame (TranscriptionFrame): The transcription frame.
        """
        logger.info(f"Querying Vector DB for transcription: {frame.text}")

        retrieved_nodes = await self.retriver.aretrieve(frame.text)
        retrieved_docs = [node.get_text() for node in retrieved_nodes if node.score > self._similarity_threshold]
        logger.info(
            f"Retrieved {len(retrieved_docs)} documents for transcription: {frame.text}"
        )
        if retrieved_docs:
            context_text = "\n".join(doc for doc in retrieved_docs)
            self.context.add_message(
                ChatCompletionSystemMessageParam(
                    content=context_text, name="RAG", role="system"
                )
            )
        logger.info("Rag task completed, cleaning up the current task.")
        self.current_rag_task = None

async def run_bot(websocket_client,
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
            serializer=serializer, #ProtobufFrameSerializer(),  # type: ignore
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            # vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            # vad_audio_passthrough=True,
            session_timeout=60 * 3,  # 3 minutes
        ),
    )
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # tts = CartesiaTTSService(
    #     api_key=os.getenv("CARTESIA_API_KEY"),
    #     voice_id="71a7ad14-091c-4e8e-a314-022ece01c121",  # British Reading Lady
    # )
    tts= ElevenLabsTTSService(
        # aiohttp_session=session,
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        # voice_id="ODq5zmih8GrVes37Dizd",
        voice_id = "21m00Tcm4TlvDq8ikWAM", #default voice rachel
        model="eleven_multilingual_v2",
        # model="eleven_flash_v2",
    )
    # tts = DeepgramTTSService(
    #     api_key=os.getenv("DEEPGRAM_API_KEY"),
    #     model='nova-2',
    #     # voice="aura-2-andromeda-en",
    #     # sample_rate=24000,
    #     # encoding="linear16"
    # )
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0.2)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            stt,  # STT,
            RAGService(context),  # RAG
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
            allow_interruptions=True,
            audio_in_sample_rate=8000,
            # audio_out_sample_rate=8000,
            audio_out_sample_rate=24000,
            report_only_initial_ttfb=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        # Start conversation - empty prompt to let LLM follow system instructions
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming audio data."""
    print("WebSocket connection established")
    await websocket.accept()
    # config = await websocket.receive()
    # print(f"Received config: {config}")
    start_data = websocket.iter_text()

    # Read first message (usually "connected")
    message = await start_data.__anext__()
    if json.loads(message)["event"] == "connected":
        logger.info(f"First message: {message}")
    message = await start_data.__anext__()
    # Read second message (usually "start" with call data)
    if json.loads(message)["event"] == "start":
        logger.info(f"Second message: {message}")
    if json.loads(message)["event"] in ["start", "media"]:
        try:
            call_data = json.loads(message)
            logger.info(f"Parsed call data: {call_data}")

            # Extract Exotel-specific data
            if call_data.get("event") == "start":
                start_data = call_data.get("start", {})
                stream_sid = start_data.get("stream_sid")
                call_sid = start_data.get("call_sid")
                custom_parameters = start_data.get("custom_parameters", {})

                logger.info(f"Stream ID: {stream_sid}")
                logger.info(f"Call SID: {call_sid}")
                logger.info(f"Custom Parameters: {custom_parameters}")

                # Exotel uses 8kHz PCM format
                await run_bot(websocket, stream_sid, call_sid)
            else:
                logger.info(f"Unexpected message format: {call_data}")

        except json.JSONDecodeError as e:
            logger.debug(f"Error parsing JSON: {e}")
        except Exception as e:
            logger.debug(f"Error handling WebSocket: {e}")
        

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
