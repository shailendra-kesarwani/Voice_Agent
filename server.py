#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import json
import logging
import uvicorn
from loguru import logger
from bot import run_bot
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def health_check():
    """Health check endpoint - Exotel doesn't use XML webhooks"""
    # return {"url"}
    ret = {
        # "status": "Exotel bot ready",
        # "url": "wss://b8ffac94f8de.ngrok-free.app/ws",
        "url": "ws://localhost:8765/ws",
        # "note": "Configure this WebSocket URL in your Exotel App Bazaar Voicebot Applet",
    }
    print(ret)
    return ret

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

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
    # uvicorn.run(app, host="0.0.0.0", port=8765)
