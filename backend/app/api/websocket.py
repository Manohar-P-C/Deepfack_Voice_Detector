"""
WebSocket endpoint for real-time audio analysis streaming.

Protocol:
    Client → Server:
        1. JSON control message: {"type": "start", "mode": "demo", "scenario": "genuine"}
        2. Binary PCM audio chunks (float32 little-endian, 16kHz mono)
        3. JSON control: {"type": "set_scenario", "scenario": "replay"}
        4. JSON control: {"type": "stop"}

    Server → Client:
        1. JSON: {"type": "session_start", ...}
        2. JSON: {"type": "analysis", ...AnalysisResult fields...}
        3. JSON: {"type": "session_end", ...}
        4. JSON: {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..core.trust_engine import TrustEngine
from ..config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Track active connections for metrics
active_sessions: Dict[str, TrustEngine] = {}
active_connections: Set[str] = set()


@router.websocket("/ws/analyze")
async def websocket_analyze(ws: WebSocket):
    """
    Main WebSocket endpoint for live audio analysis.

    The connection flows through three phases:
        1. Handshake: receive a 'start' message with mode/scenario config.
        2. Streaming: receive binary PCM chunks, process at update_interval,
           and send back AnalysisResult JSON.
        3. Teardown: on 'stop' message or disconnect, close the engine cleanly.
    """
    await ws.accept()
    engine: TrustEngine | None = None
    processing_task: asyncio.Task | None = None

    try:
        # ─── Phase 1: Wait for start message ───
        raw = await ws.receive_text()
        msg = json.loads(raw)

        if msg.get("type") != "start":
            await ws.send_json({"type": "error", "message": "First message must be type 'start'"})
            await ws.close()
            return

        mode = msg.get("mode", "demo")
        scenario = msg.get("scenario", "genuine")
        sample_rate = msg.get("sample_rate", settings.sample_rate)

        engine = TrustEngine.create(
            mode=mode,
            scenario=scenario,
            sample_rate=sample_rate,
        )
        active_sessions[engine.session_id] = engine
        active_connections.add(engine.session_id)

        # Send session start
        session_start = engine.get_session_start()
        await ws.send_json({
            "type": "session_start",
            **session_start.model_dump(mode="json"),
        })

        logger.info(f"WS session started: {engine.session_id} mode={mode}")

        # ─── Phase 2: Streaming loop ───
        async def process_and_send():
            """Periodic processing task that runs inference and sends results."""
            while True:
                await asyncio.sleep(settings.update_interval_ms / 1000.0)
                if engine is None or engine._closed:
                    break
                try:
                    results = await engine.process()
                    for result in results:
                        await ws.send_json({
                            "type": "analysis",
                            **result.model_dump(mode="json"),
                        })
                except Exception as e:
                    logger.error(f"Processing error: {e}", exc_info=True)
                    try:
                        await ws.send_json({
                            "type": "error",
                            "message": f"Processing error: {str(e)}",
                        })
                    except Exception:
                        break

        # Start the processing loop in the background
        processing_task = asyncio.create_task(process_and_send())

        # Receive audio chunks and control messages
        while True:
            message = await ws.receive()

            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"]:
                # Binary PCM audio chunk
                engine.ingest_chunk(message["bytes"], fmt="pcm_f32le")

            elif "text" in message and message["text"]:
                # JSON control message
                try:
                    ctrl = json.loads(message["text"])
                    ctrl_type = ctrl.get("type", "")

                    if ctrl_type == "stop":
                        break

                    elif ctrl_type == "set_scenario":
                        new_scenario = ctrl.get("scenario", "genuine")
                        engine.set_scenario(new_scenario)
                        await ws.send_json({
                            "type": "scenario_changed",
                            "scenario": new_scenario,
                        })

                    elif ctrl_type == "ping":
                        await ws.send_json({"type": "pong"})

                    else:
                        await ws.send_json({
                            "type": "error",
                            "message": f"Unknown control type: {ctrl_type}",
                        })
                except json.JSONDecodeError:
                    await ws.send_json({
                        "type": "error",
                        "message": "Invalid JSON in text message",
                    })

    except WebSocketDisconnect:
        logger.info(f"WS client disconnected: {engine.session_id if engine else 'unknown'}")
    except Exception as e:
        logger.error(f"WS error: {e}", exc_info=True)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        # ─── Phase 3: Teardown ───
        if processing_task and not processing_task.done():
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass

        if engine:
            session_end = engine.close()
            try:
                await ws.send_json({
                    "type": "session_end",
                    **session_end.model_dump(mode="json"),
                })
            except Exception:
                pass  # Connection may already be closed

            active_sessions.pop(engine.session_id, None)
            active_connections.discard(engine.session_id)
            logger.info(f"WS session ended: {engine.session_id}")

        try:
            await ws.close()
        except Exception:
            pass
