"""
stream.py router
─────────────────
WebSocket /api/v1/ws/stream/{camera_id}  — live JSON frame events
WebSocket /api/v1/ws/alerts              — live alert-only events (Redis sub)
"""

import asyncio
import json
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections per camera."""

    def __init__(self):
        # Dict[camera_id → List[WebSocket]]
        self._connections: dict = {}

    async def connect(self, camera_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(camera_id, []).append(ws)
        logger.info(f"WebSocket connected: camera={camera_id} | "
                    f"total={len(self._connections.get(camera_id, []))}")

    def disconnect(self, camera_id: str, ws: WebSocket) -> None:
        conns = self._connections.get(camera_id, [])
        if ws in conns:
            conns.remove(ws)
        logger.info(f"WebSocket disconnected: camera={camera_id}")

    async def broadcast(self, camera_id: str, data: dict) -> None:
        conns = self._connections.get(camera_id, [])
        dead = []
        for ws in conns:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(camera_id, ws)


manager = ConnectionManager()


@router.websocket("/ws/stream/{camera_id}")
async def stream_camera(
    websocket: WebSocket,
    camera_id: str,
    request: Request = None,
):
    """
    WebSocket endpoint for real-time camera stream processing.

    Connect with: ws://host:8000/api/v1/ws/stream/{camera_id}

    Query params:
        source : RTSP URL or video file path (default: webcam index 0)

    Emits JSON frames:
    {
        "camera_id": "CAM_01",
        "personCount": 2,
        "vehicleCount": 0,
        "alerts": [...],
        "detections": [...],
        "fps": 14.8
    }
    """
    await websocket.accept()
    pipeline = getattr(websocket.app.state if hasattr(websocket, 'app') else
                       request.app.state if request else None, 'pipeline', None)

    if pipeline is None:
        await websocket.send_json({"error": "Pipeline not initialized"})
        await websocket.close()
        return

    # Get source from query params
    params = dict(websocket.query_params) if hasattr(websocket, 'query_params') else {}
    source = params.get("source", 0)
    try:
        source = int(source)   # webcam index
    except (ValueError, TypeError):
        pass   # keep as string (RTSP URL or file path)

    logger.info(f"WebSocket stream started: camera={camera_id} source={source}")

    try:
        async for frame_data in pipeline.process_frames_generator(
            source=source, camera_id=camera_id
        ):
            await websocket.send_json(frame_data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {camera_id}")
    except Exception as e:
        logger.error(f"WebSocket stream error [{camera_id}]: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/ws/alerts")
async def alert_stream(websocket: WebSocket, request: Request = None):
    """
    WebSocket endpoint that subscribes to Redis pub/sub for live alerts.
    All alert events from ALL cameras are broadcast here in real-time.

    Emits AlertOut JSON objects as they occur.
    """
    await websocket.accept()

    app_state = getattr(request, 'app', None)
    redis = getattr(app_state.state if app_state else None, 'redis', None)

    if redis is None:
        await websocket.send_json({"error": "Redis not available. Alerts disabled."})
        await websocket.close()
        return

    logger.info("WebSocket alert subscriber connected.")

    try:
        pubsub = redis.pubsub()
        await pubsub.subscribe("intrusion_alerts")

        async def redis_listener():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        await websocket.send_json(data)
                    except Exception as e:
                        logger.error(f"Alert relay error: {e}")

        async def ws_keepalive():
            while True:
                try:
                    await websocket.receive_text()
                except WebSocketDisconnect:
                    return

        await asyncio.gather(redis_listener(), ws_keepalive())

    except WebSocketDisconnect:
        logger.info("Alert subscriber disconnected.")
    except Exception as e:
        logger.error(f"Alert WebSocket error: {e}")
    finally:
        try:
            await pubsub.unsubscribe("intrusion_alerts")
            await websocket.close()
        except Exception:
            pass
