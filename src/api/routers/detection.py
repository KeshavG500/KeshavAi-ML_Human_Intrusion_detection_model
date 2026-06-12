"""
detection.py router
────────────────────
POST /api/v1/detect          — single frame detection (upload image)
POST /api/v1/detect/base64   — single frame detection (base64 encoded)
GET  /api/v1/stats           — pipeline performance stats
"""

import base64
import io
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from src.api.schemas import FrameDetectResponse

router = APIRouter()


def get_pipeline(request: Request):
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Inference pipeline not initialized. Check model weights."
        )
    return pipeline


class Base64DetectRequest(BaseModel):
    image_b64: str
    camera_id: Optional[str] = "CAM_00"


@router.post(
    "/detect",
    response_model=FrameDetectResponse,
    summary="Detect intrusions in an uploaded image frame",
    description=(
        "Upload a JPEG/PNG frame. Returns detected persons, vehicles, "
        "and any triggered intrusion alerts with zone information."
    ),
)
async def detect_frame(
    request: Request,
    file: UploadFile = File(..., description="JPEG or PNG image frame"),
    camera_id: str = "CAM_00",
    pipeline=Depends(get_pipeline),
):
    """
    Upload a frame and get back:
    - All detected persons and vehicles with bounding boxes
    - Any active zone intrusion alerts
    - Inference latency in milliseconds
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, detail=f"Expected image file, got {file.content_type}")

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(400, detail="Could not decode image. Send valid JPEG/PNG.")

    try:
        result = await pipeline.process_single_frame(frame, camera_id=camera_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(500, detail=str(e))


@router.post(
    "/detect/base64",
    response_model=FrameDetectResponse,
    summary="Detect intrusions from a base64-encoded frame",
)
async def detect_base64(
    body: Base64DetectRequest,
    pipeline=Depends(get_pipeline),
):
    """Accepts base64-encoded JPEG/PNG frame. Useful for IoT/embedded clients."""
    try:
        img_bytes = base64.b64decode(body.image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode base64 image.")
    except Exception as e:
        raise HTTPException(400, detail=f"Invalid base64 image: {e}")

    try:
        result = await pipeline.process_single_frame(frame, camera_id=body.camera_id)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(500, detail=str(e))


@router.get(
    "/stats",
    summary="Get pipeline performance statistics",
)
async def get_stats(pipeline=Depends(get_pipeline)):
    """Returns FPS, inference latency, tracked objects, and alert counts."""
    return pipeline.get_stats()
