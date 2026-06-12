"""
alerts.py router
─────────────────
GET /api/v1/alerts              — recent alerts (ring buffer)
GET /api/v1/alerts/stats        — alert statistics
GET /api/v1/alerts/{alert_id}   — single alert by ID
DELETE /api/v1/alerts           — clear alert buffer
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from src.api.schemas import AlertListResponse

router = APIRouter()


def get_alert_engine(request: Request):
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    return pipeline.alert_engine


@router.get(
    "/alerts",
    response_model=AlertListResponse,
    summary="Get recent intrusion alerts",
    description=(
        "Returns the most recent alerts from the in-memory ring buffer. "
        "Filter by camera_id for per-camera alert history."
    ),
)
async def list_alerts(
    limit: int = Query(50, ge=1, le=500, description="Maximum alerts to return"),
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    engine=Depends(get_alert_engine),
):
    alerts = engine.get_recent_alerts(limit=limit, camera_id=camera_id)
    return AlertListResponse(
        total=len(alerts),
        alerts=alerts,
        stats=engine.get_stats(),
    )


@router.get("/alerts/stats", summary="Get alert engine statistics")
async def alert_stats(engine=Depends(get_alert_engine)):
    """Returns total alerts fired, buffer size, webhook count."""
    return engine.get_stats()


@router.delete("/alerts", status_code=204, summary="Clear the alert buffer")
async def clear_alerts(engine=Depends(get_alert_engine)):
    """Clears in-memory alert buffer. Does not affect database records."""
    engine._alert_buffer.clear()
    return None


@router.post("/alerts/webhook", status_code=201, summary="Register an alert webhook")
async def register_webhook(
    url: str = Query(..., description="HTTP endpoint to receive alert POSTs"),
    engine=Depends(get_alert_engine),
):
    """
    Register an external HTTP endpoint to receive alert payloads.
    The system will POST alert JSON to this URL whenever an alert fires.
    """
    engine.register_webhook(url)
    return {"registered": url, "total_webhooks": len(engine._webhooks)}
