"""
zones.py router
────────────────
GET    /api/v1/zones              — list all zones
POST   /api/v1/zones              — create a new zone
PUT    /api/v1/zones/{zone_id}    — update a zone
DELETE /api/v1/zones/{zone_id}    — delete a zone
GET    /api/v1/zones/{zone_id}    — get single zone details
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import ZoneCreate, ZoneOut

router = APIRouter()


def get_zone_manager(request: Request):
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")
    return pipeline.zone_manager


@router.get("/zones", response_model=List[ZoneOut], summary="List all surveillance zones")
async def list_zones(zm=Depends(get_zone_manager)):
    """Return all configured zones with their current restriction status."""
    result = []
    for zone in zm.zones.values():
        result.append(ZoneOut(
            zone_id=zone.zone_id,
            name=zone.name,
            camera_id=zone.camera_id,
            polygon=zone.polygon_pts.tolist(),
            restricted_hours={
                "start": zone.restricted_start.strftime("%H:%M") if zone.restricted_start else "00:00",
                "end":   zone.restricted_end.strftime("%H:%M")   if zone.restricted_end   else "23:59",
            } if zone.restricted_start else None,
            person_threshold=zone.person_threshold,
            vehicle_intrusion=zone.vehicle_intrusion,
            enabled=zone.enabled,
            boundary_lines=zone.boundary_lines,
            is_restricted_now=zone.is_restricted_now(),
        ))
    return result


@router.get("/zones/{zone_id}", response_model=ZoneOut, summary="Get zone details")
async def get_zone(zone_id: str, zm=Depends(get_zone_manager)):
    if zone_id not in zm.zones:
        raise HTTPException(404, f"Zone '{zone_id}' not found")
    zone = zm.zones[zone_id]
    return ZoneOut(
        zone_id=zone.zone_id,
        name=zone.name,
        camera_id=zone.camera_id,
        polygon=zone.polygon_pts.tolist(),
        person_threshold=zone.person_threshold,
        vehicle_intrusion=zone.vehicle_intrusion,
        enabled=zone.enabled,
        boundary_lines=zone.boundary_lines,
        is_restricted_now=zone.is_restricted_now(),
    )


@router.post("/zones", response_model=ZoneOut, status_code=201, summary="Create a new zone")
async def create_zone(body: ZoneCreate, zm=Depends(get_zone_manager)):
    """
    Dynamically add a zone at runtime — no restart required.
    The zone is immediately active for intrusion detection.
    """
    if body.zone_id in zm.zones:
        raise HTTPException(409, f"Zone '{body.zone_id}' already exists")
    cfg = body.model_dump()
    if cfg.get("restricted_hours"):
        cfg["restricted_hours"] = {
            "start": cfg["restricted_hours"]["start"],
            "end":   cfg["restricted_hours"]["end"],
        }
    zone = zm.add_zone(cfg)
    return ZoneOut(
        **body.model_dump(),
        is_restricted_now=zone.is_restricted_now(),
    )


@router.put("/zones/{zone_id}", response_model=ZoneOut, summary="Update an existing zone")
async def update_zone(zone_id: str, body: ZoneCreate, zm=Depends(get_zone_manager)):
    if zone_id not in zm.zones:
        raise HTTPException(404, f"Zone '{zone_id}' not found")
    zm.remove_zone(zone_id)
    cfg = body.model_dump()
    cfg["zone_id"] = zone_id
    zone = zm.add_zone(cfg)
    return ZoneOut(**body.model_dump(), is_restricted_now=zone.is_restricted_now())


@router.delete("/zones/{zone_id}", status_code=204, summary="Delete a zone")
async def delete_zone(zone_id: str, zm=Depends(get_zone_manager)):
    if zone_id not in zm.zones:
        raise HTTPException(404, f"Zone '{zone_id}' not found")
    zm.remove_zone(zone_id)
    return None
