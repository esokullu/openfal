"""
FastAPI Server for openfal:
- fal.ai-Compatible REST & Queue/SSE Endpoints for MiniMax H3 (`minimax/h3/*`, `minimax/h3-max/*`, `minimax/h3-image/*`)
- Strata-DiT 3-Tier Memory Hierarchy Telemetry & Control Endpoints (`/v1/engine/*`)
- Built-in Interactive Studio Web UI (`GET /`)
"""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from openfal.config import EngineConfig, HARDWARE_PROFILES, get_default_config
from openfal.engine.runner import OpenFalRunner
from openfal.server.queue_manager import FalQueueManager
from openfal.server.schemas import MiniMaxH3Input, ModelDownloadRequest, ProfileSwitchRequest


SUPPORTED_ENDPOINTS = [
    {
        "id": "minimax/h3/text-to-video",
        "alias": "fal-ai/minimax/h3/text-to-video",
        "partition": "FL2VA",
        "category": "Text-to-Video",
        "description": "Generate 768P-2K video with synchronized native stereo audio from text prompt.",
    },
    {
        "id": "minimax/h3-max/text-to-video",
        "alias": "fal-ai/minimax/h3-max/text-to-video",
        "partition": "FL2VA",
        "category": "Text-to-Video",
        "description": "Post-trained MiniMax H3 Max text-to-video with prompt expansion & stereo audio.",
    },
    {
        "id": "minimax/h3/image-to-video",
        "alias": "fal-ai/minimax/h3/image-to-video",
        "partition": "FL2VA",
        "category": "Image-to-Video",
        "description": "Animate a starting (and optional ending) image into a video with stereo audio.",
    },
    {
        "id": "minimax/h3-max/image-to-video",
        "alias": "fal-ai/minimax/h3-max/image-to-video",
        "partition": "FL2VA",
        "category": "Image-to-Video",
        "description": "High-fidelity H3 Max first/last-frame conditioned image-to-video.",
    },
    {
        "id": "minimax/h3/reference-to-video",
        "alias": "fal-ai/minimax/h3/reference-to-video",
        "partition": "Ref2VA",
        "category": "Reference-to-Video",
        "description": "Omni-reference video generation using up to 9 reference images, 3 videos & audio.",
    },
    {
        "id": "minimax/h3-max/reference-to-video",
        "alias": "fal-ai/minimax/h3-max/reference-to-video",
        "partition": "Ref2VA",
        "category": "Reference-to-Video",
        "description": "H3 Max subject/identity-consistent multi-reference video generation.",
    },
    {
        "id": "minimax/h3-max/recast",
        "alias": "fal-ai/minimax/h3-max/recast",
        "partition": "Ref2VA",
        "category": "Video Face Swap / Recast",
        "description": "Video face swap & character identity replacement preserving target motion and lighting.",
    },
    {
        "id": "minimax/h3-max/relight",
        "alias": "fal-ai/minimax/h3-max/relight",
        "partition": "Ref2VA",
        "category": "Video Editing",
        "description": "Relight existing video footage with text & reference lighting conditions.",
    },
    {
        "id": "minimax/h3-max/insert-video",
        "alias": "fal-ai/minimax/h3-max/insert-video",
        "partition": "Ref2VA",
        "category": "Video Editing",
        "description": "Insert reference characters or objects seamlessly into a target video scene.",
    },
    {
        "id": "minimax/h3/text-to-image",
        "alias": "fal-ai/minimax/h3/text-to-image",
        "partition": "H3-Image",
        "category": "Text-to-Image",
        "description": "Multi-megapixel (up to 4096x4096) native image generation using MiniMax-H3-Image.",
    },
    {
        "id": "minimax/h3/image-to-image",
        "alias": "fal-ai/minimax/h3/image-to-image",
        "partition": "H3-Image",
        "category": "Image-to-Image / Face Edit",
        "description": "Reference-guided image editing, style transfer, and portrait face swap.",
    },
]


def _normalize_endpoint_id(raw_path: str) -> str:
    cleaned = raw_path.strip("/")
    for prefix in ("queue/", "fal-ai/"):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    if cleaned.startswith("fal-ai/"):
        cleaned = cleaned[len("fal-ai/"):]
    return cleaned


def _extract_input_payload(raw_body: Dict[str, Any]) -> Dict[str, Any]:
    # @fal-ai/client sometimes wraps parameters inside {"input": {...}}
    if isinstance(raw_body.get("input"), dict):
        merged = {k: v for k, v in raw_body.items() if k != "input"}
        merged.update(raw_body["input"])
        return merged
    return raw_body


def _base_url_from_request(request: Request) -> str:
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host", request.url.netloc)
    return f"{scheme}://{host}"


def create_app(config: Optional[EngineConfig] = None) -> FastAPI:
    cfg = config or get_default_config()
    runner = OpenFalRunner(cfg)
    queue_mgr = FalQueueManager(runner)

    app = FastAPI(
        title="openfal — Strata-DiT Inference Engine for MiniMax H3",
        description=(
            "Strata-inspired 3-tier VRAM/RAM/NVMe offload inference engine for MiniMax H3 "
            "(Text-to-Video, Image-to-Video, Text-to-Image, Video Face Swap / Recast) "
            "with drop-in fal.ai API compatibility and Studio Web UI."
        ),
        version="0.1.0",
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.config = cfg
    app.state.runner = runner
    app.state.queue_mgr = queue_mgr

    # -------------------------------------------------------------------------
    # Web UI & Health
    # -------------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def serve_web_ui() -> HTMLResponse:
        ui_path = Path(__file__).resolve().parent.parent / "ui" / "index.html"
        if not ui_path.exists():
            return HTMLResponse("<h1>openfal UI not found</h1>", status_code=404)
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "engine": "openfal-strata-dit",
            "version": "0.1.0",
            "sleeping": runner.memory.sleeping,
            "active_profile": cfg.profile_id,
        }

    # -------------------------------------------------------------------------
    # Strata-DiT Engine Telemetry & Hardware Control Endpoints
    # -------------------------------------------------------------------------
    @app.get("/v1/engine/status")
    async def engine_status() -> dict:
        telemetry = runner.memory.get_telemetry()
        telemetry["text_encoder_cache"] = runner.text_cache.stats()
        return telemetry

    @app.get("/v1/engine/profiles")
    async def list_hardware_profiles() -> dict:
        return {
            "active_profile_id": cfg.profile_id,
            "profiles": [p.to_dict() for p in HARDWARE_PROFILES.values()],
        }

    @app.post("/v1/engine/profile")
    async def switch_hardware_profile(req: ProfileSwitchRequest) -> dict:
        try:
            updated = runner.memory.switch_profile(
                profile_id=req.profile_id,
                custom_resident_blocks=req.resident_vram_blocks,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "status": "updated",
            "profile": updated.to_dict(),
            "telemetry": runner.memory.get_telemetry(),
        }

    @app.post("/v1/engine/sleep")
    async def engine_sleep() -> dict:
        """Evict all GPU VRAM Counter weights to Pinned Host RAM (0 MB model VRAM while idle)."""
        return {"status": "sleeping", "telemetry": runner.memory.sleep()}

    @app.post("/v1/engine/wake")
    async def engine_wake() -> dict:
        """Restore pinned resident DiT blocks from Pinned Host RAM to GPU VRAM Counter."""
        return {"status": "awake", "telemetry": runner.memory.wake()}

    @app.get("/v1/endpoints")
    async def list_endpoints() -> dict:
        return {"endpoints": SUPPORTED_ENDPOINTS}

    @app.get("/v1/jobs")
    async def list_recent_jobs() -> dict:
        return {"jobs": queue_mgr.list_recent_jobs(limit=25)}

    # -------------------------------------------------------------------------
    # Strata-Style Hugging Face Model Manager & Hardware Auto-Detector Endpoints
    # -------------------------------------------------------------------------
    @app.get("/v1/models/status")
    async def models_status() -> dict:
        return runner.model_manager.get_all_status()

    @app.get("/v1/models/hardware-detect")
    async def models_hardware_detect() -> dict:
        return runner.model_manager.detect_hardware()

    @app.post("/v1/models/download")
    async def start_model_download(req: ModelDownloadRequest) -> dict:
        try:
            task = runner.model_manager.start_download_or_pack(
                component_id=req.component_id,
                quantization=req.quantization,
                mode=req.mode,
                hf_token=req.hf_token,
                custom_repo_id=req.custom_repo_id,
                run_async=req.run_async,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": task.status, "task": task.to_dict()}

    @app.delete("/v1/models/{component_id}")
    async def delete_model_component(component_id: str) -> dict:
        try:
            info = runner.model_manager.remove_component(component_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "removed", "component": info}

    # -------------------------------------------------------------------------
    # Local Media Storage & File Serving (compatible with fal.storage.upload)
    # -------------------------------------------------------------------------
    @app.post("/v1/storage/upload")
    async def upload_media(request: Request) -> dict:
        content_type = request.headers.get("content-type", "")
        ext = ".png"
        if "video" in content_type or "mp4" in content_type:
            ext = ".mp4"
        elif "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        elif "webp" in content_type:
            ext = ".webp"

        raw_bytes: bytes
        if "application/json" in content_type:
            body = await request.json()
            data_uri = body.get("data") or body.get("file") or ""
            if "," in data_uri:
                header, b64_part = data_uri.split(",", 1)
                if "video" in header:
                    ext = ".mp4"
                raw_bytes = base64.b64decode(b64_part)
            else:
                raw_bytes = base64.b64decode(data_uri)
        else:
            raw_bytes = await request.body()

        fname = f"upload_{uuid.uuid4().hex[:12]}{ext}"
        dest = cfg.output_dir / fname
        dest.write_bytes(raw_bytes)
        base = _base_url_from_request(request)
        return {
            "url": f"{base}/files/{fname}",
            "file_url": f"{base}/files/{fname}",
            "file_name": fname,
            "file_size": len(raw_bytes),
        }

    @app.get("/files/{filename}")
    async def serve_generated_file(filename: str) -> FileResponse:
        safe_name = Path(filename).name
        target = cfg.output_dir / safe_name
        if not target.exists():
            raise HTTPException(status_code=404, detail=f"File '{safe_name}' not found")
        media_type = "video/mp4" if safe_name.endswith(".mp4") else "image/png"
        return FileResponse(target, media_type=media_type, filename=safe_name)

    # -------------------------------------------------------------------------
    # fal.ai Queue Lifecycle Endpoints (Status, SSE Stream, Result, Cancel)
    # -------------------------------------------------------------------------
    @app.get("/requests/{request_id}/status")
    @app.get("/{prefix:path}/requests/{request_id}/status")
    async def get_queue_status(
        request_id: str,
        logs: int = Query(default=1),
        prefix: str = "",
    ) -> dict:
        job = queue_mgr.get_job(request_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Request '{request_id}' not found")
        return job.to_status_dict(include_logs=bool(logs))

    @app.get("/requests/{request_id}/status/stream")
    @app.get("/{prefix:path}/requests/{request_id}/status/stream")
    async def stream_queue_status(
        request_id: str,
        logs: int = Query(default=1),
        prefix: str = "",
    ) -> StreamingResponse:
        job = queue_mgr.get_job(request_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Request '{request_id}' not found")

        async def _event_generator():
            last_log_idx = 0
            while True:
                status_payload = job.to_status_dict(include_logs=bool(logs))
                if logs:
                    status_payload["logs"] = [l.model_dump() for l in job.logs[last_log_idx:]]
                    last_log_idx = len(job.logs)
                yield f"data: {json.dumps(status_payload)}\n\n"
                if job.status in ("COMPLETED", "FAILED", "CANCELLED"):
                    break
                await asyncio.sleep(0.08)

        return StreamingResponse(_event_generator(), media_type="text/event-stream")

    @app.get("/requests/{request_id}")
    @app.get("/{prefix:path}/requests/{request_id}")
    async def get_queue_result(request_id: str, prefix: str = "") -> JSONResponse:
        job = queue_mgr.get_job(request_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Request '{request_id}' not found")
        if job.status == "FAILED":
            raise HTTPException(status_code=500, detail=job.error or "Generation failed")
        if job.status != "COMPLETED" or job.result is None:
            return JSONResponse(status_code=202, content=job.to_status_dict(include_logs=True))
        return JSONResponse(status_code=200, content=job.result)

    @app.put("/requests/{request_id}/cancel")
    @app.put("/{prefix:path}/requests/{request_id}/cancel")
    async def cancel_queue_request(request_id: str, prefix: str = "") -> dict:
        job = queue_mgr.cancel_job(request_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Request '{request_id}' not found")
        return {"status": job.status, "request_id": request_id}

    # -------------------------------------------------------------------------
    # fal.ai Queued Submission (`POST /queue/{endpoint_id:path}`)
    # -------------------------------------------------------------------------
    @app.post("/queue/{endpoint_id:path}")
    async def submit_to_fal_queue(endpoint_id: str, request: Request) -> dict:
        raw_body = await request.json()
        payload = _extract_input_payload(raw_body)
        validated = MiniMaxH3Input(**payload).model_dump()
        norm_ep = _normalize_endpoint_id(endpoint_id)
        base_url = _base_url_from_request(request)
        job = queue_mgr.submit_job(
            endpoint_id=norm_ep,
            payload=validated,
            base_url=base_url,
            run_async=True,
        )
        return job.to_status_dict(include_logs=True)

    # -------------------------------------------------------------------------
    # Direct & Queue-Aware fal.ai Model Execution (`POST /minimax/h3/...`, `POST /fal-ai/...`)
    # -------------------------------------------------------------------------
    @app.post("/{endpoint_id:path}")
    async def execute_fal_endpoint(
        endpoint_id: str,
        request: Request,
        queue: bool = Query(default=False, description="Set true to enqueue and return request_id immediately."),
    ) -> Any:
        norm_ep = _normalize_endpoint_id(endpoint_id)
        if not norm_ep.startswith("minimax/"):
            raise HTTPException(
                status_code=404,
                detail=f"Unknown endpoint '{endpoint_id}'. See GET /v1/endpoints for supported MiniMax H3 routes.",
            )

        raw_body = await request.json()
        payload = _extract_input_payload(raw_body)
        validated = MiniMaxH3Input(**payload).model_dump()
        base_url = _base_url_from_request(request)

        # Support queue mode via query param `?queue=true` or header `x-fal-queue: true`
        is_queue_header = request.headers.get("x-fal-queue", "").lower() in ("1", "true", "yes")
        if queue or is_queue_header:
            job = queue_mgr.submit_job(
                endpoint_id=norm_ep,
                payload=validated,
                base_url=base_url,
                run_async=True,
            )
            return job.to_status_dict(include_logs=True)

        # Synchronous execution (still recorded in job history for Web UI visibility!)
        job = queue_mgr.submit_job(
            endpoint_id=norm_ep,
            payload=validated,
            base_url=base_url,
            run_async=False,
        )
        if job.status == "FAILED":
            raise HTTPException(status_code=500, detail=job.error or "Inference failed")
        return job.result

    return app
