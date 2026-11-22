"""
FastAPI Server for openfal:
- fal.ai-Compatible REST & Queue/SSE Endpoints for MiniMax H3 (`minimax/h3/*`, `minimax/h3-max/*`, `minimax/h3-image/*`)
- Strata-DiT 3-Tier Memory Hierarchy Telemetry & Control Endpoints (`/v1/engine/*`)
- Built-in Interactive Studio Web UI (`GET /`)
"""

from __future__ import annotations

import asyncio
import base64
import hmac
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from openfal.config import EngineConfig, HARDWARE_PROFILES, get_default_config
from openfal.engine.runner import OpenFalRunner
from openfal.server.queue_manager import FalQueueManager
from openfal.server.schemas import MiniMaxH3Input, ModelDownloadRequest, ProfileSwitchRequest


SUPPORTED_ENDPOINTS = [
    # -------------------------------------------------------------------------
    # Primary Fast MoE Stack: Alibaba Wan 2.2 MoE (27B-A14B) + Qwen-Image-2.1 (7B)
    # -------------------------------------------------------------------------
    {
        "id": "wan/v2.2/text-to-video",
        "alias": "fal-ai/wan/v2.2-a14b/text-to-video",
        "backend": "wan22_moe",
        "partition": "Wan2.2-T2V-A14B",
        "category": "Text-to-Video (Fast MoE)",
        "description": "Alibaba Wan 2.2 two-expert MoE text-to-video (27B total, 14B active per step).",
    },
    {
        "id": "wan/v2.2/image-to-video",
        "alias": "fal-ai/wan/v2.2-a14b/image-to-video",
        "backend": "wan22_moe",
        "partition": "Wan2.2-I2V-A14B",
        "category": "Image-to-Video (Fast MoE)",
        "description": "Alibaba Wan 2.2 Two-Expert MoE Image-to-Video (14B active per step).",
    },
    {
        "id": "wan/v2.2/animate",
        "alias": "fal-ai/wan/v2.2/animate",
        "backend": "wan22_moe",
        "partition": "Wan2.2-Animate-14B",
        "category": "Character Animation",
        "runtime_supported": True,
        "description": "Animate a reference character using a driving video's body pose and facial motion.",
    },
    {
        "id": "wan/v2.2/recast",
        "alias": "fal-ai/wan/v2.2/recast",
        "backend": "wan22_moe",
        "partition": "Wan2.2-Animate-14B",
        "category": "Video Character Replacement",
        "runtime_supported": True,
        "description": "Replace the driving video's character with the reference identity, using tracked SAM2 masks to preserve the background.",
    },
    {
        "id": "qwen/image-2.1/text-to-image",
        "alias": "fal-ai/qwen-image-2.1/turbo",
        "backend": "qwen_image_21",
        "partition": "Qwen-Image-2.1",
        "category": "Text-to-Image (Fast 7B Turbo)",
        "description": "Alibaba Qwen-Image-2.1-Turbo image generation (7B, 8-step checkpoint schedule).",
    },
    {
        "id": "qwen/image-2.1/edit",
        "alias": "fal-ai/qwen-image-2.1/edit",
        "backend": "qwen_image_21",
        "partition": "Qwen-Image-2.1",
        "category": "Image / Face Edit (10-Ref 7B Turbo)",
        "description": "Qwen-Image-2.1 multi-reference image and portrait editing (up to 10 images).",
    },
    # -------------------------------------------------------------------------
    # Optional Omni + Stereo Audio Stack: MiniMax H3 (33B Dense Single-Stream DiT)
    # -------------------------------------------------------------------------
    {
        "id": "minimax/h3/text-to-video",
        "alias": "fal-ai/minimax/h3/text-to-video",
        "backend": "minimax_h3",
        "partition": "FL2VA",
        "category": "Text-to-Video (MiniMax H3 Omni)",
        "description": "Generate video with native stereo audio from a text prompt; supported dimensions depend on available memory.",
    },
    {
        "id": "minimax/h3-max/text-to-video",
        "alias": "fal-ai/minimax/h3-max/text-to-video",
        "backend": "minimax_h3",
        "partition": "FL2VA",
        "category": "Text-to-Video (MiniMax H3 Max)",
        "description": "Compatibility alias for the local MiniMax-H3 FL2VA text-to-video checkpoint with stereo audio.",
    },
    {
        "id": "minimax/h3/image-to-video",
        "alias": "fal-ai/minimax/h3/image-to-video",
        "backend": "minimax_h3",
        "partition": "FL2VA",
        "category": "Image-to-Video (MiniMax H3 Omni)",
        "description": "Animate a starting (and optional ending) image into a video with stereo audio.",
    },
    {
        "id": "minimax/h3-max/image-to-video",
        "alias": "fal-ai/minimax/h3-max/image-to-video",
        "backend": "minimax_h3",
        "partition": "FL2VA",
        "category": "Image-to-Video (MiniMax H3 Max)",
        "description": "Compatibility alias for local MiniMax-H3 first/last-frame conditioned image-to-video.",
    },
    {
        "id": "minimax/h3/reference-to-video",
        "alias": "fal-ai/minimax/h3/reference-to-video",
        "backend": "minimax_h3",
        "partition": "Ref2VA",
        "category": "Reference-to-Video (MiniMax H3 Omni)",
        "description": "Omni-reference video generation using up to 9 reference images, 3 videos & audio.",
    },
    {
        "id": "minimax/h3-max/reference-to-video",
        "alias": "fal-ai/minimax/h3-max/reference-to-video",
        "backend": "minimax_h3",
        "partition": "Ref2VA",
        "category": "Reference-to-Video (MiniMax H3 Max)",
        "description": "Compatibility alias for local MiniMax-H3 Ref2VA multi-reference video generation.",
    },
    {
        "id": "minimax/h3-max/recast",
        "alias": "fal-ai/minimax/h3-max/recast",
        "backend": "minimax_h3",
        "partition": "Ref2VA",
        "category": "Reference Video Editing / Recast (MiniMax H3)",
        "description": "Generate a new video from a driving clip and identity references with Ref2VA. Exact target motion and scene preservation are not guaranteed.",
    },
    {
        "id": "minimax/h3-max/relight",
        "alias": "fal-ai/minimax/h3-max/relight",
        "backend": "minimax_h3",
        "partition": "Ref2VA",
        "category": "Video Editing (MiniMax H3 Max)",
        "description": "Generate a video using a driving clip, prompt and lighting references with Ref2VA; this is a generative edit.",
    },
    {
        "id": "minimax/h3-max/insert-video",
        "alias": "fal-ai/minimax/h3-max/insert-video",
        "backend": "minimax_h3",
        "partition": "Ref2VA",
        "category": "Video Editing (MiniMax H3 Max)",
        "description": "Generate a video using a driving clip and character/object references with Ref2VA; this is a generative edit.",
    },
    {
        "id": "minimax/h3/text-to-image",
        "alias": "fal-ai/minimax/h3/text-to-image",
        "backend": "minimax_h3",
        "partition": "H3-Image",
        "category": "Text-to-Image (MiniMax H3-Image 33B)",
        "description": "MiniMax-H3-Image generation using the installed six-step PDD checkpoint; maximum image size depends on available memory.",
    },
    {
        "id": "minimax/h3/image-to-image",
        "alias": "fal-ai/minimax/h3/image-to-image",
        "backend": "minimax_h3",
        "partition": "H3-Image",
        "category": "Image-to-Image / Face Edit (MiniMax H3-Image)",
        "description": "Reference-guided image editing, style transfer, and portrait editing.",
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


def _validate_generation_payload(raw_body: Dict[str, Any]) -> Dict[str, Any]:
    if "input" in raw_body and not isinstance(raw_body["input"], dict):
        raise RequestValidationError([{
            "type": "dict_type",
            "loc": ("body", "input"),
            "msg": "Input should be a valid dictionary",
            "input": raw_body["input"],
        }])
    payload = _extract_input_payload(raw_body)
    try:
        return MiniMaxH3Input.model_validate(payload).model_dump()
    except ValidationError as exc:
        errors = exc.errors(include_url=False)
        wrapped_fields = raw_body.get("input", {})
        for error in errors:
            loc = error["loc"]
            prefix = ("body", "input") if loc and loc[0] in wrapped_fields else ("body",)
            error["loc"] = prefix + loc
        raise RequestValidationError(errors, body=raw_body) from exc


def _base_url_from_request(request: Request) -> str:
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host", request.url.netloc)
    return f"{scheme}://{host}"


_PUBLIC_PATHS = frozenset({"/", "/health", "/docs", "/docs/oauth2-redirect", "/openapi.json", "/redoc"})


def _extract_request_api_key(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "").strip()
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() in ("key", "bearer"):
            return parts[1].strip()
    header_key = request.headers.get("x-api-key", "").strip()
    if header_key:
        return header_key
    query_key = request.query_params.get("api_key", "").strip()
    return query_key or None


def create_app(config: Optional[EngineConfig] = None) -> FastAPI:
    cfg = config or get_default_config()
    runner = OpenFalRunner(cfg)
    queue_mgr = FalQueueManager(runner)

    app = FastAPI(
        title="openfal — Local fal-Compatible Inference Server (Qwen-Image-2.1, Wan 2.2 & MiniMax H3)",
        description=(
            "Local PyTorch inference server for Qwen-Image-2.1, Wan 2.2, and MiniMax H3 "
            "with a fal-compatible REST/queue/SSE API, Python SDK, and built-in Studio Web UI."
        ),
        version="0.1.0",
        license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    )

    if cfg.api_key:
        expected_key = cfg.api_key

        @app.middleware("http")
        async def api_key_auth_middleware(request: Request, call_next):
            if request.method == "OPTIONS" or request.url.path in _PUBLIC_PATHS:
                return await call_next(request)
            provided = _extract_request_api_key(request)
            if not provided or not hmac.compare_digest(provided, expected_key):
                return JSONResponse(
                    status_code=401,
                    content={
                        "detail": (
                            "Missing or invalid API key. Supply 'Authorization: Key <key>', "
                            "'Authorization: Bearer <key>', or 'X-API-Key: <key>'."
                        )
                    },
                )
            return await call_next(request)

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

    @app.get("/memory-profile", response_class=HTMLResponse)
    async def serve_memory_profile() -> HTMLResponse:
        page_path = Path(__file__).resolve().parent.parent / "ui" / "memory_profile.html"
        if not page_path.exists():
            return HTMLResponse("<h1>Memory profile page not found</h1>", status_code=404)
        return HTMLResponse(page_path.read_text(encoding="utf-8"))

    @app.get("/health")
    async def health() -> dict:
        from openfal.runtime.common import use_synthetic_fallback

        return {
            "status": "ok",
            "engine": "openfal-strata-dit",
            "version": "0.1.0",
            "sleeping": runner.memory.sleeping,
            "active_profile": cfg.profile_id,
            "synthetic_mode": use_synthetic_fallback(),
            "auth_enabled": bool(cfg.api_key),
        }

    # -------------------------------------------------------------------------
    # Strata-DiT Engine Telemetry & Hardware Control Endpoints
    # -------------------------------------------------------------------------
    @app.get("/v1/engine/status")
    async def engine_status() -> dict:
        telemetry = runner.memory.get_telemetry()
        telemetry["telemetry_kind"] = "profile_estimate"
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
        def switch_exclusive():
            with runner.execution_lock:
                if req.profile_id not in HARDWARE_PROFILES:
                    raise ValueError(f"Unknown hardware profile '{req.profile_id}'")
                runner.evict()
                updated = runner.memory.switch_profile(
                    profile_id=req.profile_id,
                    custom_resident_blocks=req.resident_vram_blocks,
                )
                # Block residency is computed when loading, so rebuild for this profile.
                runner._real_qwen_runner = None
                runner._real_wan_runner = None
                runner._real_wan_animate_runner = None
                runner._real_minimax_runner = None
                return updated

        try:
            updated = await asyncio.to_thread(switch_exclusive)
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
        def sleep_exclusive():
            with runner.execution_lock:
                runner.evict()
                return runner.memory.sleep()

        return {"status": "sleeping", "telemetry": await asyncio.to_thread(sleep_exclusive)}

    @app.post("/v1/engine/wake")
    async def engine_wake() -> dict:
        """Restore pinned resident DiT blocks from Pinned Host RAM to GPU VRAM Counter."""
        def wake_exclusive():
            with runner.execution_lock:
                return runner.memory.wake()

        return {"status": "awake", "telemetry": await asyncio.to_thread(wake_exclusive)}

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
                source=req.source,
                cleanup_raw=req.cleanup_raw,
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
    async def submit_to_fal_queue(
        endpoint_id: str, request: Request, raw_body: Dict[str, Any] = Body(...),
    ) -> dict:
        validated = _validate_generation_payload(raw_body)
        norm_ep = _normalize_endpoint_id(endpoint_id)
        base_url = _base_url_from_request(request)
        job = await asyncio.to_thread(
            queue_mgr.submit_job,
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
        raw_body: Dict[str, Any] = Body(...),
        queue: bool = Query(default=False, description="Set true to enqueue and return request_id immediately."),
    ) -> Any:
        norm_ep = _normalize_endpoint_id(endpoint_id)
        if not norm_ep.startswith(("wan/", "qwen/", "qwen-image", "minimax/")):
            raise HTTPException(
                status_code=404,
                detail=f"Unknown endpoint '{endpoint_id}'. See GET /v1/endpoints for supported Wan 2.2 MoE, Qwen-Image-2.1, and MiniMax H3 routes.",
            )

        validated = _validate_generation_payload(raw_body)
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
        job = await asyncio.to_thread(
            queue_mgr.submit_job,
            endpoint_id=norm_ep,
            payload=validated,
            base_url=base_url,
            run_async=False,
        )
        if job.status == "FAILED":
            raise HTTPException(status_code=500, detail=job.error or "Inference failed")
        return job.result

    return app
