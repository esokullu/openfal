"""
fal.ai-Compatible Asynchronous Queue & Log Stream Manager for openfal.

Supports both:
- Direct synchronous execution (`fal.run` / `POST /minimax/h3/text-to-video`)
- Queue lifecycle (`fal.subscribe` / `queue.submit` -> `status` -> `result` + SSE `/status/stream`)
"""

from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from openfal.engine.runner import OpenFalRunner
from openfal.server.schemas import FalLogEntry


@dataclass
class QueuedJob:
    request_id: str
    endpoint_id: str
    payload: Dict[str, Any]
    base_url: str
    status: str = "IN_QUEUE"  # IN_QUEUE, IN_PROGRESS, COMPLETED, FAILED, CANCELLED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    logs: List[FalLogEntry] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    done_event: threading.Event = field(default_factory=threading.Event)

    def add_log(self, message: str, level: str = "INFO") -> None:
        ts = datetime.now(timezone.utc).isoformat()
        self.logs.append(FalLogEntry(message=message, level=level, timestamp=ts))

    def to_status_dict(self, include_logs: bool = True) -> dict:
        base = self.base_url.rstrip("/")
        ep = self.endpoint_id.strip("/")
        data: Dict[str, Any] = {
            "status": self.status,
            "request_id": self.request_id,
            "response_url": f"{base}/{ep}/requests/{self.request_id}",
            "status_url": f"{base}/{ep}/requests/{self.request_id}/status",
            "cancel_url": f"{base}/{ep}/requests/{self.request_id}/cancel",
        }
        if self.status == "IN_QUEUE":
            data["queue_position"] = 0
        if include_logs:
            data["logs"] = [log.model_dump() for log in self.logs]
        if self.started_at and self.completed_at:
            data["metrics"] = {"inference_time": round(self.completed_at - self.started_at, 3)}
        if self.error:
            data["error"] = self.error
        return data


class FalQueueManager:
    def __init__(self, runner: OpenFalRunner):
        self.runner = runner
        self._jobs: Dict[str, QueuedJob] = {}
        self._lock = threading.RLock()
        self._execution_lock = runner.execution_lock

    def submit_job(
        self,
        endpoint_id: str,
        payload: Dict[str, Any],
        base_url: str = "http://localhost:8000",
        run_async: bool = True,
    ) -> QueuedJob:
        req_id = str(uuid.uuid4())
        job = QueuedJob(
            request_id=req_id,
            endpoint_id=endpoint_id,
            payload=payload,
            base_url=base_url,
        )
        with self._lock:
            self._jobs[req_id] = job

        if run_async:
            worker = threading.Thread(target=self._execute_job, args=(job,), daemon=True)
            worker.start()
        else:
            self._execute_job(job)

        return job

    def _execute_job(self, job: QueuedJob) -> None:
        # Keep waiting requests IN_QUEUE and serialize every use of the shared models.
        with self._execution_lock:
            self._execute_exclusive(job)

    def _execute_exclusive(self, job: QueuedJob) -> None:
        # Cancellation uses _lock too. Checking and changing state must be one
        # transition, otherwise a cancellation can be overwritten by this worker.
        with self._lock:
            if job.status == "CANCELLED":
                job.done_event.set()
                return

            job.status = "IN_PROGRESS"
            job.started_at = time.time()
            job.add_log(f"Dispatching '{job.endpoint_id}' on openfal Strata-DiT engine")

        try:
            res = self.runner.execute(
                endpoint_id=job.endpoint_id,
                payload=job.payload,
                base_url=job.base_url,
                on_log=lambda msg: job.add_log(msg),
            )
            with self._lock:
                if job.status == "CANCELLED":
                    return
                job.result = res
                job.status = "COMPLETED"
                job.completed_at = time.time()
                job.add_log(f"Completed '{job.endpoint_id}' in {job.completed_at - job.started_at:.2f}s")
        except Exception as exc:
            with self._lock:
                if job.status == "CANCELLED":
                    return
                job.error = str(exc)
                job.status = "FAILED"
                job.completed_at = time.time()
                job.add_log(f"Generation failed: {exc}", level="ERROR")
        finally:
            job.done_event.set()

    def get_job(self, request_id: str) -> Optional[QueuedJob]:
        with self._lock:
            return self._jobs.get(request_id)

    def cancel_job(self, request_id: str) -> Optional[QueuedJob]:
        with self._lock:
            job = self._jobs.get(request_id)
            if job and job.status in ("IN_QUEUE", "IN_PROGRESS"):
                job.status = "CANCELLED"
                job.completed_at = time.time()
                job.add_log("Request cancelled by client", level="WARNING")
                job.done_event.set()
            return job

    def list_recent_jobs(self, limit: int = 20) -> List[dict]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
            return [
                {
                    **j.to_status_dict(include_logs=False),
                    "endpoint_id": j.endpoint_id,
                    "prompt": j.payload.get("prompt", ""),
                    "created_at": j.created_at,
                    "result": j.result,
                }
                for j in jobs
            ]
