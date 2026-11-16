"""
Drop-in Python Client SDK for openfal (`fal_client` compatible API).

Usage:
    import openfal

    result = openfal.subscribe(
        "minimax/h3-max/text-to-video",
        arguments={
            "prompt": "A neon-lit cyberpunk alley in the rain, stereo thunder and rain audio",
            "duration": 5,
            "resolution": "768P",
        },
        with_logs=True,
        on_queue_update=lambda status: print(status),
    )
    print(result["video"]["url"])
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import httpx


class OpenFalClient:
    def __init__(self, base_url: Optional[str] = None, timeout: float = 120.0):
        self.base_url = (base_url or os.environ.get("OPENFAL_BASE_URL", "http://localhost:8000")).rstrip("/")
        self.timeout = timeout

    def run(self, endpoint_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ep = endpoint_id.lstrip("/")
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.post(f"/{ep}", json=arguments)
            resp.raise_for_status()
            return resp.json()

    def submit(self, endpoint_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ep = endpoint_id.lstrip("/")
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.post(f"/queue/{ep}", json=arguments)
            resp.raise_for_status()
            return resp.json()

    def status(self, endpoint_id: str, request_id: str, with_logs: bool = True) -> Dict[str, Any]:
        ep = endpoint_id.lstrip("/")
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.get(f"/{ep}/requests/{request_id}/status", params={"logs": int(with_logs)})
            resp.raise_for_status()
            return resp.json()

    def result(self, endpoint_id: str, request_id: str) -> Dict[str, Any]:
        ep = endpoint_id.lstrip("/")
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.get(f"/{ep}/requests/{request_id}")
            resp.raise_for_status()
            return resp.json()

    def subscribe(
        self,
        endpoint_id: str,
        arguments: Dict[str, Any],
        with_logs: bool = True,
        on_queue_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        poll_interval: float = 0.1,
    ) -> Dict[str, Any]:
        queued = self.submit(endpoint_id, arguments)
        req_id = queued["request_id"]
        seen_logs = 0

        while True:
            st = self.status(endpoint_id, req_id, with_logs=with_logs)
            if on_queue_update is not None:
                all_logs = st.get("logs") or []
                delta = dict(st)
                delta["logs"] = all_logs[seen_logs:]
                seen_logs = len(all_logs)
                on_queue_update(delta)
            if st["status"] == "COMPLETED":
                return self.result(endpoint_id, req_id)
            if st["status"] in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"Request {req_id} ended with status {st['status']}: {st.get('error')}")
            time.sleep(poll_interval)

    def upload_file(self, path: str | Path) -> str:
        p = Path(path)
        content_type = "video/mp4" if p.suffix.lower() == ".mp4" else "image/png"
        with httpx.Client(base_url=self.base_url, timeout=self.timeout) as client:
            resp = client.post(
                "/v1/storage/upload",
                content=p.read_bytes(),
                headers={"Content-Type": content_type},
            )
            resp.raise_for_status()
            return resp.json()["url"]


_default_client = OpenFalClient()


def run(endpoint_id: str, arguments: Dict[str, Any], base_url: Optional[str] = None) -> Dict[str, Any]:
    client = OpenFalClient(base_url=base_url) if base_url else _default_client
    return client.run(endpoint_id, arguments)


def submit(endpoint_id: str, arguments: Dict[str, Any], base_url: Optional[str] = None) -> Dict[str, Any]:
    client = OpenFalClient(base_url=base_url) if base_url else _default_client
    return client.submit(endpoint_id, arguments)


def status(
    endpoint_id: str,
    request_id: str,
    with_logs: bool = True,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    client = OpenFalClient(base_url=base_url) if base_url else _default_client
    return client.status(endpoint_id, request_id, with_logs=with_logs)


def result(endpoint_id: str, request_id: str, base_url: Optional[str] = None) -> Dict[str, Any]:
    client = OpenFalClient(base_url=base_url) if base_url else _default_client
    return client.result(endpoint_id, request_id)


def subscribe(
    endpoint_id: str,
    arguments: Dict[str, Any],
    with_logs: bool = True,
    on_queue_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    client = OpenFalClient(base_url=base_url) if base_url else _default_client
    return client.subscribe(
        endpoint_id=endpoint_id,
        arguments=arguments,
        with_logs=with_logs,
        on_queue_update=on_queue_update,
    )


def upload_file(path: str | Path, base_url: Optional[str] = None) -> str:
    client = OpenFalClient(base_url=base_url) if base_url else _default_client
    return client.upload_file(path)
