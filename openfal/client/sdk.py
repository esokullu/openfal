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
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
    ):
        self.base_url = (base_url or os.environ.get("OPENFAL_BASE_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout if timeout is not None else float(os.environ.get("OPENFAL_TIMEOUT", "120"))
        self.api_key = api_key if api_key is not None else (
            os.environ.get("OPENFAL_API_KEY") or os.environ.get("FAL_KEY") or None
        )

    def _headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Key {self.api_key}"
        if extra:
            headers.update(extra)
        return headers

    def run(self, endpoint_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        return self.subscribe(endpoint_id, arguments, with_logs=False)

    def submit(self, endpoint_id: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        ep = endpoint_id.lstrip("/")
        with httpx.Client(base_url=self.base_url, timeout=self.timeout, headers=self._headers()) as client:
            resp = client.post(f"/queue/{ep}", json=arguments)
            resp.raise_for_status()
            return resp.json()

    def status(self, endpoint_id: str, request_id: str, with_logs: bool = True) -> Dict[str, Any]:
        ep = endpoint_id.lstrip("/")
        with httpx.Client(base_url=self.base_url, timeout=self.timeout, headers=self._headers()) as client:
            resp = client.get(f"/{ep}/requests/{request_id}/status", params={"logs": int(with_logs)})
            resp.raise_for_status()
            return resp.json()

    def result(self, endpoint_id: str, request_id: str) -> Dict[str, Any]:
        ep = endpoint_id.lstrip("/")
        with httpx.Client(base_url=self.base_url, timeout=self.timeout, headers=self._headers()) as client:
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
                headers=self._headers({"Content-Type": content_type}),
            )
            resp.raise_for_status()
            return resp.json()["url"]


_default_client = OpenFalClient()


def _resolve_client(base_url: Optional[str], api_key: Optional[str]) -> OpenFalClient:
    if base_url or api_key is not None:
        return OpenFalClient(base_url=base_url, api_key=api_key)
    return _default_client


def run(
    endpoint_id: str,
    arguments: Dict[str, Any],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    return _resolve_client(base_url, api_key).run(endpoint_id, arguments)


def submit(
    endpoint_id: str,
    arguments: Dict[str, Any],
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    return _resolve_client(base_url, api_key).submit(endpoint_id, arguments)


def status(
    endpoint_id: str,
    request_id: str,
    with_logs: bool = True,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    return _resolve_client(base_url, api_key).status(endpoint_id, request_id, with_logs=with_logs)


def result(
    endpoint_id: str,
    request_id: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    return _resolve_client(base_url, api_key).result(endpoint_id, request_id)


def subscribe(
    endpoint_id: str,
    arguments: Dict[str, Any],
    with_logs: bool = True,
    on_queue_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    return _resolve_client(base_url, api_key).subscribe(
        endpoint_id=endpoint_id,
        arguments=arguments,
        with_logs=with_logs,
        on_queue_update=on_queue_update,
    )


def upload_file(
    path: str | Path,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    return _resolve_client(base_url, api_key).upload_file(path)
