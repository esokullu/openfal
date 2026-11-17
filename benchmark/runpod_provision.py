"""Provision, bootstrap, run benchmarks on, and teardown RunPod GPU pods via REST API + SSH."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from benchmark.env_utils import load_dotenv

RUNPOD_REST_BASE = "https://rest.runpod.io/v1"
DEFAULT_STATE_FILE = Path(__file__).resolve().parent / ".runpod_pod.json"
DEFAULT_PORTS = ["22/tcp", "8000/http", "8188/http", "8091/http"]


def _require_api_key() -> str:
    key = (os.environ.get("RUNPOD_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "RUNPOD_API_KEY is missing. Copy benchmark/.env.example -> benchmark/.env and set RUNPOD_API_KEY."
        )
    return key


def _discover_ssh_pubkey() -> Optional[str]:
    explicit = (os.environ.get("RUNPOD_SSH_PUBKEY_PATH") or "").strip()
    candidates = [Path(explicit).expanduser()] if explicit else [
        Path.home() / ".ssh" / "id_ed25519.pub",
        Path.home() / ".ssh" / "id_rsa.pub",
    ]
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    return None


def build_pod_create_payload() -> Dict[str, Any]:
    """Construct RunPod REST API `POST /v1/pods` payload from env vars."""
    gpu_type = os.environ.get("RUNPOD_GPU_TYPE", "NVIDIA GeForce RTX 5090").strip()
    gpu_count = int(os.environ.get("RUNPOD_GPU_COUNT", "1"))
    cloud_type = os.environ.get("RUNPOD_CLOUD_TYPE", "SECURE").strip().upper()
    container_disk = int(os.environ.get("RUNPOD_CONTAINER_DISK_GB", "120"))
    volume_gb = int(os.environ.get("RUNPOD_VOLUME_GB", "220"))
    image_name = os.environ.get(
        "RUNPOD_IMAGE", "runpod/pytorch:2.8.0-py3.12-cuda12.8.1-devel-ubuntu2204"
    ).strip()
    pod_name = os.environ.get("RUNPOD_POD_NAME", "openfal-bench").strip()

    env_vars: Dict[str, str] = {}
    pubkey = _discover_ssh_pubkey()
    if pubkey:
        env_vars["PUBLIC_KEY"] = pubkey
    if os.environ.get("HF_TOKEN"):
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]

    payload: Dict[str, Any] = {
        "name": pod_name,
        "imageName": image_name,
        "cloudType": cloud_type,
        "computeType": "GPU",
        "gpuCount": gpu_count,
        "gpuTypeIds": [gpu_type],
        "gpuTypePriority": "availability",
        "containerDiskInGb": container_disk,
        "volumeInGb": volume_gb,
        "volumeMountPath": "/workspace",
        "ports": list(DEFAULT_PORTS),
        "supportPublicIp": True,
    }
    if env_vars:
        payload["env"] = env_vars
    return payload


def extract_ssh_endpoint(pod: Dict[str, Any]) -> Optional[Tuple[str, int]]:
    """Extract `(public_ip, ssh_port)` across RunPod REST API pod response schemas."""
    ip = pod.get("publicIp")
    port_map = pod.get("portMappings") or {}
    if ip and "22" in port_map:
        return str(ip), int(port_map["22"])
    if ip and 22 in port_map:
        return str(ip), int(port_map[22])

    runtime = pod.get("runtime") or {}
    for entry in runtime.get("ports") or []:
        if int(entry.get("privatePort", 0)) == 22 and entry.get("ip") and entry.get("publicPort"):
            return str(entry["ip"]), int(entry["publicPort"])
    return None


def _runpod_request(method: str, path: str, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {_require_api_key()}",
        "Content-Type": "application/json",
    }
    url = f"{RUNPOD_REST_BASE}{path}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.request(method, url, headers=headers, json=json_body)
        if resp.status_code >= 400:
            raise RuntimeError(f"RunPod API {method} {path} failed ({resp.status_code}): {resp.text}")
        if not resp.content:
            return {}
        return resp.json()


def _save_state(pod: Dict[str, Any], state_file: Path = DEFAULT_STATE_FILE) -> None:
    state_file.write_text(json.dumps(pod, indent=2), encoding="utf-8")


def _load_pod_id(explicit_id: Optional[str], state_file: Path = DEFAULT_STATE_FILE) -> str:
    if explicit_id:
        return explicit_id
    if state_file.is_file():
        data = json.loads(state_file.read_text(encoding="utf-8"))
        if data.get("id"):
            return str(data["id"])
    raise RuntimeError("No --pod-id given and benchmark/.runpod_pod.json not found. Run `create` first.")


def cmd_create(args: argparse.Namespace) -> None:
    payload = build_pod_create_payload()
    print(f"[runpod] Creating pod '{payload['name']}' with {payload['gpuCount']}x {payload['gpuTypeIds'][0]}...")
    pod = _runpod_request("POST", "/pods", json_body=payload)
    pod_id = pod["id"]
    _save_state(pod, args.state_file)
    print(f"[runpod] Pod created: id={pod_id}. Waiting for RUNNING state + SSH port...")

    deadline = time.monotonic() + float(args.timeout)
    while time.monotonic() < deadline:
        pod = _runpod_request("GET", f"/pods/{pod_id}")
        _save_state(pod, args.state_file)
        ssh = extract_ssh_endpoint(pod)
        status = pod.get("desiredStatus") or pod.get("status")
        if status == "RUNNING" and ssh is not None:
            host, port = ssh
            print(f"\nPod ready: id={pod_id} | status={status} | $/hr={pod.get('costPerHr')}")
            print(f"  SSH command : ssh root@{host} -p {port}")
            print(f"  openfal UI  : https://{pod_id}-8000.proxy.runpod.net")
            print(f"  ComfyUI     : https://{pod_id}-8188.proxy.runpod.net")
            print(f"  vLLM-Omni   : https://{pod_id}-8091.proxy.runpod.net")
            print("\nNext steps:")
            print("  python3 -m benchmark.runpod_provision bootstrap")
            print("  python3 -m benchmark.runpod_provision run")
            return
        time.sleep(4)

    raise TimeoutError(f"Pod {pod_id} did not expose an SSH endpoint within {args.timeout}s.")


def cmd_status(args: argparse.Namespace) -> None:
    pod_id = _load_pod_id(args.pod_id, args.state_file)
    pod = _runpod_request("GET", f"/pods/{pod_id}")
    _save_state(pod, args.state_file)
    ssh = extract_ssh_endpoint(pod)
    print(
        json.dumps(
            {
                "id": pod.get("id"),
                "name": pod.get("name"),
                "desiredStatus": pod.get("desiredStatus"),
                "costPerHr": pod.get("costPerHr"),
                "ssh": f"ssh root@{ssh[0]} -p {ssh[1]}" if ssh else None,
            },
            indent=2,
        )
    )


def _ssh_args(pod: Dict[str, Any]) -> Tuple[List[str], str, int]:
    ssh = extract_ssh_endpoint(pod)
    if ssh is None:
        raise RuntimeError(f"Pod {pod.get('id')} does not have an active SSH public IP/port yet.")
    host, port = ssh
    base = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-p",
        str(port),
        f"root@{host}",
    ]
    return base, host, port


def cmd_bootstrap(args: argparse.Namespace) -> None:
    pod_id = _load_pod_id(args.pod_id, args.state_file)
    pod = _runpod_request("GET", f"/pods/{pod_id}")
    ssh_base, host, port = _ssh_args(pod)

    ws = os.environ.get("BENCH_WORKSPACE", "/workspace/openfal_bench")
    repo_url = os.environ.get("OPENFAL_REPO_URL", "https://github.com/esokullu/openfal.git")
    branch = os.environ.get("OPENFAL_BRANCH", "main")
    flags = " ".join(
        f for f, enabled in (("--skip-comfyui", args.skip_comfyui), ("--skip-vllm", args.skip_vllm)) if enabled
    )

    remote_prep = (
        f"mkdir -p {shlex.quote(ws)} && "
        f"if [ ! -d {shlex.quote(ws + '/openfal/.git')} ]; then "
        f"git clone --branch {shlex.quote(branch)} {shlex.quote(repo_url)} {shlex.quote(ws + '/openfal')}; "
        f"else git -C {shlex.quote(ws + '/openfal')} pull --ff-only; fi"
    )
    subprocess.run([*ssh_base, remote_prep], check=True)

    env_local = Path(__file__).resolve().parent / ".env"
    if env_local.is_file():
        subprocess.run(
            [
                "scp",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-P",
                str(port),
                str(env_local),
                f"root@{host}:{ws}/openfal/benchmark/.env",
            ],
            check=True,
        )

    subprocess.run(
        [*ssh_base, f"bash {shlex.quote(ws + '/openfal/benchmark/setup_pod.sh')} {flags}"],
        check=True,
    )


def cmd_run(args: argparse.Namespace) -> None:
    pod_id = _load_pod_id(args.pod_id, args.state_file)
    pod = _runpod_request("GET", f"/pods/{pod_id}")
    ssh_base, host, port = _ssh_args(pod)
    ws = os.environ.get("BENCH_WORKSPACE", "/workspace/openfal_bench")
    extra = " ".join(shlex.quote(a) for a in args.bench_args)
    remote_cmd = (
        f"cd {shlex.quote(ws + '/openfal')} && "
        f"python3 -m benchmark.bench_runner --sequential-local {extra}"
    )
    subprocess.run([*ssh_base, remote_cmd], check=True)

    out_dir = Path(os.environ.get("BENCH_OUTPUT_DIR", "benchmark/results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "scp",
            "-r",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-P",
            str(port),
            f"root@{host}:{ws}/openfal/benchmark/results/.",
            str(out_dir),
        ],
        check=True,
    )
    print(f"[runpod] Results synced locally -> {out_dir.resolve()}")


def cmd_stop(args: argparse.Namespace) -> None:
    pod_id = _load_pod_id(args.pod_id, args.state_file)
    res = _runpod_request("POST", f"/pods/{pod_id}/stop")
    print(f"[runpod] Stopped pod {pod_id}: {res}")


def cmd_terminate(args: argparse.Namespace) -> None:
    pod_id = _load_pod_id(args.pod_id, args.state_file)
    _runpod_request("DELETE", f"/pods/{pod_id}")
    if args.state_file.is_file():
        args.state_file.unlink()
    print(f"[runpod] Terminated pod {pod_id} and cleaned {args.state_file.name}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="RunPod orchestrator for openfal benchmarks")
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Create a RunPod GPU pod from benchmark/.env")
    p_create.add_argument("--timeout", type=int, default=300)
    p_create.set_defaults(func=cmd_create)

    p_status = sub.add_parser("status", help="Inspect current pod state and SSH endpoint")
    p_status.add_argument("--pod-id", default=None)
    p_status.set_defaults(func=cmd_status)

    p_boot = sub.add_parser("bootstrap", help="Sync code + .env and run benchmark/setup_pod.sh remotely")
    p_boot.add_argument("--pod-id", default=None)
    p_boot.add_argument("--skip-comfyui", action="store_true")
    p_boot.add_argument("--skip-vllm", action="store_true")
    p_boot.set_defaults(func=cmd_bootstrap)

    p_run = sub.add_parser("run", help="Run benchmark sequentially on the pod and sync results back")
    p_run.add_argument("--pod-id", default=None)
    p_run.add_argument("bench_args", nargs=argparse.REMAINDER)
    p_run.set_defaults(func=cmd_run)

    p_stop = sub.add_parser("stop", help="Stop pod (keeps /workspace volume)")
    p_stop.add_argument("--pod-id", default=None)
    p_stop.set_defaults(func=cmd_stop)

    p_term = sub.add_parser("terminate", help="Delete pod completely to stop all billing")
    p_term.add_argument("--pod-id", default=None)
    p_term.set_defaults(func=cmd_terminate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
