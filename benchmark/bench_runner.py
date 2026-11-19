"""
Apples-to-apples benchmark harness: openfal vs ComfyUI vs vLLM-Omni.

Supports:
- External running servers (`--targets openfal,comfyui,vllm_omni`) OR
- `--sequential-local`: boots each engine sequentially in complete VRAM isolation on the GPU host,
  records server readiness time, cold-start (1st request), warm latency stats (mean/p50/min/max),
  peak GPU VRAM + average GPU utilization via `nvidia-smi`, and optional concurrent batching.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from benchmark.comfy_workflows import build_comfy_prompt
from benchmark.env_utils import env_bool, load_dotenv
from benchmark.scenarios import SCENARIOS, BenchmarkScenario


@dataclass
class RunSample:
    """Timing and delivery telemetry for a single inference invocation."""

    iteration: int
    is_cold: bool
    is_warmup: bool
    wall_sec: float
    reported_infer_sec: Optional[float]
    output_bytes: int
    engine_mode: Optional[str] = None


@dataclass
class BenchmarkEntry:
    """Aggregated benchmark measurements for one `(engine, scenario, concurrency)` cell."""

    engine: str
    scenario_id: str
    modality: str
    concurrency: int
    status: str  # "ok" | "error" | "skipped"
    error: Optional[str] = None
    startup_sec: Optional[float] = None
    cold_sec: Optional[float] = None
    warm_mean_sec: Optional[float] = None
    warm_p50_sec: Optional[float] = None
    warm_min_sec: Optional[float] = None
    warm_max_sec: Optional[float] = None
    batch_wall_sec: Optional[float] = None
    throughput_req_per_min: Optional[float] = None
    peak_vram_mib: Optional[float] = None
    mean_gpu_util_pct: Optional[float] = None
    engine_mode: Optional[str] = None
    samples: List[RunSample] = field(default_factory=list)


def parse_nvidia_smi_csv(output: str) -> List[Tuple[float, float]]:
    """Parse `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits` lines."""
    readings: List[Tuple[float, float]] = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mem = float(parts[0])
            util = float(parts[1])
            readings.append((mem, util))
        except ValueError:
            continue
    return readings


def _target_gpu_index() -> Optional[str]:
    idx = os.environ.get("BENCH_GPU_INDEX")
    if idx is not None and idx.strip():
        return idx.strip()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "0").strip()
    if cvd and "," not in cvd:
        return cvd
    return "0"


class GpuMonitor:
    """Background `nvidia-smi` poller capturing peak VRAM (MiB) and mean utilization (%)."""

    def __init__(self, interval_sec: float = 0.25):
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_vram_mib: Optional[float] = None
        self.util_samples: List[float] = []

    def __enter__(self) -> "GpuMonitor":
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _poll_loop(self) -> None:
        gpu_idx = _target_gpu_index()
        cmd = ["nvidia-smi"]
        if gpu_idx is not None:
            cmd.extend(["-i", gpu_idx])
        cmd.extend(["--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
        while not self._stop.is_set():
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                if proc.returncode == 0:
                    parsed = parse_nvidia_smi_csv(proc.stdout)
                    if parsed:
                        total_mem = sum(m for m, _ in parsed)
                        avg_util = sum(u for _, u in parsed) / len(parsed)
                        if self.peak_vram_mib is None or total_mem > self.peak_vram_mib:
                            self.peak_vram_mib = total_mem
                        self.util_samples.append(avg_util)
            except (FileNotFoundError, subprocess.SubprocessError):
                return
            self._stop.wait(self.interval_sec)

    def summary(self) -> Tuple[Optional[float], Optional[float]]:
        mean_util = round(sum(self.util_samples) / len(self.util_samples), 1) if self.util_samples else None
        peak = round(self.peak_vram_mib, 1) if self.peak_vram_mib is not None else None
        return peak, mean_util


def query_gpu_label() -> str:
    """Query primary GPU model name for result metadata (best-effort)."""
    try:
        gpu_idx = _target_gpu_index()
        cmd = ["nvidia-smi"]
        if gpu_idx is not None:
            cmd.extend(["-i", gpu_idx])
        cmd.extend(["--query-gpu=name", "--format=csv,noheader"])
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            return f"{len(lines)}x {lines[0]}" if len(lines) > 1 else lines[0]
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return "unknown_gpu"


def wait_for_vram_release(threshold_mib: float = 1500.0, timeout_sec: float = 20.0) -> None:
    """Wait until target GPU VRAM drops below `threshold_mib` after server shutdown."""
    gpu_idx = _target_gpu_index()
    cmd = ["nvidia-smi"]
    if gpu_idx is not None:
        cmd.extend(["-i", gpu_idx])
    cmd.extend(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
            if proc.returncode == 0 and proc.stdout.strip():
                used = float(proc.stdout.strip().splitlines()[0].strip())
                if used <= threshold_mib:
                    return
        except Exception:
            return
        time.sleep(1.0)


class ManagedEngineProcess:
    """Spawn and cleanly tear down a local inference server for VRAM-isolated benchmarking."""

    def __init__(
        self,
        name: str,
        cmd: List[str],
        health_url: str,
        cwd: Optional[Path] = None,
        timeout_sec: float = 300.0,
        env: Optional[Dict[str, str]] = None,
    ):
        self.name = name
        self.cmd = cmd
        self.health_url = health_url
        self.cwd = cwd
        self.timeout_sec = timeout_sec
        self.env = env
        self.proc: Optional[subprocess.Popen[str]] = None
        self.startup_sec: Optional[float] = None
        self._log_file = None

    def start(self, log_dir: Path) -> float:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"server_{self.name}.log"
        self._log_file = open(log_path, "w", encoding="utf-8")
        print(f"  [server:{self.name}] Starting: {' '.join(shlex.quote(c) for c in self.cmd)}")
        t0 = time.perf_counter()
        proc_env = os.environ.copy()
        proc_env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        proc_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        if self.env:
            proc_env.update(self.env)
        self.proc = subprocess.Popen(
            self.cmd,
            cwd=str(self.cwd) if self.cwd else None,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(os.name != "nt"),
            env=proc_env,
        )
        deadline = time.monotonic() + self.timeout_sec
        with httpx.Client(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(
                        f"{self.name} exited early (rc={self.proc.returncode}). See log: {log_path}"
                    )
                try:
                    resp = client.get(self.health_url)
                    if resp.status_code < 500:
                        self.startup_sec = round(time.perf_counter() - t0, 3)
                        print(f"  [server:{self.name}] Ready in {self.startup_sec:.2f}s ({self.health_url})")
                        return self.startup_sec
                except httpx.RequestError:
                    pass
                time.sleep(1.0)
        self.stop()
        raise TimeoutError(f"{self.name} did not become ready at {self.health_url} within {self.timeout_sec}s.")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            print(f"  [server:{self.name}] Stopping PID {self.proc.pid}...")
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                    capture_output=True,
                    timeout=15.0,
                )
                if self.cmd and "wsl" in Path(self.cmd[0]).name.lower():
                    wsl_distro = os.environ.get("VLLM_WSL_DISTRO", "Ubuntu-24.04")
                    subprocess.run(
                        ["wsl", "-d", wsl_distro, "--", "pkill", "-9", "-f", "vllm"],
                        capture_output=True,
                        timeout=10.0,
                    )
                try:
                    self.proc.wait(timeout=10.0)
                except Exception:
                    pass
            else:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                    self.proc.wait(timeout=15.0)
                except Exception:
                    try:
                        os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                        self.proc.wait(timeout=5.0)
                    except Exception:
                        pass
            wait_for_vram_release()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


_OPENFAL_UPLOAD_CACHE: Dict[Tuple[str, str], str] = {}


def _ensure_openfal_uploaded_image(client: httpx.Client, base_url: str, image_path_str: str) -> str:
    key = (base_url.rstrip("/"), image_path_str)
    if key in _OPENFAL_UPLOAD_CACHE:
        return _OPENFAL_UPLOAD_CACHE[key]
    img_path = Path(image_path_str)
    if not img_path.is_file():
        img_path = Path(__file__).resolve().parent.parent / image_path_str
    raw = img_path.read_bytes()
    resp = client.post(
        f"{base_url.rstrip('/')}/v1/storage/upload",
        content=raw,
        headers={"Content-Type": "image/png"},
    )
    resp.raise_for_status()
    url = resp.json()["url"]
    _OPENFAL_UPLOAD_CACHE[key] = url
    return url


def _run_openfal_once(
    base_url: str,
    scenario: BenchmarkScenario,
    seed: int,
    allow_synthetic: bool,
    timeout_sec: float,
) -> Tuple[Optional[float], int, Optional[str]]:
    payload: Dict[str, Any] = {
        "prompt": scenario.prompt,
        "resolution": scenario.openfal_resolution,
        "aspect_ratio": scenario.openfal_aspect_ratio,
        "num_inference_steps": scenario.steps,
        "guidance_scale": scenario.guidance_scale,
        "seed": seed,
    }
    if scenario.negative_prompt:
        payload["negative_prompt"] = scenario.negative_prompt
    if scenario.modality == "video":
        payload["duration"] = scenario.duration_sec
        payload["fps"] = scenario.fps

    with httpx.Client(timeout=timeout_sec) as client:
        if scenario.input_image_path:
            payload["start_image_url"] = _ensure_openfal_uploaded_image(
                client, base_url, scenario.input_image_path
            )
        sub = client.post(f"{base_url.rstrip('/')}/queue/{scenario.openfal_endpoint}", json=payload)
        sub.raise_for_status()
        qinfo = sub.json()
        status_url = qinfo["status_url"]
        response_url = qinfo["response_url"]

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            st = client.get(status_url).json()
            state = st.get("status")
            if state == "COMPLETED":
                break
            if state == "FAILED":
                logs = st.get("logs") or []
                tail = logs[-1].get("message") if logs else (st.get("error") or "FAILED")
                raise RuntimeError(f"openfal job failed: {tail}")
            time.sleep(0.25)
        else:
            raise TimeoutError(f"openfal job timed out after {timeout_sec}s")

        res = client.get(response_url).json()
        strata = res.get("strata_engine") or {}
        mode = strata.get("mode", "synthetic")
        if not allow_synthetic and mode != "real_pytorch":
            raise RuntimeError(
                f"openfal ran in non-real mode '{mode}' for {scenario.scenario_id}. "
                "Ensure real weights are installed (or pass --allow-synthetic)."
            )

        media_url = (
            res["video"]["url"]
            if scenario.modality == "video" and "video" in res
            else res["images"][0]["url"]
        )
        raw_bytes = client.get(media_url).content
        _maybe_save_sample(scenario.scenario_id, seed, "mp4" if scenario.modality == "video" else "png", raw_bytes)
        reported = (res.get("timings") or {}).get("inference")
        return reported, len(raw_bytes), mode


def _run_comfyui_once(
    base_url: str,
    scenario: BenchmarkScenario,
    seed: int,
    allow_synthetic: bool,  # unused symmetry parameter
    timeout_sec: float,
) -> Tuple[Optional[float], int, Optional[str]]:
    del allow_synthetic
    graph = build_comfy_prompt(scenario, seed=seed)
    base = base_url.rstrip("/")
    with httpx.Client(timeout=timeout_sec) as client:
        resp = client.post(f"{base}/prompt", json={"prompt": graph})
        if resp.status_code >= 400:
            raise RuntimeError(f"ComfyUI /prompt rejected graph ({resp.status_code}): {resp.text}")
        prompt_id = resp.json()["prompt_id"]

        deadline = time.monotonic() + timeout_sec
        item: Optional[Dict[str, Any]] = None
        while time.monotonic() < deadline:
            hist = client.get(f"{base}/history/{prompt_id}").json()
            if prompt_id in hist:
                item = hist[prompt_id]
                status = (item.get("status") or {}).get("status_str")
                if status == "error":
                    raise RuntimeError(f"ComfyUI workflow error: {item.get('status')}")
                if item.get("outputs"):
                    break
            time.sleep(0.3)
        if item is None:
            raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {timeout_sec}s")

        total_bytes = 0
        for node_out in (item.get("outputs") or {}).values():
            for key in ("images", "gifs", "videos", "audio"):
                for file_info in node_out.get(key) or []:
                    if not isinstance(file_info, dict):
                        continue
                    fn = file_info.get("filename")
                    if fn:
                        subf = file_info.get("subfolder", "")
                        ftype = file_info.get("type", "output")
                        blob = client.get(
                            f"{base}/view",
                            params={"filename": fn, "subfolder": subf, "type": ftype},
                        ).content
                        total_bytes += len(blob)
        return None, total_bytes, "comfyui"


def _maybe_save_sample(scenario_id: str, seed: int, ext: str, data: bytes) -> None:
    sample_dir = os.environ.get("BENCH_SAMPLE_DIR")
    if not sample_dir or not data:
        return
    out_dir = Path(sample_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{scenario_id}_seed{seed}.{ext}").write_bytes(data)


def _run_vllm_omni_once(
    base_url: str,
    scenario: BenchmarkScenario,
    seed: int,
    allow_synthetic: bool,
    timeout_sec: float,
) -> Tuple[Optional[float], int, Optional[str]]:
    del allow_synthetic
    base = base_url.rstrip("/")
    with httpx.Client(timeout=timeout_sec) as client:
        if scenario.modality == "image":
            img_payload: Dict[str, Any] = {
                "model": scenario.vllm_model_id,
                "prompt": scenario.prompt,
                "size": f"{scenario.width}x{scenario.height}",
                "seed": seed,
                "n": 1,
                "response_format": "b64_json",
                "output_format": "png",
            }
            if scenario.scenario_id == "qwen21_t2i":
                img_payload["true_cfg_scale"] = 1.0
                img_payload["sigmas"] = [
                    1.0,
                    0.978453,
                    0.95418,
                    0.926626,
                    0.89508,
                    0.845148,
                    0.704534,
                    0.414568,
                ]
            elif scenario.scenario_id == "minimax_h3_t2i":
                img_payload["num_inference_steps"] = scenario.steps
                img_payload["extra_params"] = {"min_generation_area": 1024}
            else:
                img_payload["num_inference_steps"] = scenario.steps
                img_payload["guidance_scale"] = scenario.guidance_scale

            resp = client.post(
                f"{base}/v1/images/generations",
                json=img_payload,
            )
            if resp.status_code >= 400:
                raise RuntimeError(f"vLLM-Omni /v1/images/generations HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            b64_str = (data.get("data") or [{}])[0].get("b64_json", "")
            raw_bytes = base64.b64decode(b64_str) if b64_str else b""
            _maybe_save_sample(scenario.scenario_id, seed, "png", raw_bytes)
            return None, (len(raw_bytes) or int(len(b64_str) * 0.75)), "vllm_omni"

        form_fields: Dict[str, str] = {
            "model": scenario.vllm_model_id,
            "prompt": scenario.prompt,
            "width": str(scenario.width),
            "height": str(scenario.height),
            "num_frames": str(scenario.num_frames),
            "fps": str(scenario.fps),
            "num_inference_steps": str(scenario.steps),
            "seed": str(seed),
        }
        if scenario.scenario_id == "wan22_i2v_14b":
            form_fields["guidance_scale"] = str(scenario.guidance_scale)
            form_fields["guidance_scale_2"] = str(scenario.guidance_scale)
            form_fields["boundary_ratio"] = "0.9"
            if scenario.negative_prompt:
                form_fields["negative_prompt"] = scenario.negative_prompt
        elif scenario.scenario_id == "minimax_h3_i2v":
            # vLLM-Omni enforces video duration in [4, 15]s -> minimum valid at 24fps is 97f (4.04s).
            form_fields["num_frames"] = str(max(97, scenario.num_frames))
            form_fields["flow_shift"] = "12"
            form_fields["extra_params"] = json.dumps({"task": "fl2va", "audio_flow_shift": 3.0})
        else:
            form_fields["guidance_scale"] = str(scenario.guidance_scale)
            if scenario.negative_prompt:
                form_fields["negative_prompt"] = scenario.negative_prompt

        files = None
        if scenario.input_image_path:
            img_path = Path(scenario.input_image_path)
            if not img_path.is_file():
                img_path = Path(__file__).resolve().parent.parent / scenario.input_image_path
            files = {"input_reference": (img_path.name, img_path.read_bytes(), "image/png")}

        sub = client.post(f"{base}/v1/videos", data=form_fields, files=files)
        if sub.status_code >= 400:
            raise RuntimeError(f"vLLM-Omni /v1/videos HTTP {sub.status_code}: {sub.text}")
        job = sub.json()
        vid_id = job["id"]

        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            st = client.get(f"{base}/v1/videos/{vid_id}").json()
            state = (st.get("status") or "").lower()
            if state == "completed":
                break
            if state in ("failed", "error"):
                raise RuntimeError(f"vLLM-Omni video job failed: {st.get('error') or st}")
            time.sleep(0.5)
        else:
            raise TimeoutError(f"vLLM-Omni video job {vid_id} timed out after {timeout_sec}s")

        raw_bytes = client.get(f"{base}/v1/videos/{vid_id}/content").content
        _maybe_save_sample(scenario.scenario_id, seed, "mp4", raw_bytes)
        reported = st.get("inference_time_s")
        return (float(reported) if reported is not None else None), len(raw_bytes), "vllm_omni"


ENGINE_CALLERS: Dict[
    str,
    Callable[[str, BenchmarkScenario, int, bool, float], Tuple[Optional[float], int, Optional[str]]],
] = {
    "openfal": _run_openfal_once,
    "comfyui": _run_comfyui_once,
    "vllm_omni": _run_vllm_omni_once,
}


def benchmark_cell(
    engine: str,
    base_url: str,
    scenario: BenchmarkScenario,
    warmup: int,
    repeats: int,
    concurrency: int,
    base_seed: int,
    allow_synthetic: bool,
    timeout_sec: float,
    startup_sec: Optional[float] = None,
) -> BenchmarkEntry:
    """Execute cold + warm + concurrency measurements under active `nvidia-smi` tracking."""
    caller = ENGINE_CALLERS[engine]
    entry = BenchmarkEntry(
        engine=engine,
        scenario_id=scenario.scenario_id,
        modality=scenario.modality,
        concurrency=concurrency,
        status="ok",
        startup_sec=startup_sec,
    )

    try:
        with GpuMonitor() as gpu_mon:
            # 1. Sequential warmup (iteration 0 captures cold lazy load / torch compilation)
            for w_idx in range(max(0, warmup)):
                t0 = time.perf_counter()
                rep, size_b, mode = caller(base_url, scenario, base_seed + w_idx, allow_synthetic, timeout_sec)
                wall = round(time.perf_counter() - t0, 3)
                entry.samples.append(
                    RunSample(
                        iteration=w_idx,
                        is_cold=(w_idx == 0),
                        is_warmup=True,
                        wall_sec=wall,
                        reported_infer_sec=rep,
                        output_bytes=size_b,
                        engine_mode=mode,
                    )
                )
                if w_idx == 0:
                    entry.cold_sec = wall
                if mode:
                    entry.engine_mode = mode

            # 2. Measured steady-state repeats (optionally at concurrency > 1)
            warm_walls: List[float] = []
            batch_t0 = time.perf_counter()
            if concurrency <= 1:
                for r_idx in range(repeats):
                    t0 = time.perf_counter()
                    rep, size_b, mode = caller(
                        base_url, scenario, base_seed + 100 + r_idx, allow_synthetic, timeout_sec
                    )
                    wall = round(time.perf_counter() - t0, 3)
                    warm_walls.append(wall)
                    entry.samples.append(
                        RunSample(
                            iteration=warmup + r_idx,
                            is_cold=(warmup == 0 and r_idx == 0),
                            is_warmup=False,
                            wall_sec=wall,
                            reported_infer_sec=rep,
                            output_bytes=size_b,
                            engine_mode=mode,
                        )
                    )
                    if mode:
                        entry.engine_mode = mode
            else:
                def _task(idx: int) -> Tuple[float, Optional[float], int, Optional[str]]:
                    t_start = time.perf_counter()
                    rep, size_b, mode = caller(
                        base_url, scenario, base_seed + 100 + idx, allow_synthetic, timeout_sec
                    )
                    return round(time.perf_counter() - t_start, 3), rep, size_b, mode

                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    results = list(pool.map(_task, range(repeats)))
                for r_idx, (wall, rep, size_b, mode) in enumerate(results):
                    warm_walls.append(wall)
                    entry.samples.append(
                        RunSample(
                            iteration=warmup + r_idx,
                            is_cold=False,
                            is_warmup=False,
                            wall_sec=wall,
                            reported_infer_sec=rep,
                            output_bytes=size_b,
                            engine_mode=mode,
                        )
                    )
                    if mode:
                        entry.engine_mode = mode

            batch_wall = round(time.perf_counter() - batch_t0, 3)
            entry.batch_wall_sec = batch_wall
            if entry.cold_sec is None and warm_walls:
                entry.cold_sec = warm_walls[0]
            if warm_walls:
                entry.warm_mean_sec = round(statistics.mean(warm_walls), 3)
                entry.warm_p50_sec = round(statistics.median(warm_walls), 3)
                entry.warm_min_sec = min(warm_walls)
                entry.warm_max_sec = max(warm_walls)
                entry.throughput_req_per_min = round((len(warm_walls) / max(0.001, batch_wall)) * 60.0, 2)

        peak_vram, mean_util = gpu_mon.summary()
        entry.peak_vram_mib = peak_vram
        entry.mean_gpu_util_pct = mean_util
    except Exception as exc:
        entry.status = "error"
        entry.error = str(exc)

    return entry


def format_markdown_report(gpu_name: str, entries: List[BenchmarkEntry]) -> str:
    """Format a concise comparison table suitable for terminal viewing or docs."""
    lines = [
        f"# openfal vs ComfyUI vs vLLM-Omni Benchmark ({gpu_name})",
        "",
        "| Scenario | Engine | Concurrency | Cold (s) | Warm Mean (s) | Warm p50 (s) | Req/min | Peak VRAM (GB) | GPU Util (%) | Mode / Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for e in entries:
        vram_gb = f"{e.peak_vram_mib / 1024.0:.1f}" if e.peak_vram_mib is not None else "n/a"
        util = f"{e.mean_gpu_util_pct:.0f}%" if e.mean_gpu_util_pct is not None else "n/a"
        cold = f"{e.cold_sec:.2f}" if e.cold_sec is not None else "-"
        wmean = f"{e.warm_mean_sec:.2f}" if e.warm_mean_sec is not None else "-"
        wp50 = f"{e.warm_p50_sec:.2f}" if e.warm_p50_sec is not None else "-"
        rpm = f"{e.throughput_req_per_min:.2f}" if e.throughput_req_per_min is not None else "-"
        status_cell = e.engine_mode or "ok" if e.status == "ok" else f"ERR: {(e.error or '')[:45]}"
        lines.append(
            f"| {e.scenario_id} | {e.engine} | {e.concurrency} | {cold} | {wmean} | {wp50} | {rpm} | {vram_gb} | {util} | {status_cell} |"
        )
    return "\n".join(lines) + "\n"


def _port_from_url(url: str, fallback: int) -> int:
    parsed = urlparse(url)
    return parsed.port or fallback


def _resolve_comfy_command(port: int) -> Tuple[List[str], Optional[Path]]:
    comfy_dir_env = os.environ.get("COMFYUI_DIR")
    comfy_python = os.environ.get("COMFYUI_PYTHON")
    comfy_base_dir = os.environ.get("COMFYUI_BASE_DIR")
    extra_paths_yaml = os.environ.get(
        "COMFYUI_EXTRA_MODEL_PATHS",
        str(Path(__file__).resolve().parent / "comfy_extra_model_paths.yaml"),
    )
    extra = shlex.split(os.environ.get("COMFYUI_EXTRA_ARGS", ""))

    # Auto-detect ComfyUI Desktop on Windows when COMFYUI_DIR is not explicitly set
    win_desktop_code = Path(
        os.environ.get("COMFYUI_DESKTOP_DIR")
        or (Path.home() / "AppData/Local/Comfy-Desktop/ComfyUI-Installs/ComfyUI/ComfyUI")
    )
    win_desktop_base = Path(comfy_base_dir or (Path.home() / "Documents/ComfyUI2"))
    win_desktop_py = win_desktop_base / ".venv" / "Scripts" / "python.exe"

    if not comfy_dir_env and win_desktop_code.is_dir() and win_desktop_py.is_file():
        py_bin = comfy_python or str(win_desktop_py)
        base_dir = comfy_base_dir or str(win_desktop_base)
        cmd = [
            py_bin,
            "main.py",
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--base-directory",
            base_dir,
        ]
        if Path(extra_paths_yaml).is_file():
            cmd.extend(["--extra-model-paths-config", extra_paths_yaml])
        cmd.extend(extra)
        return cmd, win_desktop_code

    comfy_dir = Path(comfy_dir_env or "/workspace/openfal_bench/ComfyUI")
    py_bin = comfy_python or sys.executable
    cmd = [py_bin, "main.py", "--listen", "127.0.0.1", "--port", str(port)]
    if comfy_base_dir:
        cmd.extend(["--base-directory", comfy_base_dir])
    if Path(extra_paths_yaml).is_file():
        cmd.extend(["--extra-model-paths-config", extra_paths_yaml])
    cmd.extend(extra)
    return cmd, comfy_dir


_VLLM_LOCAL_SUBDIRS: Dict[str, str] = {
    "qwen21_t2i": "Qwen-Image-2.1-Turbo",
    "wan22_i2v_14b": "Wan2.2-I2V-A14B",
    "minimax_h3_i2v": "MiniMax-H3/FL2VA",
    "minimax_h3_t2i": "MiniMax-H3-Image",
}


def _resolve_vllm_omni_command(sc: BenchmarkScenario, port: int, vllm_offload: bool) -> List[str]:
    use_wsl = os.name == "nt" and not os.environ.get("VLLM_OMNI_NATIVE_WINDOWS")
    wsl_models_dir = os.environ.get("VLLM_WSL_MODELS_DIR")
    if use_wsl and wsl_models_dir and sc.scenario_id in _VLLM_LOCAL_SUBDIRS:
        default_model = f"{wsl_models_dir.rstrip('/')}/{_VLLM_LOCAL_SUBDIRS[sc.scenario_id]}"
    else:
        default_model = sc.vllm_model_id
    model_override = os.environ.get(f"VLLM_MODEL_{sc.scenario_id.upper()}", default_model)
    extra_env = os.environ.get(f"VLLM_OMNI_ARGS_{sc.scenario_id.upper()}", os.environ.get("VLLM_OMNI_EXTRA_ARGS", ""))
    extra = shlex.split(extra_env)
    base_args = [
        "serve",
        model_override,
        "--omni",
        "--served-model-name",
        sc.vllm_model_id,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if vllm_offload and not extra:
        if sc.scenario_id == "qwen21_t2i":
            base_args.extend(["--quantization", "fp8"])
        elif sc.scenario_id == "wan22_i2v_14b":
            base_args.extend(["--quantization", "fp8", "--enable-cpu-offload"])
        elif sc.scenario_id in ("minimax_h3_i2v", "minimax_h3_t2i"):
            # DistributedLayerwiseOffloadBackend (checkpoint_mmap + bounded host staging)
            # avoids LayerWiseOffloadBackend pinning ~92 GB in host RAM.
            if sc.scenario_id == "minimax_h3_i2v":
                base_args.extend(["--task-type", "fl2va"])
            base_args.extend(
                [
                    "--trust-remote-code",
                    "--num-gpus",
                    "1",
                    "--tensor-parallel-size",
                    "1",
                    "--text-encoder-tp-size",
                    "1",
                    "--usp",
                    "1",
                    "--ring",
                    "1",
                ]
            )
            if sc.scenario_id == "minimax_h3_i2v":
                base_args.extend(["--vae-patch-parallel-size", "1", "--vae-parallel-mode", "tile", "--vae-use-tiling"])
            base_args.extend(
                [
                    "--enable-distributed-layerwise-offload",
                    "--dlo-no-use-allgather",
                    "--dlo-resident-layers",
                    "12",
                    "--enforce-eager",
                    "--diffusion-attention-backend",
                    "CUDNN_ATTN",
                ]
            )
        elif sc.vllm_default_offload:
            base_args.append("--enable-layerwise-offload")
    base_args.extend(extra)

    if use_wsl:
        wsl_distro = os.environ.get("VLLM_WSL_DISTRO", "Ubuntu-24.04")
        wsl_venv_vllm = os.environ.get("VLLM_WSL_BIN", "$HOME/vllm/vllm-omni-env/bin/vllm")
        quoted = " ".join(shlex.quote(a) for a in base_args)
        bash_cmd = (
            "export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 "
            "CUDA_HOME=/usr/local/cuda-13.0 "
            "LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/cuda-12.8/lib64:/usr/lib/wsl/lib "
            "VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=14400 && "
            f"{wsl_venv_vllm} {quoted}"
        )
        return ["wsl", "-d", wsl_distro, "--", "bash", "-lc", bash_cmd]
    return ["vllm", *base_args]


def _run_sequential_local(
    targets: List[str],
    scenarios: List[BenchmarkScenario],
    concurrencies: List[int],
    args: argparse.Namespace,
    urls: Dict[str, str],
) -> List[BenchmarkEntry]:
    entries: List[BenchmarkEntry] = []
    weights_dir = Path(os.environ.get("OPENFAL_WEIGHTS_DIR", "./weights/MiniMax-H3"))
    profile = os.environ.get("OPENFAL_PROFILE", "auto")
    vllm_offload = env_bool("VLLM_OMNI_OFFLOAD", True)
    log_dir = Path(args.output_dir) / "logs"

    for engine in targets:
        if engine == "openfal":
            port = _port_from_url(urls["openfal"], 8008)
            cmd = [
                sys.executable,
                "-m",
                "openfal.cli",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--profile",
                profile,
                "--weights-dir",
                str(weights_dir),
            ]
            for sc in scenarios:
                srv = ManagedEngineProcess(f"openfal_{sc.scenario_id}", cmd, f"{urls['openfal'].rstrip('/')}/health")
                try:
                    startup = srv.start(log_dir)
                    for c in concurrencies:
                        print(f"  [openfal] Running scenario={sc.scenario_id} concurrency={c}...")
                        entries.append(
                            benchmark_cell(
                                engine, urls["openfal"], sc, args.warmup, args.repeats, c,
                                args.seed, args.allow_synthetic, args.request_timeout, startup_sec=startup,
                            )
                        )
                finally:
                    srv.stop()

        elif engine == "comfyui":
            port = _port_from_url(urls["comfyui"], 8188)
            cmd, comfy_cwd = _resolve_comfy_command(port)
            srv = ManagedEngineProcess("comfyui", cmd, f"{urls['comfyui'].rstrip('/')}/system_stats", cwd=comfy_cwd)
            try:
                startup = srv.start(log_dir)
                for sc in scenarios:
                    try:
                        _http_json(
                            "POST",
                            f"{urls['comfyui'].rstrip('/')}/free",
                            payload={"unload_models": True, "free_memory": True},
                            timeout=15.0,
                        )
                    except Exception:
                        pass
                    for c in concurrencies:
                        print(f"  [comfyui] Running scenario={sc.scenario_id} concurrency={c}...")
                        entries.append(
                            benchmark_cell(
                                engine, urls["comfyui"], sc, args.warmup, args.repeats, c,
                                args.seed, args.allow_synthetic, args.request_timeout, startup_sec=startup,
                            )
                        )
            finally:
                srv.stop()

        elif engine == "vllm_omni":
            port = _port_from_url(urls["vllm_omni"], 8091)
            for sc in scenarios:
                cmd = _resolve_vllm_omni_command(sc, port, vllm_offload)
                srv = ManagedEngineProcess(
                    f"vllm_omni_{sc.scenario_id}", cmd, f"{urls['vllm_omni'].rstrip('/')}/health", timeout_sec=600.0
                )
                try:
                    startup = srv.start(log_dir)
                    for c in concurrencies:
                        print(f"  [vllm_omni] Running scenario={sc.scenario_id} concurrency={c}...")
                        entries.append(
                            benchmark_cell(
                                engine, urls["vllm_omni"], sc, args.warmup, args.repeats, c,
                                args.seed, args.allow_synthetic, args.request_timeout, startup_sec=startup,
                            )
                        )
                except Exception as exc:
                    for c in concurrencies:
                        entries.append(
                            BenchmarkEntry(
                                engine=engine,
                                scenario_id=sc.scenario_id,
                                modality=sc.modality,
                                concurrency=c,
                                status="error",
                                error=str(exc),
                            )
                        )
                finally:
                    srv.stop()

    return entries


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Benchmark openfal vs ComfyUI vs vLLM-Omni")
    parser.add_argument("--targets", default=os.environ.get("BENCH_TARGETS", "openfal,comfyui,vllm_omni"))
    parser.add_argument(
        "--scenarios",
        default=os.environ.get("BENCH_SCENARIOS", "qwen21_t2i,minimax_h3_t2i,wan22_i2v_14b,minimax_h3_i2v"),
    )
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("BENCH_WARMUP", "1")))
    parser.add_argument("--repeats", type=int, default=int(os.environ.get("BENCH_REPEATS", "3")))
    parser.add_argument("--concurrency", default=os.environ.get("BENCH_CONCURRENCY", "1"))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("BENCH_SEED", "42")))
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--output-dir", default=os.environ.get("BENCH_OUTPUT_DIR", "benchmark/results"))
    parser.add_argument(
        "--sequential-local",
        action="store_true",
        help="Start and stop each local server sequentially on the GPU host so engines never share VRAM.",
    )
    parser.add_argument("--allow-synthetic", action="store_true", help="Allow synthetic fallback on openfal.")
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip() in ENGINE_CALLERS]
    scenarios = [SCENARIOS[s.strip()] for s in args.scenarios.split(",") if s.strip() in SCENARIOS]
    concurrencies = [int(c.strip()) for c in args.concurrency.split(",") if c.strip()]
    urls = {
        "openfal": os.environ.get("OPENFAL_URL", "http://127.0.0.1:8008"),
        "comfyui": os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188"),
        "vllm_omni": os.environ.get("VLLM_OMNI_URL", "http://127.0.0.1:8091"),
    }

    gpu_name = query_gpu_label()
    print(f"=== Benchmark Suite ({gpu_name}) ===")
    print(f"Targets={targets} | Scenarios={[s.scenario_id for s in scenarios]} | Concurrency={concurrencies}")

    if args.sequential_local:
        entries = _run_sequential_local(targets, scenarios, concurrencies, args, urls)
    else:
        entries = []
        for engine in targets:
            for sc in scenarios:
                for c in concurrencies:
                    print(f"Running engine={engine} scenario={sc.scenario_id} concurrency={c}...")
                    entries.append(
                        benchmark_cell(
                            engine, urls[engine], sc, args.warmup, args.repeats, c,
                            args.seed, args.allow_synthetic, args.request_timeout,
                        )
                    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_gpu = "".join(ch if ch.isalnum() else "_" for ch in gpu_name).strip("_")
    json_path = out_dir / f"bench_{safe_gpu}_{ts}.json"
    md_path = out_dir / f"bench_{safe_gpu}_{ts}.md"

    md_text = format_markdown_report(gpu_name, entries)
    json_path.write_text(
        json.dumps({"gpu": gpu_name, "entries": [asdict(e) for e in entries]}, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(md_text, encoding="utf-8")

    print("\n" + md_text)
    print(f"Saved JSON -> {json_path}")
    print(f"Saved MD   -> {md_path}")


if __name__ == "__main__":
    main()
