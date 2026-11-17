"""Offline unit tests for the benchmark harness, ComfyUI API graphs, and RunPod utilities."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmark.bench_runner import (
    BenchmarkEntry,
    benchmark_cell,
    format_markdown_report,
    parse_nvidia_smi_csv,
)
from benchmark.comfy_workflows import build_comfy_prompt
from benchmark.env_utils import env_bool, parse_dotenv_file
from benchmark.runpod_provision import build_pod_create_payload, extract_ssh_endpoint
from benchmark.scenarios import SCENARIOS


def test_dotenv_parser_and_runpod_payload(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# comment\nRUNPOD_GPU_TYPE="NVIDIA GeForce RTX 5090"\nVLLM_OMNI_OFFLOAD=false\nEMPTY=\n',
        encoding="utf-8",
    )
    parsed = parse_dotenv_file(env_file)
    assert parsed["RUNPOD_GPU_TYPE"] == "NVIDIA GeForce RTX 5090"
    assert parsed["VLLM_OMNI_OFFLOAD"] == "false"
    assert parsed["EMPTY"] == ""

    monkeypatch.setenv("VLLM_OMNI_OFFLOAD", "false")
    assert env_bool("VLLM_OMNI_OFFLOAD", default=True) is False
    monkeypatch.setenv("RUNPOD_GPU_TYPE", "NVIDIA H100 80GB HBM3")
    monkeypatch.setenv("RUNPOD_GPU_COUNT", "2")
    monkeypatch.setenv("RUNPOD_CLOUD_TYPE", "SECURE")
    payload = build_pod_create_payload()
    assert payload["gpuTypeIds"] == ["NVIDIA H100 80GB HBM3"]
    assert payload["gpuCount"] == 2
    assert "22/tcp" in payload["ports"]

    assert extract_ssh_endpoint({"publicIp": "1.2.3.4", "portMappings": {"22": 40123}}) == ("1.2.3.4", 40123)
    assert extract_ssh_endpoint(
        {"runtime": {"ports": [{"privatePort": 22, "ip": "5.6.7.8", "publicPort": 31022}]}}
    ) == ("5.6.7.8", 31022)


def test_scenarios_match_openfal_dimensions_and_comfy_graphs():
    import openfal.engine.runner  # noqa: F401  # initialize engine before openfal.models to avoid package cycle
    from openfal.models.minimax_h3 import MiniMaxH3DiT

    # Every scenario's pixel dimensions must match openfal's resolution rounding cleanly
    for sc_id, sc in SCENARIOS.items():
        raw_w, raw_h = MiniMaxH3DiT.resolve_dimensions(sc.openfal_resolution, sc.openfal_aspect_ratio)
        if sc_id == "wan22_t2v_14b":
            aligned_w, aligned_h = (raw_w // 16) * 16, (raw_h // 16) * 16
        else:
            # Image-2.1 (`// 32 * 32`) & MiniMax-H3 (`round(x / 32) * 32`) both yield 1344x768 and 832x480
            aligned_w = (raw_w // 32) * 32
            aligned_h = (raw_h // 32) * 32
        assert (sc.width, sc.height) == (aligned_w, aligned_h), sc_id

        graph = build_comfy_prompt(sc, seed=42)
        assert isinstance(graph, dict) and len(graph) >= 7
        for node in graph.values():
            assert "class_type" in node and "inputs" in node
        output_types = {node["class_type"] for node in graph.values()}
        if sc.modality == "image":
            assert "SaveImage" in output_types
        else:
            assert "SaveVideo" in output_types


def test_nvidia_smi_parser_and_synthetic_guard():
    parsed = parse_nvidia_smi_csv("24120, 98\n 1024 , 12 \nbad,row\n")
    assert parsed == [(24120.0, 98.0), (1024.0, 12.0)]

    # Verify allow_synthetic=False fails loudly when openfal responds with synthetic mode
    fake_post = MagicMock()
    fake_post.raise_for_status.return_value = None
    fake_post.json.return_value = {
        "status_url": "http://127.0.0.1:8000/requests/id/status",
        "response_url": "http://127.0.0.1:8000/requests/id",
    }
    fake_get_status = MagicMock()
    fake_get_status.json.return_value = {"status": "COMPLETED"}
    fake_get_resp = MagicMock()
    fake_get_resp.json.return_value = {
        "images": [{"url": "http://127.0.0.1:8000/files/out.png"}],
        "strata_engine": {"mode": "synthetic"},
    }

    with patch("httpx.Client") as mock_client_cls:
        client_inst = mock_client_cls.return_value.__enter__.return_value
        client_inst.post.return_value = fake_post
        client_inst.get.side_effect = [fake_get_status, fake_get_resp]
        res = benchmark_cell(
            engine="openfal",
            base_url="http://127.0.0.1:8000",
            scenario=SCENARIOS["qwen21_t2i"],
            warmup=0,
            repeats=1,
            concurrency=1,
            base_seed=42,
            allow_synthetic=False,
            timeout_sec=30.0,
        )
    assert res.status == "error"
    assert "non-real mode 'synthetic'" in (res.error or "")

    md = format_markdown_report(
        "NVIDIA GeForce RTX 5090",
        [
            BenchmarkEntry(
                engine="openfal",
                scenario_id="qwen21_t2i",
                modality="image",
                concurrency=1,
                status="ok",
                cold_sec=20.0,
                warm_mean_sec=7.1,
                warm_p50_sec=7.1,
                throughput_req_per_min=8.45,
                peak_vram_mib=18432.0,
                mean_gpu_util_pct=94.0,
                engine_mode="real_pytorch",
            )
        ],
    )
    assert "| qwen21_t2i | openfal | 1 | 20.00 | 7.10 | 7.10 | 8.45 | 18.0 | 94% | real_pytorch |" in md
