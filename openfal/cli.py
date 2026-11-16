"""
CLI Entrypoint for openfal (`openfal serve`, `openfal download`, `openfal detect`, `openfal models`, `openfal info`, `openfal benchmark`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import uvicorn

from openfal.config import HARDWARE_PROFILES, get_default_config
from openfal.engine.downloader import MODEL_COMPONENTS, StrataModelManager
from openfal.engine.runner import OpenFalRunner


def cmd_serve(args: argparse.Namespace) -> None:
    cfg = get_default_config(profile_id=args.profile)
    cfg.host = args.host
    cfg.port = args.port
    if args.weights_dir:
        cfg.weights_dir = Path(args.weights_dir)

    from openfal.server.app import create_app

    app = create_app(cfg)
    print("==========================================================================")
    print("  openfal v0.1.0 — Strata-DiT Offload Inference Engine for MiniMax H3")
    print(f"  Hardware Profile : {cfg.profile.display_name}")
    print(f"  VRAM Counter     : {cfg.profile.resident_vram_blocks}/50 DiT blocks ({cfg.profile.quantization.value})")
    print(f"  Host RAM Pantry  : {50 - cfg.profile.resident_vram_blocks}/50 DiT blocks (Async PCIe DMA)")
    print(f"  Studio Web UI    : http://{cfg.host if cfg.host != '0.0.0.0' else 'localhost'}:{cfg.port}/")
    print(f"  fal.ai REST API  : http://{cfg.host if cfg.host != '0.0.0.0' else 'localhost'}:{cfg.port}/minimax/h3/text-to-video")
    print("==========================================================================")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


def cmd_info(args: argparse.Namespace) -> None:
    cfg = get_default_config(profile_id=args.profile)
    runner = OpenFalRunner(cfg)
    print(json.dumps(runner.memory.get_telemetry(), indent=2))


def cmd_detect(args: argparse.Namespace) -> None:
    mgr = StrataModelManager(Path(args.weights_dir))
    hw = mgr.detect_hardware()
    print(json.dumps(hw, indent=2))


def cmd_models(args: argparse.Namespace) -> None:
    mgr = StrataModelManager(Path(args.weights_dir))
    print(json.dumps(mgr.get_all_status(), indent=2))


def cmd_download(args: argparse.Namespace) -> None:
    weights_dir = Path(args.weights_dir)
    mgr = StrataModelManager(weights_dir)

    if args.auto_detect:
        hw = mgr.detect_hardware()
        quant = hw["recommended_quantization"]
        print(
            f"[openfal download] Auto-detected hardware -> profile={hw['recommended_profile_id']}, "
            f"quant={quant}, free_nvme={hw['free_nvme_gb']} GB"
        )
    else:
        quant = args.quant

    mode = "strata_repack" if args.pack_only else "hf_hub"
    print(
        f"[openfal download] Target={weights_dir.resolve()} | Component={args.component} | "
        f"Mode={mode} | Quant={quant}"
    )

    task = mgr.start_download_or_pack(
        component_id=args.component,
        quantization=quant,
        mode=mode,
        hf_token=args.hf_token,
        custom_repo_id=args.repo_id,
        run_async=False,
        on_log=lambda msg: print(f"  {msg}"),
    )
    if task.status == "FAILED":
        print(f"[openfal download] FAILED: {task.error}", file=sys.stderr)
        sys.exit(1)
    print(f"[openfal download] SUCCESS ({task.downloaded_mb} MB processed at {task.speed_mbps} MB/s)")


def cmd_benchmark(args: argparse.Namespace) -> None:
    cfg = get_default_config(profile_id=args.profile)
    runner = OpenFalRunner(cfg)
    print(f"[openfal benchmark] Profile: {cfg.profile.display_name}")
    t0 = time.perf_counter()
    res = runner.execute(
        endpoint_id=args.endpoint,
        payload={
            "prompt": args.prompt,
            "resolution": args.resolution,
            "duration": args.duration,
            "num_inference_steps": args.steps,
        },
        on_log=lambda m: print(f"  {m}"),
    )
    dt = time.perf_counter() - t0
    print(f"[openfal benchmark] Completed in {dt:.2f}s -> {json.dumps(res, indent=2)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="openfal",
        description="Strata-inspired 3-Tier Offload Inference Engine & fal.ai Server for MiniMax H3",
    )
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="Start the fal.ai-compatible API server & Studio Web UI")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p_serve.add_argument(
        "--profile",
        choices=list(HARDWARE_PROFILES.keys()),
        default="dual_5060ti_32gb",
        help="Consumer GPU offload profile",
    )
    p_serve.add_argument("--weights-dir", default=None, help="Path to MiniMax-H3 weights directory")
    p_serve.set_defaults(func=cmd_serve)

    p_info = sub.add_parser("info", help="Show Strata 3-Tier memory hierarchy telemetry for a profile")
    p_info.add_argument(
        "--profile",
        choices=list(HARDWARE_PROFILES.keys()),
        default="dual_5060ti_32gb",
    )
    p_info.set_defaults(func=cmd_info)

    p_detect = sub.add_parser("detect", help="Auto-detect GPUs, System RAM, NVMe space and recommend a Strata profile")
    p_detect.add_argument("--weights-dir", default="./weights/MiniMax-H3")
    p_detect.set_defaults(func=cmd_detect)

    p_models = sub.add_parser("models", help="Inspect local installation & 50-block Strata shard status of MiniMax-H3")
    p_models.add_argument("--weights-dir", default="./weights/MiniMax-H3")
    p_models.set_defaults(func=cmd_models)

    p_dl = sub.add_parser(
        "download",
        help="Download MiniMax-H3 / MiniMax-H3-Image weights from Hugging Face & pack 50-block Strata shards",
    )
    p_dl.add_argument("--weights-dir", default="./weights/MiniMax-H3", help="Target directory for weights")
    p_dl.add_argument(
        "--component",
        choices=["all"] + list(MODEL_COMPONENTS.keys()),
        default="all",
        help="Which partition/component to download (default: all)",
    )
    p_dl.add_argument(
        "--quant",
        choices=["fp8_e4m3", "int4_awq", "nf4", "bf16"],
        default="fp8_e4m3",
        help="Target quantization format for 50-block Strata shards",
    )
    p_dl.add_argument("--auto-detect", action="store_true", help="Auto-select quantization based on detected GPUs")
    p_dl.add_argument(
        "--pack-only",
        action="store_true",
        help="Build 50-block Strata safetensors shards locally without contacting Hugging Face",
    )
    p_dl.add_argument("--hf-token", default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    p_dl.add_argument("--repo-id", default=None, help="Optional custom Hugging Face repo ID override")
    p_dl.set_defaults(func=cmd_download)

    p_bench = sub.add_parser("benchmark", help="Run a local Strata-DiT pipeline benchmark")
    p_bench.add_argument("--profile", choices=list(HARDWARE_PROFILES.keys()), default="dual_5060ti_32gb")
    p_bench.add_argument("--endpoint", default="wan/v2.2/text-to-video")
    p_bench.add_argument("--prompt", default="A futuristic high-speed train gliding over an ocean bridge at sunset")
    p_bench.add_argument("--resolution", default="768P")
    p_bench.add_argument("--duration", type=float, default=2.0)
    p_bench.add_argument("--steps", type=int, default=4)
    p_bench.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
