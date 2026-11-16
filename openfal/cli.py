"""
CLI Entrypoint for openfal (`openfal serve`, `openfal download`, `openfal detect`, `openfal models`, `openfal info`, `openfal benchmark`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import uvicorn

from openfal.config import HARDWARE_PROFILES, get_default_config
from openfal.engine.downloader import COMPONENT_BUNDLES, MODEL_COMPONENTS, StrataModelManager
from openfal.engine.runner import OpenFalRunner


def cmd_serve(args: argparse.Namespace) -> None:
    weights_path = Path(args.weights_dir) if args.weights_dir else Path("./weights/MiniMax-H3")
    selected_profile = args.profile
    hw_info = None

    if selected_profile == "auto":
        mgr = StrataModelManager(weights_path)
        hw_info = mgr.detect_hardware()
        selected_profile = hw_info["recommended_profile_id"]
        if hw_info.get("recommended_cuda_visible_devices") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
            os.environ["CUDA_VISIBLE_DEVICES"] = hw_info["recommended_cuda_visible_devices"]

    cfg = get_default_config(profile_id=selected_profile)
    cfg.host = args.host
    cfg.port = args.port
    if args.weights_dir:
        cfg.weights_dir = weights_path

    from openfal.server.app import create_app

    app = create_app(cfg)
    print("==========================================================================")
    print("  openfal v0.1.0 — Strata-DiT Offload Engine (Wan 2.2 MoE / Qwen-Image-2.1 / MiniMax H3)")
    print(f"  Hardware Profile : {cfg.profile.display_name} ({cfg.profile_id})")
    if hw_info and hw_info.get("selected_compute_gpus"):
        gpu_desc = ", ".join(f"GPU {g['index']}: {g['name']} ({g['vram_gb']} GB)" for g in hw_info["selected_compute_gpus"])
        print(f"  Compute GPU(s)   : {gpu_desc}")
    if hw_info and hw_info.get("ignored_gpus"):
        ign_desc = ", ".join(f"GPU {g['index']}: {g['name']} [{g['reason']}]" for g in hw_info["ignored_gpus"])
        print(f"  Ignored GPU(s)   : {ign_desc}")
    print(f"  VRAM Counter     : {cfg.profile.resident_vram_blocks}/50 DiT blocks ({cfg.profile.quantization.value})")
    print(f"  Host RAM Pantry  : {50 - cfg.profile.resident_vram_blocks}/50 DiT blocks (Async PCIe DMA)")
    print(f"  Studio Web UI    : http://{cfg.host if cfg.host != '0.0.0.0' else 'localhost'}:{cfg.port}/")
    print(f"  fal.ai REST API  : http://{cfg.host if cfg.host != '0.0.0.0' else 'localhost'}:{cfg.port}/wan/v2.2/text-to-video")
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
        f"Mode={mode} | Source={args.source} | Quant={quant} | CleanupRaw={args.cleanup_raw}"
    )

    task = mgr.start_download_or_pack(
        component_id=args.component,
        quantization=quant,
        mode=mode,
        source=args.source,
        cleanup_raw=args.cleanup_raw,
        hf_token=args.hf_token,
        custom_repo_id=args.repo_id,
        run_async=False,
        on_log=lambda msg: print(f"  {msg}"),
    )
    if task.status == "FAILED":
        print(f"[openfal download] FAILED: {task.error}", file=sys.stderr)
        sys.exit(1)
    print(f"[openfal download] SUCCESS ({task.downloaded_mb} MB processed at {task.speed_mbps} MB/s)")


def cmd_cleanup(args: argparse.Namespace) -> None:
    mgr = StrataModelManager(Path(args.weights_dir))
    cache_path = Path(args.cache_dir) if args.cache else None
    output_path = Path(args.output_dir) if args.outputs else None

    preview = mgr.cleanup(
        component_id=args.component,
        incomplete_only=(args.component is None),
        cache_dir=cache_path,
        output_dir=output_path,
        dry_run=True,
    )

    if preview["removed_count"] == 0:
        print("[openfal cleanup] Nothing to clean up.")
        return

    print(f"[openfal cleanup] Found {preview['removed_count']} target(s) totaling {preview['freed_mb']} MB ({preview['freed_gb']} GB):")
    for item in preview["removed"]:
        print(f"  - {item['path']} ({item['freed_mb']} MB) [{item['reason']}]")

    if args.dry_run:
        print("[openfal cleanup] Dry run complete (no files deleted).")
        return

    if not args.yes:
        reply = input("Proceed with deletion? [y/N]: ").strip().lower()
        if reply not in ("y", "yes"):
            print("[openfal cleanup] Aborted.")
            return

    res = mgr.cleanup(
        component_id=args.component,
        incomplete_only=(args.component is None),
        cache_dir=cache_path,
        output_dir=output_path,
        dry_run=False,
    )
    print(json.dumps(res, indent=2))


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
        choices=["auto"] + list(HARDWARE_PROFILES.keys()),
        default="auto",
        help="Consumer GPU offload profile (default: auto-detect compute GPUs)",
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
        choices=list(COMPONENT_BUNDLES.keys()) + list(MODEL_COMPONENTS.keys()),
        default="minimax-only",
        help="Component or bundle to download (e.g. 'minimax-only', 'minimax-omni-compact', 'alibaba-only', 'all')",
    )
    p_dl.add_argument(
        "--quant",
        choices=["fp8_e4m3", "int4_awq", "nf4", "bf16"],
        default="fp8_e4m3",
        help="Target quantization format for 50-block Strata shards",
    )
    p_dl.add_argument(
        "--source",
        choices=["prequantized", "official-bf16"],
        default="prequantized",
        help="'prequantized' (default: Comfy-Org/unsloth fast links) or 'official-bf16' (upstream BF16 fallback + local quantization)",
    )
    p_dl.add_argument(
        "--cleanup-raw",
        action="store_true",
        help="After local quantization from official-bf16, delete raw BF16 .safetensors files to save disk space",
    )
    p_dl.add_argument("--auto-detect", action="store_true", help="Auto-select quantization based on detected GPUs")
    p_dl.add_argument(
        "--pack-only",
        action="store_true",
        help="Quantize/pack existing local .safetensors (or build structural shards) without contacting Hugging Face",
    )
    p_dl.add_argument("--hf-token", default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    p_dl.add_argument("--repo-id", default=None, help="Optional custom Hugging Face repo ID override")
    p_dl.set_defaults(func=cmd_download)

    p_clean = sub.add_parser(
        "cleanup",
        help="Clean up partial/aborted Hugging Face downloads, specific model components, or caches",
    )
    p_clean.add_argument("--weights-dir", default="./weights/MiniMax-H3", help="Weights directory to clean")
    p_clean.add_argument(
        "--component",
        choices=list(COMPONENT_BUNDLES.keys()) + list(MODEL_COMPONENTS.keys()),
        default=None,
        help="Remove a specific component or bundle completely (e.g. 'all', 'wan22_t2v_moe', 'minimax-only')",
    )
    p_clean.add_argument("--cache", action="store_true", help="Also clear openfal Layer-50 text encoder cache")
    p_clean.add_argument("--cache-dir", default="./.cache/openfal")
    p_clean.add_argument("--outputs", action="store_true", help="Also clear generated outputs directory")
    p_clean.add_argument("--output-dir", default="./outputs")
    p_clean.add_argument("--dry-run", action="store_true", help="Show what would be deleted without removing files")
    p_clean.add_argument("-y", "--yes", action="store_true", help="Skip interactive confirmation prompt")
    p_clean.set_defaults(func=cmd_cleanup)

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
