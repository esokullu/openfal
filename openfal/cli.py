"""
CLI Entrypoint for openfal (`openfal serve`, `openfal info`, `openfal benchmark`, `openfal download`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import uvicorn

from openfal.config import HARDWARE_PROFILES, get_default_config
from openfal.engine.runner import OpenFalRunner


def cmd_serve(args: argparse.Namespace) -> None:
    cfg = get_default_config(profile_id=args.profile)
    cfg.host = args.host
    cfg.port = args.port
    if args.weights_dir:
        cfg.weights_dir = Path(args.weights_dir)

    from openfal.server.app import create_app

    app = create_app(cfg)
    print(f"==========================================================================")
    print(f"  openfal v0.1.0 — Strata-DiT Offload Inference Engine for MiniMax H3")
    print(f"  Hardware Profile : {cfg.profile.display_name}")
    print(f"  VRAM Counter     : {cfg.profile.resident_vram_blocks}/50 DiT blocks ({cfg.profile.quantization.value})")
    print(f"  Host RAM Pantry  : {50 - cfg.profile.resident_vram_blocks}/50 DiT blocks (Async PCIe DMA)")
    print(f"  Studio Web UI    : http://{cfg.host if cfg.host != '0.0.0.0' else 'localhost'}:{cfg.port}/")
    print(f"  fal.ai REST API  : http://{cfg.host if cfg.host != '0.0.0.0' else 'localhost'}:{cfg.port}/minimax/h3/text-to-video")
    print(f"==========================================================================")
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


def cmd_info(args: argparse.Namespace) -> None:
    cfg = get_default_config(profile_id=args.profile)
    runner = OpenFalRunner(cfg)
    print(json.dumps(runner.memory.get_telemetry(), indent=2))


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


def cmd_download(args: argparse.Namespace) -> None:
    target = Path(args.weights_dir)
    print(f"[openfal download] Target directory: {target.resolve()}")
    print("To download official open-weights MiniMaxAI/MiniMax-H3 & MiniMaxAI/MiniMax-H3-Image:")
    print(f"  hf download MiniMaxAI/MiniMax-H3 --local-dir {target}")
    print(f"  hf download MiniMaxAI/MiniMax-H3-Image --local-dir {target / 'H3-Image'}")


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

    p_bench = sub.add_parser("benchmark", help="Run a local MiniMax-H3 Strata-DiT pipeline benchmark")
    p_bench.add_argument("--profile", choices=list(HARDWARE_PROFILES.keys()), default="dual_5060ti_32gb")
    p_bench.add_argument("--endpoint", default="minimax/h3-max/text-to-video")
    p_bench.add_argument("--prompt", default="A futuristic high-speed train gliding over an ocean bridge at sunset")
    p_bench.add_argument("--resolution", default="768P")
    p_bench.add_argument("--duration", type=float, default=2.0)
    p_bench.add_argument("--steps", type=int, default=4)
    p_bench.set_defaults(func=cmd_benchmark)

    p_dl = sub.add_parser("download", help="Print Hugging Face download commands for MiniMax-H3 weights")
    p_dl.add_argument("--weights-dir", default="./weights/MiniMax-H3")
    p_dl.set_defaults(func=cmd_download)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
