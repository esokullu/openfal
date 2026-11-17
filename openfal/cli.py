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
from openfal.engine.downloader import (
    COMPONENT_BUNDLES,
    MODEL_COMPONENTS,
    StrataModelManager,
    recommend_autosetup_plans,
)
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


def _run_download_with_progress(
    mgr: StrataModelManager,
    weights_dir: Path,
    component_id: str,
    quantization: str,
    mode: str,
    source: str = "prequantized",
    cleanup_raw: bool = False,
    hf_token: str | None = None,
    custom_repo_id: str | None = None,
    verbose: bool = False,
):
    log_path = weights_dir.resolve() / "openfal_download.log"
    print(f"  Detail log : {log_path.as_uri()}")
    if not verbose:
        print("  View mode  : Compact overall % + active task (pass --verbose for raw HF bars)")

    if verbose:
        return mgr.start_download_or_pack(
            component_id=component_id,
            quantization=quantization,
            mode=mode,
            source=source,
            cleanup_raw=cleanup_raw,
            hf_token=hf_token,
            custom_repo_id=custom_repo_id,
            run_async=False,
            on_log=lambda msg: print(f"  {msg}"),
            log_file=log_path,
        )

    state = {
        "component_idx": 0,
        "total_components": 1,
        "component_id": component_id,
        "completed_gb": 0.0,
        "total_gb": 1.0,
        "component_approx_gb": 0.0,
        "current_desc": "Preparing...",
        "last_render_len": 0,
        "last_render_t": 0.0,
    }

    def _render_bar(overall_pct: float, detail_text: str) -> None:
        filled = int(max(0.0, min(100.0, overall_pct)) / 5.0)
        bar = "#" * filled + "-" * (20 - filled)
        line = (
            f"\r  Overall [{bar}] {overall_pct:5.1f}% "
            f"({state['completed_gb']:.1f}/{state['total_gb']:.1f} GB) | "
            f"[{state['component_idx']}/{state['total_components']}] {state['component_id']}: {detail_text}"
        )
        pad = max(0, state["last_render_len"] - len(line))
        state["last_render_len"] = len(line)
        sys.stdout.write(line + (" " * pad))
        sys.stdout.flush()

    def _on_progress(ev: dict) -> None:
        phase = ev.get("phase")
        state["component_idx"] = ev.get("component_idx", state["component_idx"])
        state["total_components"] = ev.get("total_components", state["total_components"])
        state["component_id"] = ev.get("component_id", state["component_id"])
        state["completed_gb"] = ev.get("completed_gb", state["completed_gb"])
        state["total_gb"] = ev.get("total_gb", state["total_gb"])
        state["component_approx_gb"] = ev.get("component_approx_gb", state["component_approx_gb"])
        if phase == "component_start":
            _render_bar(ev.get("overall_pct", 0.0), f"connecting to {ev.get('repo_id', 'HF')}...")
        elif phase == "component_done":
            _render_bar(ev.get("overall_pct", 100.0), "done")
            sys.stdout.write("\n")
            sys.stdout.flush()
            state["last_render_len"] = 0

    tqdm_cls = None
    try:
        from tqdm.auto import tqdm as _BaseTqdm

        class _CompactHFTqdm(_BaseTqdm):
            def __init__(self, *args, **kwargs):
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)
                self._t_start = time.perf_counter()

            def update(self, n=1):
                super().update(n)
                now = time.perf_counter()
                if now - state["last_render_t"] < 0.25 and self.n < (self.total or 0):
                    return
                state["last_render_t"] = now
                desc = (self.desc or "downloading").strip()
                if self.total and self.total > 0:
                    frac = min(1.0, self.n / self.total)
                    if self.total > 1024 * 1024:
                        # Byte-level progress bar -> estimate overall GB smoothly
                        est_done_gb = state["completed_gb"] + frac * state["component_approx_gb"]
                        overall_pct = min(99.9, (est_done_gb / max(0.01, state["total_gb"])) * 100.0)
                        elapsed = max(0.01, now - self._t_start)
                        mbps = (self.n / (1024 * 1024)) / elapsed
                        done_gb = self.n / (1024**3)
                        tot_gb = self.total / (1024**3)
                        _render_bar(overall_pct, f"{desc} {frac * 100:.1f}% ({done_gb:.1f}/{tot_gb:.1f} GB, {mbps:.1f} MB/s)")
                    else:
                        overall_pct = (state["completed_gb"] / max(0.01, state["total_gb"])) * 100.0
                        _render_bar(overall_pct, f"{desc} {self.n}/{self.total}")

        tqdm_cls = _CompactHFTqdm
    except Exception:
        tqdm_cls = None

    return mgr.start_download_or_pack(
        component_id=component_id,
        quantization=quantization,
        mode=mode,
        source=source,
        cleanup_raw=cleanup_raw,
        hf_token=hf_token,
        custom_repo_id=custom_repo_id,
        run_async=False,
        on_log=None,
        progress_callback=_on_progress,
        tqdm_class=tqdm_cls,
        log_file=log_path,
    )


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

    task = _run_download_with_progress(
        mgr=mgr,
        weights_dir=weights_dir,
        component_id=args.component,
        quantization=quant,
        mode=mode,
        source=args.source,
        cleanup_raw=args.cleanup_raw,
        hf_token=args.hf_token,
        custom_repo_id=args.repo_id,
        verbose=getattr(args, "verbose", False),
    )
    if task.status == "FAILED":
        print(f"\n[openfal download] FAILED: {task.error}", file=sys.stderr)
        sys.exit(1)
    print(f"[openfal download] SUCCESS ({task.downloaded_mb} MB processed) — Full log: {(weights_dir.resolve() / 'openfal_download.log').as_uri()}")


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


def cmd_autosetup(args: argparse.Namespace) -> None:
    weights_dir = Path(args.weights_dir)
    mgr = StrataModelManager(weights_dir)
    hw = mgr.detect_hardware()
    plans = recommend_autosetup_plans(hw)

    print("==========================================================================")
    print("  openfal autosetup — Hardware & Disk-Aware Installer")
    print(f"  Platform         : {hw['platform']}")
    print(f"  System RAM       : {hw['system_ram_gb']} GB | Free Disk: {hw['free_nvme_gb']} GB")
    if hw.get("selected_compute_gpus"):
        gdesc = ", ".join(f"GPU {g['index']}: {g['name']} ({g['vram_gb']} GB)" for g in hw["selected_compute_gpus"])
        print(f"  Compute GPU(s)   : {gdesc}")
    if hw.get("ignored_gpus"):
        idesc = ", ".join(f"GPU {g['index']}: {g['name']} [{g['reason']}]" for g in hw["ignored_gpus"])
        print(f"  Ignored GPU(s)   : {idesc}")
    print(f"  Rec. Profile     : {hw['recommended_profile_id']} (quant={hw['recommended_quantization']})")
    print("==========================================================================")
    print("  Available installation tiers for your disk & GPU:")
    for idx, key in enumerate(("minimal", "moderate", "aggressive"), start=1):
        p = plans[key]
        default_tag = " [DEFAULT]" if key == "moderate" else ""
        fit_tag = "" if p["fits_disk"] else " [EXCEEDS FREE DISK]"
        print(
            f"    [{idx}] {key.upper()}{default_tag}{fit_tag}: "
            f"--component {p['component']} --quant {p['quantization']} (~{p['approx_gb']} GB)"
        )
        print(f"        {p['description']}")

    chosen_tier = args.tier
    if chosen_tier is None:
        if args.yes or args.dry_run:
            chosen_tier = "moderate"
        else:
            raw = input("\nChoose tier [1=minimal, 2=moderate (default), 3=aggressive] or [q]uit: ").strip().lower()
            if raw in ("q", "quit", "exit"):
                print("[openfal autosetup] Aborted.")
                return
            tier_map = {"": "moderate", "1": "minimal", "minimal": "minimal", "2": "moderate", "moderate": "moderate", "3": "aggressive", "aggressive": "aggressive"}
            if raw not in tier_map:
                print(f"[openfal autosetup] Invalid choice '{raw}'. Aborted.", file=sys.stderr)
                sys.exit(1)
            chosen_tier = tier_map[raw]

    plan = plans[chosen_tier]
    print(
        f"\n[openfal autosetup] Selected '{chosen_tier}' -> component={plan['component']}, "
        f"quant={plan['quantization']} (~{plan['approx_gb']} GB) -> {weights_dir.resolve()}"
    )

    if args.dry_run:
        print("[openfal autosetup] Dry run complete (no files downloaded).")
        return

    if not args.yes and args.tier is None:
        confirm = input("Proceed with download/setup? [Y/n]: ").strip().lower()
        if confirm in ("n", "no"):
            print("[openfal autosetup] Aborted.")
            return

    mode = "strata_repack" if args.pack_only else "hf_hub"
    task = _run_download_with_progress(
        mgr=mgr,
        weights_dir=weights_dir,
        component_id=plan["component"],
        quantization=plan["quantization"],
        mode=mode,
        hf_token=args.hf_token,
        verbose=getattr(args, "verbose", False),
    )
    if task.status == "FAILED":
        print(f"[openfal autosetup] FAILED: {task.error}", file=sys.stderr)
        sys.exit(1)

    print(f"\n[openfal autosetup] Complete! Start your server with:")
    print(f"  openfal serve --profile {hw['recommended_profile_id']} --port 8000")


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
    p_dl.add_argument("-v", "--verbose", action="store_true", help="Show full raw Hugging Face tqdm bars and inline logs")
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

    p_auto = sub.add_parser(
        "autosetup",
        help="Auto-detect GPU/RAM/Disk and install the best model tier (minimal, moderate, or aggressive)",
    )
    p_auto.add_argument("--weights-dir", default="./weights/MiniMax-H3", help="Target directory for weights")
    p_auto.add_argument(
        "--tier",
        choices=["minimal", "moderate", "aggressive"],
        default=None,
        help="Skip tier question and install the specified tier directly",
    )
    p_auto.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Non-interactive mode: skip prompts (defaults to 'moderate' tier unless --tier is set)",
    )
    p_auto.add_argument("--dry-run", action="store_true", help="Preview detected hardware and tier plans without downloading")
    p_auto.add_argument("--pack-only", action="store_true", help="Pack structural/local shards without contacting Hugging Face")
    p_auto.add_argument("--hf-token", default=None, help="Optional Hugging Face token")
    p_auto.add_argument("-v", "--verbose", action="store_true", help="Show full raw Hugging Face tqdm bars and inline logs")
    p_auto.set_defaults(func=cmd_autosetup)

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
