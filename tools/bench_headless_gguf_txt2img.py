from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _import_runner_api():
    """
    Import the benchmark runner while shielding repository-global argparse setup
    from this tool''s CLI flags.
    """
    original_argv = list(sys.argv)
    try:
        sys.argv = [original_argv[0]]
        from modules.gguf_headless_runner import (
            HeadlessGGUFRunner,
            append_metrics_jsonl,
            collect_environment,
            scenario_library,
            write_environment_report,
        )
    finally:
        sys.argv = original_argv

    return {
        "HeadlessGGUFRunner": HeadlessGGUFRunner,
        "append_metrics_jsonl": append_metrics_jsonl,
        "collect_environment": collect_environment,
        "scenario_library": scenario_library,
        "write_environment_report": write_environment_report,
    }


RUNNER_API = _import_runner_api()
HeadlessGGUFRunner = RUNNER_API["HeadlessGGUFRunner"]
append_metrics_jsonl = RUNNER_API["append_metrics_jsonl"]
collect_environment = RUNNER_API["collect_environment"]
scenario_library = RUNNER_API["scenario_library"]
write_environment_report = RUNNER_API["write_environment_report"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the W03 headless GGUF txt2img runner.")
    parser.add_argument(
        "--route",
        required=True,
        choices=["headless_intermediate", "headless_clean", "backend_explicit", "direct_sdxl_gguf", "glass_sdxl_gguf"],
        help="Residency path to benchmark.",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=sorted(scenario_library().keys()),
        help="Named benchmark scenario.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Total runs including the cold run.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "P4-M09-W03"))
    parser.add_argument("--force-high-vram", action="store_true")
    parser.add_argument("--unet-budget-mb", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--unet-path", default=None)
    parser.add_argument("--clip-path", default=None, help="Override both clip_l_path and clip_g_path.")
    parser.add_argument("--vae-path", default=None)
    parser.add_argument("--notes", default=None, help="Append a free-form note to the scenario.")
    parser.add_argument("--route-checkpoints", action="store_true", help="Enable route checkpoint metadata capture for checkpoint-capable routes.")
    parser.add_argument("--glass-checkpoints", action="store_true", help="Alias for --route-checkpoints retained for W03 glass-route commands.")
    parser.add_argument("--glass-checkpoint-tensors", action="store_true", help="Persist full glass checkpoint tensors when checkpointing is enabled.")
    parser.add_argument(
        "--glass-ancestral-noise-policy",
        choices=["direct_compatible", "seeded"],
        default="direct_compatible",
        help="Control ancestral noise RNG policy for the glass route; default matches the M08 direct runtime.",
    )
    parser.add_argument(
        "--glass-checkpoint-steps",
        default=None,
        help="Comma-separated step indices for full tensor persistence; metadata is still captured for all steps.",
    )
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace):
    scenario = scenario_library()[args.scenario]
    updates = {}
    if args.prompt is not None:
        updates["prompt"] = args.prompt
    if args.negative_prompt is not None:
        updates["negative_prompt"] = args.negative_prompt
    if args.width is not None:
        updates["width"] = args.width
    if args.height is not None:
        updates["height"] = args.height
    if args.steps is not None:
        updates["steps"] = args.steps
    if args.cfg is not None:
        updates["cfg"] = args.cfg
    if args.seed is not None:
        updates["seed"] = args.seed
    if args.unet_path is not None:
        updates["unet_path"] = args.unet_path
    if args.clip_path is not None:
        updates["clip_l_path"] = args.clip_path
        updates["clip_g_path"] = args.clip_path
    if args.vae_path is not None:
        updates["vae_path"] = args.vae_path
    if args.notes is not None:
        base_notes = scenario.notes.strip()
        updates["notes"] = (base_notes + " " + args.notes).strip() if base_notes else args.notes
    return replace(scenario, **updates) if updates else scenario


def ensure_paths_exist(scenario) -> None:
    missing = []
    for label, value in {
        "unet": scenario.unet_path,
        "clip_l": scenario.clip_l_path,
        "clip_g": scenario.clip_g_path,
        "vae": scenario.vae_path,
    }.items():
        if not Path(value).exists():
            missing.append(f"{label}={value}")
    if missing:
        raise FileNotFoundError("Missing model paths: " + ", ".join(missing))


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    scenario = apply_overrides(args)
    ensure_paths_exist(scenario)

    original_argv = list(sys.argv)
    sys.argv = [original_argv[0]]
    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = Path(args.output_dir) / f"{scenario.name}_{args.route}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        environment = collect_environment(args.route, scenario)
        write_environment_report(environment, output_dir)

        checkpoint_steps = None
        if args.glass_checkpoint_steps:
            checkpoint_steps = [int(value.strip()) for value in args.glass_checkpoint_steps.split(",") if value.strip()]

        runner = HeadlessGGUFRunner(
            scenario,
            args.route,
            force_high_vram=args.force_high_vram,
            explicit_unet_budget_mb=args.unet_budget_mb,
            checkpoint_enabled=args.route_checkpoints or args.glass_checkpoints,
            checkpoint_persist_full_tensors=args.glass_checkpoint_tensors,
            checkpoint_persist_steps=checkpoint_steps,
            glass_ancestral_noise_policy=args.glass_ancestral_noise_policy,
        )

        results = []
        run_labels = ["cold"] + [f"warm_{index}" for index in range(1, args.runs)]

        try:
            for run_label in run_labels:
                metrics = runner.run_once(run_label, output_dir)
                append_metrics_jsonl(metrics, output_dir)
                results.append(asdict(metrics))
                print(
                    json.dumps(
                        {
                            "scenario": metrics.scenario,
                            "route": metrics.route_label,
                            "run": metrics.run_label,
                            "denoise_s_per_it": round(metrics.denoise_s_per_it, 4),
                            "total_wall": round(metrics.total_wall, 4),
                            "image_path": metrics.image_path,
                        }
                    )
                )
        finally:
            runner.close()

        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "environment": asdict(environment),
                    "scenario": asdict(scenario),
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"summary": str(summary_path), "output_dir": str(output_dir)}))
        return 0
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())


