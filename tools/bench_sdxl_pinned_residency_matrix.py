from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - optional benchmark dependency
    psutil = None

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    try:
        from backend.staging_manager import ExecutionClass as _ExecutionClass

        if isinstance(value, _ExecutionClass):
            return value.value
    except Exception:
        pass
    return str(value)


def _prompt_hash(prompt: str, negative_prompt: str) -> str:
    payload = f"{prompt}\n---\n{negative_prompt}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _parse_execution_class(value: str):
    from backend.staging_manager import ExecutionClass

    try:
        return ExecutionClass[value]
    except KeyError as exc:  # pragma: no cover - argparse guards normal use
        raise argparse.ArgumentTypeError(f"Unknown execution class: {value}") from exc


def _parse_lora_spec(spec: str) -> tuple[str, float]:
    if ":" not in spec:
        return spec, 1.0
    if len(spec) >= 3 and spec[1] == ":" and spec[2] in ("\\", "/") and spec.count(":") == 1:
        return spec, 1.0
    source, weight_text = spec.rsplit(":", 1)
    if not source:
        raise argparse.ArgumentTypeError("LoRA spec must include a source path before ':'")
    try:
        weight = float(weight_text)
    except ValueError as exc:  # pragma: no cover - argparse guards normal use
        if len(spec) >= 3 and spec[1] == ":" and spec[2] in ("\\", "/"):
            return spec, 1.0
        raise argparse.ArgumentTypeError(f"Invalid LoRA weight: {weight_text}") from exc
    return source, weight


@dataclass
class MemorySnapshot:
    peak_rss_bytes: int = 0
    peak_vram_allocated_bytes: int = 0
    peak_vram_reserved_bytes: int = 0


@dataclass
class PhaseMemorySnapshot:
    phase: str
    rss_bytes: int
    vram_allocated_bytes: int
    vram_reserved_bytes: int


@dataclass
class PinnedResidencyProbe:
    cache_size_before: int = 0
    cache_size_after_first: int = 0
    cache_size_after_second: int = 0
    first_wall: float = 0.0
    second_wall: float = 0.0
    first_digest: str = ""
    second_digest: str = ""
    reuse_confirmed: bool = False


class MemorySampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process = psutil.Process() if psutil is not None else None
        self.snapshot = MemorySnapshot()

    def __enter__(self) -> "MemorySampler":
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set() and self._process is not None:
            try:
                rss = int(self._process.memory_info().rss)
                if rss > self.snapshot.peak_rss_bytes:
                    self.snapshot.peak_rss_bytes = rss
            except Exception:
                pass
            time.sleep(self.interval_s)

    def __exit__(self, exc_type, exc, tb) -> None:
        import torch

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.snapshot.peak_vram_allocated_bytes = int(torch.cuda.max_memory_allocated())
            self.snapshot.peak_vram_reserved_bytes = int(torch.cuda.max_memory_reserved())


def _capture_phase_memory(phase: str) -> PhaseMemorySnapshot:
    import torch

    rss_bytes = 0
    if psutil is not None:
        try:
            rss_bytes = int(psutil.Process().memory_info().rss)
        except Exception:
            rss_bytes = 0

    vram_allocated_bytes = 0
    vram_reserved_bytes = 0
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            vram_allocated_bytes = int(torch.cuda.memory_allocated())
            vram_reserved_bytes = int(torch.cuda.memory_reserved())
        except Exception:
            pass

    return PhaseMemorySnapshot(
        phase=phase,
        rss_bytes=rss_bytes,
        vram_allocated_bytes=vram_allocated_bytes,
        vram_reserved_bytes=vram_reserved_bytes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pinned-host SDXL GGUF residency modes with optional prompt-cache and LoRA probes.",
    )
    parser.add_argument("--runs", type=int, default=2, help="Total runs including the cold run.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "P4-M13-W07b-R"))
    parser.add_argument("--scenario", default="mission_q8_acceptance", help="Scenario name from modules.gguf_headless_runner.scenario_library().")
    parser.add_argument(
        "--route",
        default="both",
        choices=("direct", "true_streaming", "both"),
        help="Select which SDXL GGUF runtime route(s) to benchmark.",
    )
    parser.add_argument(
        "--clip-modes",
        nargs="+",
        default=("gpu_then_offload", "cpu_only"),
        choices=("gpu_then_offload", "cpu_only"),
        help="Benchmark one or both CLIP residency modes.",
    )
    parser.add_argument(
        "--direct-execution-class",
        default="SDXL_RESIDENT_T2",
        help="Execution class to use for the direct GGUF route.",
    )
    parser.add_argument("--unet-path", default="", help="Optional UNet override path for the benchmark.")
    parser.add_argument("--clip-path", default="", help="Optional CLIP override path for the benchmark.")
    parser.add_argument("--vae-path", default="", help="Optional VAE override path for the benchmark.")
    parser.add_argument(
        "--unet-budget-mb",
        type=int,
        default=None,
        help="Optional low-vram budget to pass to the runtime attach path.",
    )
    parser.add_argument(
        "--lora",
        dest="lora_specs",
        action="append",
        default=[],
        help="Optional LoRA spec in the form path[:weight]. Repeat the flag for multiple LoRAs.",
    )
    parser.add_argument(
        "--lora-state",
        default="both",
        choices=("off", "on", "both"),
        help="Select whether to run LoRA-off control rows, LoRA-on rows, or both.",
    )
    parser.add_argument(
        "--stage-conditioning-to-gpu",
        action="store_true",
        default=False,
        help="Stage encoded prompt conditioning to GPU before ADM construction.",
    )
    parser.add_argument(
        "--probe-cache",
        dest="probe_cache",
        action="store_true",
        default=True,
        help="Probe the SDXL prompt-conditioning cache using the loaded clip.",
    )
    parser.add_argument(
        "--skip-cache-probe",
        dest="probe_cache",
        action="store_false",
        help="Skip the prompt-conditioning cache probe.",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--traceback", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_default) + "\n")


def _clone_scenario(args: argparse.Namespace):
    from dataclasses import replace
    from modules.gguf_headless_runner import scenario_library

    library = scenario_library()
    if args.scenario not in library:
        raise ValueError(f"Unknown scenario {args.scenario!r}. Available: {', '.join(sorted(library.keys()))}")
    scenario = library[args.scenario]
    if args.notes:
        scenario = replace(scenario, notes=(scenario.notes + " " + args.notes).strip())
    return scenario


def _resolve_local_model_path(path: str, folders: list[str]) -> str:
    import os
    from modules.util import get_file_from_folder_list

    if os.path.isfile(path):
        return path
    if not folders:
        return path
    parts = [part for part in Path(path).parts if part not in ("", os.path.sep)]
    for folder in folders:
        folder = str(folder)
        for suffix_len in range(1, len(parts) + 1):
            candidate = os.path.join(folder, *parts[-suffix_len:])
            if os.path.isfile(candidate):
                return candidate
    basename = os.path.basename(path)
    resolved = get_file_from_folder_list(basename, folders)
    if os.path.isfile(resolved):
        return resolved
    return path


def _as_folder_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return [str(value)]


def _resolve_scenario_against_config(scenario, *, unet_path: str = "", clip_path: str = "", vae_path: str = ""):
    import modules.config as config
    from dataclasses import replace

    selected_model = unet_path or getattr(config, "default_base_model_name", None) or getattr(config, "default_model", None) or scenario.unet_path
    default_clip = clip_path or getattr(config, "default_clip", None) or scenario.clip_l_path
    default_vae = vae_path or getattr(config, "default_vae", None) or scenario.vae_path

    return replace(
        scenario,
        unet_path=_resolve_local_model_path(
            str(selected_model),
            _as_folder_list(getattr(config, "paths_checkpoints", [])) + _as_folder_list(getattr(config, "path_unet", [])),
        ),
        clip_l_path=_resolve_local_model_path(str(default_clip), _as_folder_list(getattr(config, "paths_clips", []))),
        clip_g_path=_resolve_local_model_path(str(default_clip), _as_folder_list(getattr(config, "paths_clips", []))),
        vae_path=_resolve_local_model_path(str(default_vae), _as_folder_list(getattr(config, "path_vae", []))),
    )


def _save_png(path: Path, image: torch.Tensor) -> None:
    import numpy as np
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(image_uint8).save(path)


def _tensor_digest(tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return ""
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest()


def _clip_encode_digest(result: Any) -> str:
    payload: list[Any] = []
    for cond, meta in result or []:
        payload.append(
            {
                "cond": _tensor_digest(cond),
                "pooled": _tensor_digest(meta.get("pooled_output")) if isinstance(meta, dict) else "",
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_runtime_config(
    scenario,
    *,
    clip_residency_mode: str,
    execution_class: Any,
    stage_prompt_conditioning_to_device: bool,
):
    from backend.gguf.direct_sdxl_runtime import DirectSDXLGGUFRunConfig

    return DirectSDXLGGUFRunConfig(
        unet_path=scenario.unet_path,
        clip_l_path=scenario.clip_l_path,
        clip_g_path=scenario.clip_g_path,
        vae_path=scenario.vae_path,
        prompt=scenario.prompt,
        negative_prompt=scenario.negative_prompt,
        width=scenario.width,
        height=scenario.height,
        steps=scenario.steps,
        cfg=scenario.cfg,
        sampler=scenario.sampler,
        scheduler=scenario.scheduler,
        seed=scenario.seed,
        clip_layer=scenario.clip_layer,
        clip_residency_mode=clip_residency_mode,
        denoise=scenario.denoise,
        batch_size=scenario.batch_size,
        quality=scenario.quality.as_sampling_dict(),
        execution_class=execution_class,
        stage_prompt_conditioning_to_device=stage_prompt_conditioning_to_device,
    )


def _build_runtime(route: str, config, *, device, unet_budget_mb: Optional[int]):
    from backend.gguf.direct_sdxl_runtime import DirectSDXLGGUFRuntime
    from tools.gguf_true_streaming_runtime import TrueStreamingSDXLGGUFRuntime

    if route == "true_streaming":
        return TrueStreamingSDXLGGUFRuntime(config, device=device, unet_budget_mb=unet_budget_mb)
    return DirectSDXLGGUFRuntime(config, device=device, unet_budget_mb=unet_budget_mb)


def _apply_loras_to_runtime(
    runtime: Any,
    *,
    scenario,
    lora_specs: list[tuple[str, float]],
) -> dict[str, Any]:
    import modules.core as core
    import modules.model_taxonomy as model_taxonomy

    if not lora_specs:
        return {
            "enabled": False,
            "spec_count": 0,
            "artifact_count": 0,
            "artifact_sources": [],
            "artifact_scales": [],
            "refresh_loras_wall": 0.0,
        }

    if runtime.unet is None or runtime.clip is None:
        raise RuntimeError("Runtime must load UNet and CLIP before applying LoRAs.")

    model = core.StableDiffusionModel(
        unet=runtime.unet,
        clip=runtime.clip,
        filename=scenario.unet_path,
        compatibility_family="gguf",
        architecture=model_taxonomy.ARCHITECTURE_SDXL,
    )
    start = time.perf_counter()
    model.refresh_loras(lora_specs)
    refresh_loras_wall = time.perf_counter() - start

    runtime.unet = model.unet_with_lora
    runtime.clip = model.clip_with_lora
    if runtime.clip is not None and not hasattr(runtime.clip, "fcs_cond_cache"):
        runtime.clip.fcs_cond_cache = {}
    if runtime.unet is not None and not hasattr(runtime.unet, "fcs_cond_cache"):
        runtime.unet.fcs_cond_cache = {}

    return {
        "enabled": True,
        "spec_count": len(lora_specs),
        "artifact_count": len(model.lora_artifact_registry),
        "artifact_sources": [artifact.source_path for artifact in model.lora_artifact_registry],
        "artifact_scales": [float(artifact.default_scale) for artifact in model.lora_artifact_registry],
        "refresh_loras_wall": refresh_loras_wall,
        "artifact_registry": model.lora_artifact_registry,
    }


def _probe_prompt_cache(
    runtime: Any,
    *,
    scenario,
    execution_class: Any,
    clip_residency_mode: str,
    lora_artifact_registry: tuple[Any, ...] = (),
) -> PinnedResidencyProbe:
    import modules.default_pipeline as default_pipeline
    import modules.model_taxonomy as model_taxonomy
    from backend import sdxl_runtime_policy

    previous_final_clip = default_pipeline.final_clip
    previous_model_base = default_pipeline.model_base

    try:
        if runtime.clip is None:
            raise RuntimeError("Runtime clip must be loaded before probing prompt cache.")
        runtime.clip.fcs_cond_cache = {}
        default_pipeline.final_clip = runtime.clip
        default_pipeline.model_base = SimpleNamespace(
            filename=scenario.unet_path,
            architecture=model_taxonomy.ARCHITECTURE_SDXL,
            compatibility_family="gguf",
            lora_artifact_registry=lora_artifact_registry,
            sdxl_residency_class=execution_class.value,
            sdxl_execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_GGUF_STAGED,
            sdxl_clip_residency_mode=clip_residency_mode,
        )
        default_pipeline.clear_all_caches()

        probe = PinnedResidencyProbe()
        probe.cache_size_before = len(runtime.clip.fcs_cond_cache)

        start = time.perf_counter()
        first = default_pipeline.clip_encode(
            [scenario.prompt, scenario.negative_prompt],
            pool_top_k=1,
            route_family="gguf",
            residency_class=execution_class.value,
            execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_GGUF_STAGED,
            clip_residency_mode=clip_residency_mode,
        )
        probe.first_wall = time.perf_counter() - start
        probe.cache_size_after_first = len(runtime.clip.fcs_cond_cache)
        probe.first_digest = _clip_encode_digest(first)

        start = time.perf_counter()
        second = default_pipeline.clip_encode(
            [scenario.prompt, scenario.negative_prompt],
            pool_top_k=1,
            route_family="gguf",
            residency_class=execution_class.value,
            execution_family=sdxl_runtime_policy.EXECUTION_FAMILY_GGUF_STAGED,
            clip_residency_mode=clip_residency_mode,
        )
        probe.second_wall = time.perf_counter() - start
        probe.cache_size_after_second = len(runtime.clip.fcs_cond_cache)
        probe.second_digest = _clip_encode_digest(second)
        probe.reuse_confirmed = (
            probe.first_digest == probe.second_digest
            and probe.cache_size_after_first == probe.cache_size_after_second
            and probe.cache_size_after_first >= 2
        )
        return probe
    finally:
        default_pipeline.final_clip = previous_final_clip
        default_pipeline.model_base = previous_model_base


def _combo_name(route: str, clip_mode: str, lora_enabled: bool) -> str:
    lora_tag = "lora_on" if lora_enabled else "lora_off"
    return f"{route}_{clip_mode}_{lora_tag}"


def _run_combo(
    *,
    route: str,
    scenario,
    run_dir: Path,
    run_label: str,
    clip_mode: str,
    execution_class: Any,
    lora_specs: list[tuple[str, float]],
    unet_budget_mb: Optional[int],
    probe_cache: bool,
    stage_conditioning_to_gpu: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import gc
    import torch

    config = _build_runtime_config(
        scenario,
        clip_residency_mode=clip_mode,
        execution_class=execution_class,
        stage_prompt_conditioning_to_device=stage_conditioning_to_gpu,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runtime = _build_runtime(route, config, device=device, unet_budget_mb=unet_budget_mb)

    cache_probe: Optional[PinnedResidencyProbe] = None
    lora_artifact_registry: tuple[Any, ...] = ()
    phase_memory: list[PhaseMemorySnapshot] = []
    lora_probe = {
        "enabled": False,
        "spec_count": 0,
        "artifact_count": 0,
        "artifact_sources": [],
        "artifact_scales": [],
        "refresh_loras_wall": 0.0,
    }
    payload: Optional[dict[str, Any]] = None

    try:
        start = time.perf_counter()
        with MemorySampler() as memory:
            runtime.load_components()
            phase_memory.append(_capture_phase_memory("after_load_components"))

            if lora_specs:
                lora_probe = _apply_loras_to_runtime(runtime, scenario=scenario, lora_specs=lora_specs)
                lora_artifact_registry = tuple(lora_probe.pop("artifact_registry", ()))
                phase_memory.append(_capture_phase_memory("after_lora_refresh"))

            if probe_cache:
                cache_probe = _probe_prompt_cache(
                    runtime,
                    scenario=scenario,
                    execution_class=execution_class,
                    clip_residency_mode=clip_mode,
                    lora_artifact_registry=lora_artifact_registry,
                )
                phase_memory.append(_capture_phase_memory("after_prompt_cache_probe"))

            prepared_inputs, prep_metrics = runtime.prepare_inputs()
            phase_memory.append(_capture_phase_memory("after_prepare_inputs"))

            denoise_result = runtime.denoise_prepared_inputs(prepared_inputs)
            phase_memory.append(_capture_phase_memory("after_denoise"))

            images, vae_attach, vae_decode = runtime.decode_latent(denoise_result.samples)
            phase_memory.append(_capture_phase_memory("after_decode"))

            result = SimpleNamespace(
                images=images,
                latents=denoise_result.samples,
                benchmark={
                    "route_label": runtime.route_label,
                    "clip_residency_mode": runtime.config.clip_residency_mode,
                    "cold_model_load_cpu": getattr(runtime, "_cold_model_load_cpu", 0.0),
                    "clip_residency_attach": prep_metrics["clip_residency_attach"],
                    "clip_residency_offload": prep_metrics["clip_residency_offload"],
                    "clip_encode": prep_metrics["clip_encode"],
                    "conditioning_stage_to_device": prep_metrics.get("conditioning_stage_to_device", 0.0),
                    "adm_build": prep_metrics["adm_build"],
                    "latent_noise_prep": prep_metrics["latent_noise_prep"],
                    "sampler_model_attach": denoise_result.sampler_model_attach,
                    "cond_prepare_explicit": denoise_result.cond_prepare_duration,
                    "denoise_wall": denoise_result.denoise_wall,
                    "denoise_s_per_it": denoise_result.denoise_wall / max(1, runtime.config.steps),
                    "denoise_cpu_proc": denoise_result.denoise_cpu_proc,
                    "gguf_dequant": float(denoise_result.gguf_trace_stats.get("dequant_seconds", 0.0)),
                    "gguf_dequant_cpu_proc": float(denoise_result.gguf_trace_stats.get("dequant_cpu_process_seconds", 0.0)),
                    "vae_attach": vae_attach,
                    "vae_decode": vae_decode,
                    "total_wall": 0.0,
                },
            )
            total_wall = time.perf_counter() - start
            result.benchmark["total_wall"] = total_wall

            image_path = run_dir / f"{scenario.name}_{_combo_name(route, clip_mode, bool(lora_specs))}_{run_label}.png"
            image_save_start = time.perf_counter()
            _save_png(image_path, result.images[0])
            image_save = time.perf_counter() - image_save_start

            payload: dict[str, Any] = {
                "scenario": scenario.name,
                "route": route,
                "route_label": result.benchmark.get("route_label", route),
                "run_label": run_label,
                "prompt_hash": _prompt_hash(scenario.prompt, scenario.negative_prompt),
                "quant_model": Path(scenario.unet_path).name,
                "resolution": f"{scenario.width}x{scenario.height}",
                "steps": scenario.steps,
                "cfg": scenario.cfg,
                "sampler": scenario.sampler,
                "scheduler": scenario.scheduler,
                "seed": scenario.seed,
                "batch_size": scenario.batch_size,
                "clip_residency_mode": clip_mode,
                "execution_class": execution_class.value,
                "lora_probe": lora_probe,
                "prompt_cache_probe": asdict(cache_probe) if cache_probe is not None else {},
                "benchmark": dict(result.benchmark),
                "phase_memory": [asdict(snapshot) for snapshot in phase_memory],
                "image_path": str(image_path),
                "image_save": image_save,
                "total_wall": total_wall,
                "peak_rss_bytes": memory.snapshot.peak_rss_bytes,
                "peak_vram_allocated_bytes": memory.snapshot.peak_vram_allocated_bytes,
                "peak_vram_reserved_bytes": memory.snapshot.peak_vram_reserved_bytes,
                "notes": getattr(scenario, "notes", ""),
            }
            if "cold_model_load_cpu" not in payload["benchmark"]:
                payload["benchmark"]["cold_model_load_cpu"] = 0.0

    finally:
        runtime.close()
        gc.collect()
        if payload is not None:
            payload["post_close_memory"] = asdict(_capture_phase_memory("after_close"))

    if payload is None:
        raise RuntimeError("Benchmark payload was not produced.")
    return payload, dict(payload["benchmark"])


def main() -> int:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be at least 1")

    if isinstance(args.direct_execution_class, str):
        args.direct_execution_class = _parse_execution_class(args.direct_execution_class)

    run_dir = Path(args.output_dir)
    original_sys_argv = list(sys.argv)
    sys.argv = [sys.argv[0]]
    try:
        from backend.staging_manager import ExecutionClass
        from modules.gguf_headless_runner import collect_environment, write_environment_report

        scenario = _clone_scenario(args)
        scenario = _resolve_scenario_against_config(
            scenario,
            unet_path=args.unet_path,
            clip_path=args.clip_path,
            vae_path=args.vae_path,
        )
        run_dir_root = run_dir
        run_dir_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = run_dir_root / f"{scenario.name}_pinned_residency_matrix_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        environment = collect_environment("pinned_residency_matrix", scenario)
        write_environment_report(environment, run_dir)

        routes = ("direct", "true_streaming") if args.route == "both" else (args.route,)
        clip_modes = tuple(args.clip_modes)
        lora_specs = [_parse_lora_spec(spec) for spec in args.lora_specs]
        if not lora_specs:
            lora_states = (False,)
        elif args.lora_state == "off":
            lora_states = (False,)
        elif args.lora_state == "on":
            lora_states = (True,)
        else:
            lora_states = (False, True)
        if args.lora_state == "on" and not lora_specs:
            raise ValueError("--lora-state on requires at least one --lora spec")
        run_labels = ["cold"] + [f"warm_{index}" for index in range(1, args.runs)]

        results: list[dict[str, Any]] = []
        for route in routes:
            execution_class = args.direct_execution_class if route == "direct" else ExecutionClass.SDXL_STREAMING_T1
            for clip_mode in clip_modes:
                for lora_enabled in lora_states:
                    combo_dir = run_dir / _combo_name(route, clip_mode, lora_enabled)
                    combo_dir.mkdir(parents=True, exist_ok=True)
                    active_loras = lora_specs if lora_enabled else []
                    for run_label in run_labels:
                        payload, raw_result = _run_combo(
                            route=route,
                            scenario=scenario,
                            run_dir=combo_dir,
                            run_label=run_label,
                            clip_mode=clip_mode,
                            execution_class=execution_class,
                            lora_specs=active_loras,
                            unet_budget_mb=args.unet_budget_mb,
                            probe_cache=args.probe_cache,
                            stage_conditioning_to_gpu=args.stage_conditioning_to_gpu,
                        )
                        _append_jsonl(combo_dir / "benchmark_results.jsonl", payload)
                        results.append(payload)
                        print(
                            json.dumps(
                                {
                                    "run": run_label,
                                    "route": route,
                                    "clip_mode": clip_mode,
                                    "lora": "on" if lora_enabled else "off",
                                    "cache_reuse": payload.get("prompt_cache_probe", {}).get("reuse_confirmed", False),
                                    "denoise_s_per_it": round(float(raw_result.get("denoise_s_per_it", 0.0)), 4),
                                    "total_wall": round(float(payload.get("total_wall", 0.0)), 4),
                                    "peak_rss_mb": round(float(payload.get("peak_rss_bytes", 0)) / (1024 * 1024), 2),
                                    "peak_vram_mb": round(float(payload.get("peak_vram_reserved_bytes", 0)) / (1024 * 1024), 2),
                                    "image_path": payload.get("image_path", ""),
                                },
                                default=_json_default,
                            )
                        )

        summary = {
            "environment": asdict(environment),
            "scenario": asdict(scenario),
            "matrix": {
                "routes": list(routes),
                "clip_modes": list(clip_modes),
                "lora_enabled": bool(lora_specs),
                "lora_state": args.lora_state,
                "stage_conditioning_to_gpu": bool(args.stage_conditioning_to_gpu),
                "direct_execution_class": args.direct_execution_class.value,
                "probe_cache": args.probe_cache,
                "unet_budget_mb": args.unet_budget_mb,
            },
            "results": results,
            "output_dir": str(run_dir),
            "notes": "Tool-only pinned residency matrix for SDXL GGUF comparison.",
        }
        _write_json(run_dir / "summary.json", summary)
        print(json.dumps({"summary": str(run_dir / "summary.json"), "output_dir": str(run_dir)}, default=_json_default))
        return 0
    except Exception as exc:
        error = {
            "status": "error",
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            "output_dir": str(run_dir),
        }
        if args.traceback:
            import traceback

            error["traceback"] = traceback.format_exc()
        _write_json(run_dir / "error.json", error)
        print(json.dumps(error, default=_json_default))
        return 1
    finally:
        sys.argv = original_sys_argv


if __name__ == "__main__":
    raise SystemExit(main())
