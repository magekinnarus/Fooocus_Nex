from __future__ import annotations

import gc
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from PIL import Image

from backend import conditioning, decode as backend_decode, inference_lifecycle, loader, precision, resources, sampling
from backend.gguf.direct_sdxl_runtime import DirectSDXLGGUFRuntime, DirectSDXLGGUFRunConfig
from backend.gguf.sdxl_glass_pipeline import GlassSDXLGGUFCheckpointConfig, GlassSDXLGGUFRunConfig, GlassSDXLGGUFPipeline
from ldm_patched.modules import latent_formats
from ldm_patched.modules import model_base as apply_model_trace

try:
    from backend.gguf import ops as gguf_ops
except Exception:  # pragma: no cover - import safety
    gguf_ops = None


DEFAULT_POSITIVE_PROMPT = (
    "a futuristic cityscape with flying cars and neon lights, cyberpunk style, "
    "1girl, pretty girl, full body shot, purple hair, red glowing eyes, "
    "smiling,highly detailed"
)

DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, text, watermark"

COMFY_POSITIVE_PROMPT = (
    "score_9, score_8_up, score_8, score_9, score_8_up, score_8, HiRes, "
    "(shallow depth of field:1.4), Chiaroscuro Lighting Style, (epiCPhoto:1.4), "
    "lights off, dim light, (lens flare:.4), (cinematic bokeh:1.3), pretty girl,"
    "(( rustic interior, diagonal 3/4 view, side back view, looking at viewer)), "
    "1girl, 18 years old, adorable face, pink hairbows,short torso, small hips, "
    "skinny waist, petite, long blonde hair, (pigtails with bows:1.3), thin nose, "
    "pretty eyes, beautiful face, Hazel green eyes, extremely cute face, twilight "
    "hour, low light, dark, light particles, inside a public cafe, large windows "
    "with street view"
)

COMFY_NEGATIVE_PROMPT = (
    "score_6, score_5, score_4, worst quality:1.2, lowres, bad anatomy, bad hands, "
    "signature, watermarks, ugly, muscular, dick girl, furry, skewed eyes, "
    "unnatural face, unnatural body, error, extra limb, missing limbs, painting by "
    "bad-artist, eyes closed, elderly, ugly, MILF, wrinkled, (watermark)"
)


@dataclass
class QualityConfig:
    sharpness: float = 0.0
    adaptive_cfg: float = 0.0
    adm_scale_positive: float = 1.0
    adm_scale_negative: float = 1.0
    adm_scaler_end: float = 0.3

    def as_sampling_dict(self) -> Dict[str, float]:
        return {
            "sharpness": self.sharpness,
            "adaptive_cfg": self.adaptive_cfg,
            "adm_scale_positive": self.adm_scale_positive,
            "adm_scale_negative": self.adm_scale_negative,
            "adm_scaler_end": self.adm_scaler_end,
        }

    def needs_quality_patch(self) -> bool:
        return any(
            (
                self.sharpness != 0.0,
                self.adaptive_cfg != 0.0,
                self.adm_scale_positive != 1.0,
                self.adm_scale_negative != 1.0,
            )
        )


@dataclass
class ScenarioConfig:
    name: str
    unet_path: str
    clip_l_path: str
    clip_g_path: str
    vae_path: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    seed: int
    clip_layer: int = -2
    denoise: float = 1.0
    batch_size: int = 1
    quality: QualityConfig = field(default_factory=QualityConfig)
    notes: str = ""


@dataclass
class EnvironmentSnapshot:
    gpu_model: str
    total_vram_bytes: Optional[int]
    torch_version: str
    cuda_runtime: Optional[str]
    python_version: str
    os: str
    xformers_version: str
    repo_commit: str
    route_type: str
    model_paths: Dict[str, str]


@dataclass
class RunMetrics:
    scenario: str
    route_label: str
    run_label: str
    prompt_hash: str
    quant_model: str
    resolution: str
    steps: int
    cfg: float
    sampler: str
    scheduler: str
    seed: int
    batch_size: int
    process_start: float
    cold_model_load_cpu: float
    clip_residency_attach: float
    clip_residency_offload: float
    clip_gpu_load: float
    clip_encode: float
    adm_build: float
    sampler_model_attach: float
    unet_gpu_load_or_patch: float
    cond_prepare_explicit: float
    cond_prep: float
    denoise_wall: float
    denoise_s_per_it: float
    denoise_cpu_proc: float
    gguf_dequant: float
    gguf_dequant_cpu_proc: float
    vae_attach: float
    vae_gpu_load: float
    vae_decode: float
    cleanup_reset: float
    image_save: float
    total_wall: float
    image_path: str
    warm_state_annotation: Dict[str, Any] = field(default_factory=dict)
    checkpoint_records_path: str = ""
    checkpoint_record_count: int = 0
    glass_ancestral_noise_policy: str = ""
    gguf_trace_stats: Dict[str, Any] = field(default_factory=dict)
    apply_model_trace_stats: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def _git_short_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _xformers_version() -> str:
    try:
        import xformers  # type: ignore

        return getattr(xformers, "__version__", "unknown")
    except Exception:
        return "not installed"


def collect_environment(route_type: str, scenario: ScenarioConfig) -> EnvironmentSnapshot:
    gpu_model = "cpu"
    total_vram_bytes: Optional[int] = None
    if torch.cuda.is_available():
        gpu_model = torch.cuda.get_device_name(0)
        total_vram_bytes = int(torch.cuda.get_device_properties(0).total_memory)

    return EnvironmentSnapshot(
        gpu_model=gpu_model,
        total_vram_bytes=total_vram_bytes,
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
        python_version=sys.version.replace("\n", " "),
        os=platform.platform(),
        xformers_version=_xformers_version(),
        repo_commit=_git_short_head(),
        route_type=route_type,
        model_paths={
            "unet": scenario.unet_path,
            "clip_l": scenario.clip_l_path,
            "clip_g": scenario.clip_g_path,
            "vae": scenario.vae_path,
        },
    )


def _hash_prompt(prompt: str, negative_prompt: str) -> str:
    import hashlib

    payload = f"{prompt}\n---\n{negative_prompt}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _tensor_digest(tensor: Optional[torch.Tensor]) -> str:
    if tensor is None:
        return ""
    import hashlib

    cpu_tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
    return hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest()


def _tensor_stats(tensor: Optional[torch.Tensor]) -> Dict[str, Any]:
    if tensor is None:
        return {}
    stats_tensor = tensor.detach().float().to(device="cpu")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "mean": float(stats_tensor.mean().item()),
        "std": float(stats_tensor.std(unbiased=False).item()),
        "min": float(stats_tensor.min().item()),
        "max": float(stats_tensor.max().item()),
    }


def _tensor_checkpoint_record(
    name: str,
    tensor: Optional[torch.Tensor],
    *,
    step_index: Optional[int] = None,
    summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "kind": "tensor",
        "step_index": step_index,
        "digest": _tensor_digest(tensor),
        "summary": summary or {},
        "tensor_stats": _tensor_stats(tensor),
        "artifact_path": None,
    }


def _png_save(image: torch.Tensor, destination: Path) -> None:
    image_uint8 = (image.clamp(0.0, 1.0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(image_uint8).save(destination)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _summarize_loaded_models(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    entries = list((state or {}).get("loaded_models", []))
    roles_present = sorted({str(entry.get("role")) for entry in entries if entry.get("role")})
    loaded_memory_by_role: Dict[str, float] = {}
    lowvram_patch_counter_by_role: Dict[str, int] = {}
    patch_uuid_by_role: Dict[str, str] = {}
    devices_by_role: Dict[str, list[str]] = {}

    for entry in entries:
        role = str(entry.get("role") or "unknown")

        loaded_memory_mb = entry.get("loaded_memory_mb")
        if isinstance(loaded_memory_mb, (int, float)):
            loaded_memory_by_role[role] = round(
                loaded_memory_by_role.get(role, 0.0) + float(loaded_memory_mb),
                4,
            )

        lowvram_patch_counter = entry.get("lowvram_patch_counter")
        if isinstance(lowvram_patch_counter, (int, float)):
            current_value = lowvram_patch_counter_by_role.get(role)
            lowvram_patch_counter_by_role[role] = int(
                lowvram_patch_counter
                if current_value is None
                else max(current_value, int(lowvram_patch_counter))
            )

        patch_uuid = entry.get("current_weight_patches_uuid")
        if patch_uuid is not None:
            patch_uuid_by_role[role] = str(patch_uuid)

        current_device = entry.get("current_loaded_device")
        if current_device is not None:
            devices_by_role.setdefault(role, []).append(str(current_device))

    return {
        "loaded_model_count": len(entries),
        "resident_roles": roles_present,
        "role_presence": {role: role in roles_present for role in ("clip", "unet", "vae")},
        "loaded_memory_mb_by_role": loaded_memory_by_role,
        "lowvram_patch_counter_by_role": lowvram_patch_counter_by_role,
        "current_loaded_device_by_role": devices_by_role,
        "patch_uuid_by_role": patch_uuid_by_role,
        "labels": [str(entry.get("label")) for entry in entries if entry.get("label")],
    }


def _numeric_role_delta(before_map: Dict[str, Any], after_map: Dict[str, Any]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for role in sorted(set(before_map) | set(after_map)):
        before_value = before_map.get(role)
        after_value = after_map.get(role)
        if isinstance(before_value, (int, float)) or isinstance(after_value, (int, float)):
            deltas[role] = round(float(after_value or 0.0) - float(before_value or 0.0), 4)
    return deltas


class HeadlessGGUFRunner:
    def __init__(
        self,
        scenario: ScenarioConfig,
        route_label: str,
        *,
        force_high_vram: bool = False,
        explicit_unet_budget_mb: Optional[int] = None,
        checkpoint_enabled: bool = False,
        checkpoint_persist_full_tensors: bool = False,
        checkpoint_persist_steps: Optional[list[int]] = None,
        glass_ancestral_noise_policy: str = "direct_compatible",
    ) -> None:
        if route_label not in {"headless_intermediate", "headless_clean", "backend_explicit", "direct_sdxl_gguf", "glass_sdxl_gguf"}:
            raise ValueError(f"Unsupported route label: {route_label}")

        self.scenario = scenario
        self.route_label = route_label
        self.force_high_vram = force_high_vram
        self.explicit_unet_budget_mb = explicit_unet_budget_mb
        self.checkpoint_enabled = checkpoint_enabled
        self.checkpoint_persist_full_tensors = checkpoint_persist_full_tensors
        self.checkpoint_persist_steps = checkpoint_persist_steps
        self.glass_ancestral_noise_policy = glass_ancestral_noise_policy
        self.device = resources.get_torch_device()
        self._models_loaded = False
        self._process_started = time.perf_counter()
        self._cold_model_load_cpu = 0.0
        self._last_explicit_cond_prep = 0.0

        resources.config.highvram = force_high_vram
        resources.apply_config()

        self.unet = None
        self.clip = None
        self.vae = None
        self.direct_runtime = None
        self.glass_pipeline = None

    def load_models(self) -> float:
        if self._models_loaded:
            return self._cold_model_load_cpu

        start = time.perf_counter()
        if self.route_label == "direct_sdxl_gguf":
            self.direct_runtime = DirectSDXLGGUFRuntime(
                DirectSDXLGGUFRunConfig(
                    unet_path=self.scenario.unet_path,
                    clip_l_path=self.scenario.clip_l_path,
                    clip_g_path=self.scenario.clip_g_path,
                    vae_path=self.scenario.vae_path,
                    prompt=self.scenario.prompt,
                    negative_prompt=self.scenario.negative_prompt,
                    width=self.scenario.width,
                    height=self.scenario.height,
                    steps=self.scenario.steps,
                    cfg=self.scenario.cfg,
                    sampler=self.scenario.sampler,
                    scheduler=self.scenario.scheduler,
                    seed=self.scenario.seed,
                    clip_layer=self.scenario.clip_layer,
                    denoise=self.scenario.denoise,
                    batch_size=self.scenario.batch_size,
                    quality=self.scenario.quality.as_sampling_dict(),
                ),
                device=self.device,
                unet_budget_mb=self.explicit_unet_budget_mb,
            )
            self.direct_runtime.load_components()
            self.unet = self.direct_runtime.unet
            self.clip = self.direct_runtime.clip
            self.vae = self.direct_runtime.vae
        elif self.route_label == "glass_sdxl_gguf":
            self.glass_pipeline = GlassSDXLGGUFPipeline(
                GlassSDXLGGUFRunConfig(
                    unet_path=self.scenario.unet_path,
                    clip_l_path=self.scenario.clip_l_path,
                    clip_g_path=self.scenario.clip_g_path,
                    vae_path=self.scenario.vae_path,
                    prompt=self.scenario.prompt,
                    negative_prompt=self.scenario.negative_prompt,
                    width=self.scenario.width,
                    height=self.scenario.height,
                    steps=self.scenario.steps,
                    cfg=self.scenario.cfg,
                    sampler=self.scenario.sampler,
                    scheduler=self.scenario.scheduler,
                    seed=self.scenario.seed,
                    clip_layer=self.scenario.clip_layer,
                    denoise=self.scenario.denoise,
                    batch_size=self.scenario.batch_size,
                    quality=self.scenario.quality.as_sampling_dict(),
                    ancestral_noise_policy=self.glass_ancestral_noise_policy,
                ),
                device=self.device,
                unet_budget_mb=self.explicit_unet_budget_mb,
            )
            self.glass_pipeline.load_components()
            self.unet = self.glass_pipeline.unet
            self.clip = self.glass_pipeline.clip
            self.vae = self.glass_pipeline.vae
        else:
            self.unet = loader.load_sdxl_unet(self.scenario.unet_path, dtype=torch.float16)
            self.clip = loader.load_sdxl_clip(
                self.scenario.clip_l_path,
                self.scenario.clip_g_path,
                dtype=torch.float16,
            )
            self.clip.clip_layer(self.scenario.clip_layer)
            self.vae = loader.load_vae(
                self.scenario.vae_path,
                dtype=torch.float32,
                latent_format=latent_formats.SDXL(),
            )
            if self.scenario.quality.needs_quality_patch():
                loader.patch_unet_for_quality(self.unet, self.scenario.quality.as_sampling_dict())

        self._cold_model_load_cpu = time.perf_counter() - start
        self._models_loaded = True
        return self._cold_model_load_cpu

    def _measure_intermediate_model_load(self, patchers: list[Any]) -> float:
        start = time.perf_counter()
        resources.load_models_gpu(patchers, force_high_vram=self.force_high_vram)
        return time.perf_counter() - start

    def _measure_explicit_attach(self, model_patcher: Any, stage: str, *, target_phase) -> float:
        attach_result = inference_lifecycle.attach_patcher_for_stage(
            model_patcher,
            stage,
            force_high_vram=self.force_high_vram,
            target_phase=target_phase,
            notes={"route": self.route_label, "scenario": self.scenario.name},
        )
        return float(attach_result.get("duration_s", 0.0))

    def _measure_explicit_detach(self, model_patcher: Any, stage: str) -> float:
        detach_result = inference_lifecycle.detach_patcher_after_stage(
            model_patcher,
            stage,
            flush_cache=False,
            notes={"route": self.route_label, "scenario": self.scenario.name},
        )
        return float(detach_result.get("duration_s", 0.0))

    def _capture_run_boundary_state(self, tag: str, run_label: str) -> Dict[str, Any]:
        return inference_lifecycle.snapshot_inference_state(
            tag,
            notes={"route": self.route_label, "scenario": self.scenario.name, "run": run_label},
        )

    def _clip_cache_entry_count(self) -> int:
        if self.clip is None:
            return 0
        return len(getattr(self.clip, "fcs_cond_cache", {}) or {})

    def _gguf_mmap_released(self) -> Optional[bool]:
        if self.unet is None:
            return None
        value = getattr(self.unet, "mmap_released", None)
        if value is None:
            value = getattr(getattr(self.unet, "model", None), "mmap_released", None)
        if value is None:
            return None
        return bool(value)

    def _build_warm_state_annotation(
        self,
        run_label: str,
        start_state: Dict[str, Any],
        end_state: Dict[str, Any],
        *,
        clip_cache_before: int,
        clip_cache_after: int,
        cleanup_result: Optional[Dict[str, Any]],
        gguf_mmap_released_before: Optional[bool],
        gguf_mmap_released_after: Optional[bool],
    ) -> Dict[str, Any]:
        start_summary = _summarize_loaded_models(start_state)
        end_summary = _summarize_loaded_models(end_state)
        cleanup_mode = "legacy_route_no_explicit_reset"
        cleanup_support_actions = []

        if cleanup_result is not None:
            cleanup_mode = "explicit_finalize_reset"
            cleanup_snapshot = cleanup_result.get("cleanup_snapshot", {})
            cleanup_notes = cleanup_snapshot.get("notes", {}) if isinstance(cleanup_snapshot, dict) else {}
            cleanup_support_actions = list(cleanup_notes.get("support_actions", []) or [])

        return {
            "run_label": run_label,
            "route": self.route_label,
            "cleanup_mode": cleanup_mode,
            "cleanup_support_actions": cleanup_support_actions,
            "clip_cache_entries_before": clip_cache_before,
            "clip_cache_entries_after": clip_cache_after,
            "clip_cache_reused": bool(clip_cache_before > 0),
            "clip_cache_cleared_during_run": bool(clip_cache_before > 0 and clip_cache_after == 0),
            "gguf_mmap_released_before": gguf_mmap_released_before,
            "gguf_mmap_released_after": gguf_mmap_released_after,
            "gguf_first_load_only_behavior_observed": bool(
                gguf_mmap_released_before is False and gguf_mmap_released_after is True
            ),
            "run_start": start_summary,
            "run_end": end_summary,
            "loaded_memory_delta_mb_by_role": _numeric_role_delta(
                start_summary["loaded_memory_mb_by_role"],
                end_summary["loaded_memory_mb_by_role"],
            ),
            "lowvram_patch_counter_delta_by_role": _numeric_role_delta(
                start_summary["lowvram_patch_counter_by_role"],
                end_summary["lowvram_patch_counter_by_role"],
            ),
        }

    def _run_glass_pipeline_once(self, run_label: str, output_dir: Path) -> RunMetrics:
        assert self.glass_pipeline is not None

        total_start = time.perf_counter()
        start_state = self._capture_run_boundary_state("run_start", run_label)
        clip_cache_before = self._clip_cache_entry_count()
        gguf_mmap_released_before = self._gguf_mmap_released()

        prepared_inputs, prep_metrics = self.glass_pipeline.prepare_inputs()
        checkpoint_config = None
        checkpoint_records_path = ""
        if self.checkpoint_enabled:
            tensor_output_dir = output_dir / f"{self.scenario.name}_{self.route_label}_{run_label}_tensors"
            checkpoint_config = GlassSDXLGGUFCheckpointConfig(
                enabled=True,
                persist_full_tensors=self.checkpoint_persist_full_tensors,
                tensor_output_dir=str(tensor_output_dir),
                persist_steps=self.checkpoint_persist_steps,
            )
        if gguf_ops is not None:
            gguf_ops.reset_trace_stats()
        apply_model_trace.reset_apply_model_trace_stats()
        denoise_result = self.glass_pipeline.run_prepared_inputs(
            prepared_inputs,
            prepare_metrics=prep_metrics,
            checkpoint_config=checkpoint_config,
            callback=None,
            disable_pbar=True,
        )
        self.unet = self.glass_pipeline.unet
        self.clip = self.glass_pipeline.clip
        self.vae = self.glass_pipeline.vae

        gguf_dequant = 0.0
        gguf_dequant_cpu_proc = 0.0
        gguf_trace_stats = {}
        if gguf_ops is not None:
            gguf_trace_stats = gguf_ops.consume_trace_stats()
            gguf_dequant = float(gguf_trace_stats.get("dequant_seconds", 0.0))
            gguf_dequant_cpu_proc = float(gguf_trace_stats.get("dequant_cpu_process_seconds", 0.0))
        apply_model_trace_stats = apply_model_trace.consume_apply_model_trace_stats()

        if denoise_result.checkpoint_records:
            checkpoint_records_path = str(output_dir / f"{self.scenario.name}_{self.route_label}_{run_label}_checkpoints.json")
            Path(checkpoint_records_path).write_text(
                json.dumps(denoise_result.checkpoint_records, indent=2),
                encoding="utf-8",
            )

        decoded, vae_gpu_load, vae_decode = self.glass_pipeline.decode_latent(denoise_result.samples)
        image_name = f"{self.scenario.name}_{self.route_label}_{run_label}.png"
        image_path = output_dir / image_name
        _ensure_parent(image_path)
        save_start = time.perf_counter()
        _png_save(decoded[0], image_path)
        image_save = time.perf_counter() - save_start

        end_state = self._capture_run_boundary_state("run_end", run_label)
        clip_cache_after = self._clip_cache_entry_count()
        gguf_mmap_released_after = self._gguf_mmap_released()
        total_wall = time.perf_counter() - total_start

        return RunMetrics(
            scenario=self.scenario.name,
            route_label=self.route_label,
            run_label=run_label,
            prompt_hash=_hash_prompt(self.scenario.prompt, self.scenario.negative_prompt),
            quant_model=Path(self.scenario.unet_path).name,
            resolution=f"{self.scenario.width}x{self.scenario.height}",
            steps=self.scenario.steps,
            cfg=self.scenario.cfg,
            sampler=self.scenario.sampler,
            scheduler=self.scenario.scheduler,
            seed=self.scenario.seed,
            batch_size=self.scenario.batch_size,
            process_start=self._process_started,
            cold_model_load_cpu=self._cold_model_load_cpu if run_label == "cold" else 0.0,
            clip_residency_attach=prep_metrics.clip_residency_attach,
            clip_residency_offload=prep_metrics.clip_residency_offload,
            clip_gpu_load=prep_metrics.clip_residency_attach,
            clip_encode=prep_metrics.clip_encode,
            adm_build=prep_metrics.adm_build,
            sampler_model_attach=denoise_result.sampler_model_attach,
            unet_gpu_load_or_patch=denoise_result.sampler_model_attach,
            cond_prepare_explicit=prep_metrics.cond_prepare,
            cond_prep=prep_metrics.cond_prepare,
            denoise_wall=denoise_result.denoise_wall,
            denoise_s_per_it=denoise_result.denoise_wall / max(1, self.scenario.steps),
            denoise_cpu_proc=denoise_result.denoise_cpu_proc,
            gguf_dequant=gguf_dequant,
            gguf_dequant_cpu_proc=gguf_dequant_cpu_proc,
            vae_attach=vae_gpu_load,
            vae_gpu_load=vae_gpu_load,
            vae_decode=vae_decode,
            cleanup_reset=0.0,
            image_save=image_save,
            total_wall=total_wall,
            image_path=str(image_path),
            warm_state_annotation=self._build_warm_state_annotation(
                run_label,
                start_state,
                end_state,
                clip_cache_before=clip_cache_before,
                clip_cache_after=clip_cache_after,
                cleanup_result=None,
                gguf_mmap_released_before=gguf_mmap_released_before,
                gguf_mmap_released_after=gguf_mmap_released_after,
            ),
            checkpoint_records_path=checkpoint_records_path,
            checkpoint_record_count=len(denoise_result.checkpoint_records),
            glass_ancestral_noise_policy=self.glass_pipeline.config.ancestral_noise_policy,
            gguf_trace_stats=gguf_trace_stats,
            apply_model_trace_stats=apply_model_trace_stats,
            notes=self._route_notes(),
        )

    def _clean_unet_budget(self) -> int:
        if self.explicit_unet_budget_mb is not None:
            return self.explicit_unet_budget_mb * 1024 * 1024
        return int(resources.maximum_vram_for_weights(self.device))

    def _ensure_clean_unet_loaded(self) -> float:
        assert self.unet is not None
        start = time.perf_counter()
        budget = self._clean_unet_budget()
        model_size = int(self.unet.model_size())
        lowvram_model_memory = 0 if budget >= model_size else budget
        self.unet.patch_model(device_to=self.device, lowvram_model_memory=lowvram_model_memory)
        return time.perf_counter() - start

    def _ensure_clean_vae_loaded(self) -> float:
        assert self.vae is not None
        start = time.perf_counter()
        self.vae.patcher.patch_model(device_to=self.device, lowvram_model_memory=0)
        self.vae.first_stage_model.to(device=self.device, dtype=torch.float32)
        return time.perf_counter() - start

    def _encode_prompts(self) -> tuple[Any, Any, float, float, float, float]:
        assert self.clip is not None

        if self.route_label == "backend_explicit":
            clip_gpu_load = self._measure_explicit_attach(
                self.clip.patcher,
                "clip_encode",
                target_phase=resources.MemoryPhase.PROMPT_ENCODE,
            )
            encode_start = time.perf_counter()
            with torch.inference_mode(), precision.autocast_context(self.device, enabled=True):
                encoded_pair = conditioning.encode_prompt_pair_sdxl(
                    self.clip,
                    self.scenario.prompt,
                    self.scenario.negative_prompt,
                    use_explicit_residency=True,
                )
            clip_encode = time.perf_counter() - encode_start
            clip_residency_offload = self._measure_explicit_detach(self.clip.patcher, "clip_encode")

            adm_start = time.perf_counter()
            adm_pair = conditioning.build_sdxl_adm_pair(
                encoded_pair,
                self.scenario.width,
                self.scenario.height,
                target_width=self.scenario.width,
                target_height=self.scenario.height,
                adm_scale_positive=self.scenario.quality.adm_scale_positive,
                adm_scale_negative=self.scenario.quality.adm_scale_negative,
            )
            adm_build = time.perf_counter() - adm_start

            positive = [[encoded_pair["positive"]["cond"], {"pooled_output": encoded_pair["positive"]["pooled"], "model_conds": {"y": adm_pair["positive"]}}]]
            negative = [[encoded_pair["negative"]["cond"], {"pooled_output": encoded_pair["negative"]["pooled"], "model_conds": {"y": adm_pair["negative"]}}]]
            return positive, negative, clip_gpu_load, clip_residency_offload, clip_encode, adm_build

        clip_gpu_load = 0.0
        clip_residency_offload = 0.0
        load_device = self.clip.patcher.load_device
        offload_device = self.clip.patcher.offload_device
        model = self.clip.cond_stage_model

        if self.clip.layer_idx is not None:
            model.clip_layer(self.clip.layer_idx)
        else:
            model.reset_clip_layer()

        if self.route_label == "headless_intermediate":
            clip_gpu_load = self._measure_intermediate_model_load([self.clip.patcher])
        else:
            load_start = time.perf_counter()
            if load_device != offload_device:
                self.clip.patcher.model.to(load_device)
            clip_gpu_load = time.perf_counter() - load_start

        encode_start = time.perf_counter()
        with torch.inference_mode(), precision.autocast_context(self.device, enabled=True):
            pos_tokens = self.clip.tokenize(self.scenario.prompt)
            pos_cond, pos_pooled = model.encode_token_weights(pos_tokens)

            neg_tokens = self.clip.tokenize(self.scenario.negative_prompt)
            neg_cond, neg_pooled = model.encode_token_weights(neg_tokens)
        clip_encode = time.perf_counter() - encode_start

        if self.route_label == "headless_clean" and load_device != offload_device:
            self.clip.patcher.model.to(offload_device)

        adm_start = time.perf_counter()
        adm_pos = conditioning.get_adm_embeddings_sdxl(
            pos_pooled,
            self.scenario.width,
            self.scenario.height,
            target_width=self.scenario.width,
            target_height=self.scenario.height,
            prompt_type="positive",
            adm_scale_positive=self.scenario.quality.adm_scale_positive,
            adm_scale_negative=self.scenario.quality.adm_scale_negative,
        )
        adm_neg = conditioning.get_adm_embeddings_sdxl(
            neg_pooled,
            self.scenario.width,
            self.scenario.height,
            target_width=self.scenario.width,
            target_height=self.scenario.height,
            prompt_type="negative",
            adm_scale_positive=self.scenario.quality.adm_scale_positive,
            adm_scale_negative=self.scenario.quality.adm_scale_negative,
        )
        adm_build = time.perf_counter() - adm_start

        positive = [[pos_cond, {"pooled_output": pos_pooled, "model_conds": {"y": adm_pos}}]]
        negative = [[neg_cond, {"pooled_output": neg_pooled, "model_conds": {"y": adm_neg}}]]
        return positive, negative, clip_gpu_load, clip_residency_offload, clip_encode, adm_build

    def _build_latent_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.unet is not None
        latent_h = self.scenario.height // 8
        latent_w = self.scenario.width // 8
        dtype = self.unet.model.get_dtype()

        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.scenario.seed)
        noise = torch.randn(
            (self.scenario.batch_size, 4, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=dtype,
        )
        latent = torch.zeros(
            (self.scenario.batch_size, 4, latent_h, latent_w),
            device=self.device,
            dtype=dtype,
        )
        return noise, latent

    def _denoise(self, positive: Any, negative: Any, noise: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, float, float, float, float, float]:
        assert self.unet is not None

        if self.route_label == "backend_explicit":
            unet_load = self._measure_explicit_attach(
                self.unet,
                "sample",
                target_phase=resources.MemoryPhase.DIFFUSION,
            )
            sampler_instance = sampling.KSampler(
                self.unet,
                self.scenario.steps,
                self.device,
                self.scenario.sampler,
                self.scenario.scheduler,
                self.scenario.denoise,
                model_options={"quality": self.scenario.quality.as_sampling_dict()},
            )
            guider = sampling.prepare_sampler_conds(
                self.unet,
                noise,
                positive,
                negative,
                self.scenario.cfg,
                sampler_name=self.scenario.sampler,
                latent_image=latent,
                denoise_mask=None,
                seed=self.scenario.seed,
                model_options={"quality": self.scenario.quality.as_sampling_dict()},
                quality=self.scenario.quality.as_sampling_dict(),
                inner_model=self.unet.model,
            )
            self._last_explicit_cond_prep = guider.cond_prep_duration

            denoise_start = time.perf_counter()
            denoise_cpu_start = time.process_time()
            with torch.inference_mode():
                samples = sampling.sample_prepared_sdxl(
                    guider,
                    noise,
                    sampler_instance.sigmas,
                    sampler=sampling.ksampler(self.scenario.sampler),
                    latent_image=latent,
                    denoise_mask=None,
                    callback=None,
                    disable_pbar=True,
                    seed=self.scenario.seed,
                    attach_model=False,
                )
            denoise_wall = time.perf_counter() - denoise_start
            denoise_cpu_proc = time.process_time() - denoise_cpu_start
            gguf_stats = getattr(guider, "last_gguf_trace_stats", {}) or {}
            gguf_dequant = float(gguf_stats.get("dequant_seconds", 0.0))
            gguf_dequant_cpu_proc = float(gguf_stats.get("dequant_cpu_process_seconds", 0.0))
            self._measure_explicit_detach(self.unet, "sample")
            return samples, unet_load, denoise_wall, denoise_cpu_proc, gguf_dequant, gguf_dequant_cpu_proc

        if self.route_label == "headless_intermediate":
            unet_load = self._measure_intermediate_model_load([self.unet])
            inner_model = self.unet.model
        else:
            unet_load = self._ensure_clean_unet_loaded()
            inner_model = self.unet.model

        sampler_instance = sampling.KSampler(
            self.unet,
            self.scenario.steps,
            self.device,
            self.scenario.sampler,
            self.scenario.scheduler,
            self.scenario.denoise,
            model_options={"quality": self.scenario.quality.as_sampling_dict()},
        )
        kernel = sampling.ksampler(self.scenario.sampler)
        guider = sampling.CFGGuider(self.unet)
        guider.set_conds(positive, negative)
        guider.set_cfg(self.scenario.cfg, cfg_pp="_cfg_pp" in self.scenario.sampler)
        guider.set_quality(self.scenario.quality.as_sampling_dict())
        guider.inner_model = inner_model
        guider.conds = {
            key: [entry.copy() for entry in guider.original_conds[key]]
            for key in guider.original_conds
        }

        if gguf_ops is not None:
            gguf_ops.reset_trace_stats()
        sampling.reset_sampler_trace_stats()
        sampling.reset_cond_batch_trace_stats()
        apply_model_trace.reset_apply_model_trace_stats()

        denoise_start = time.perf_counter()
        denoise_cpu_start = time.process_time()
        with torch.inference_mode():
            samples = kernel.sample(
                guider,
                sampler_instance.sigmas,
                {},
                None,
                noise,
                latent,
                None,
                True,
            )
        denoise_wall = time.perf_counter() - denoise_start
        denoise_cpu_proc = time.process_time() - denoise_cpu_start

        gguf_dequant = 0.0
        gguf_dequant_cpu_proc = 0.0
        if gguf_ops is not None:
            stats = gguf_ops.consume_trace_stats()
            gguf_dequant = float(stats.get("dequant_seconds", 0.0))
            gguf_dequant_cpu_proc = float(stats.get("dequant_cpu_process_seconds", 0.0))

        return samples, unet_load, denoise_wall, denoise_cpu_proc, gguf_dequant, gguf_dequant_cpu_proc

    def _decode(self, samples: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        assert self.vae is not None

        if self.route_label == "backend_explicit":
            vae_gpu_load = self._measure_explicit_attach(
                self.vae.patcher,
                "decode",
                target_phase=resources.MemoryPhase.DECODE,
            )
            decode_start = time.perf_counter()
            with torch.inference_mode():
                pixels = backend_decode.decode_preloaded_vae(self.vae, samples)
            vae_decode = time.perf_counter() - decode_start
            self._measure_explicit_detach(self.vae.patcher, "decode")
            return pixels.cpu(), vae_gpu_load, vae_decode

        if self.route_label == "headless_intermediate":
            vae_gpu_load = self._measure_intermediate_model_load([self.vae.patcher])
            self.vae.first_stage_model.to(device=self.device, dtype=torch.float32)
        else:
            vae_gpu_load = self._ensure_clean_vae_loaded()

        decode_start = time.perf_counter()
        with torch.inference_mode():
            scaled = self.vae.latent_format.process_out(samples).to(device=self.device, dtype=torch.float32)
            pixels = torch.clamp((self.vae.first_stage_model.decode(scaled).float() + 1.0) / 2.0, min=0.0, max=1.0)
        vae_decode = time.perf_counter() - decode_start
        return pixels.movedim(1, -1).cpu(), vae_gpu_load, vae_decode

    def _route_notes(self) -> str:
        notes = self.scenario.notes
        if self.route_label == "headless_intermediate":
            return (notes + " " if notes else "") + "Uses resources.load_models_gpu() for CLIP/UNet/VAE residency."
        if self.route_label == "backend_explicit":
            return (notes + " " if notes else "") + "Uses exposed backend lifecycle seams for CLIP attach/encode, sampler cond prep/denoise, and VAE decode."
        if self.route_label == "direct_sdxl_gguf":
            return (notes + " " if notes else "") + "Uses backend.gguf.direct_sdxl_runtime direct GGUF runtime path."
        if self.route_label == "glass_sdxl_gguf":
            return (notes + " " if notes else "") + "Uses backend.gguf.sdxl_glass_pipeline glass pipeline path with explicit denoise boundaries."
        return (notes + " " if notes else "") + "Uses direct patch_model()/decode path without resources.load_models_gpu()."

    def run_once(self, run_label: str, output_dir: Path) -> RunMetrics:
        self.load_models()

        if self.route_label == "direct_sdxl_gguf":
            return self._run_direct_runtime_once(run_label, output_dir)
        if self.route_label == "glass_sdxl_gguf":
            return self._run_glass_pipeline_once(run_label, output_dir)

        total_start = time.perf_counter()
        start_state = self._capture_run_boundary_state("run_start", run_label)
        clip_cache_before = self._clip_cache_entry_count()
        gguf_mmap_released_before = self._gguf_mmap_released()
        positive, negative, clip_gpu_load, clip_residency_offload, clip_encode, adm_build = self._encode_prompts()

        cond_prep_start = time.perf_counter()
        noise, latent = self._build_latent_inputs()
        cond_prep = time.perf_counter() - cond_prep_start

        (
            samples,
            unet_gpu_load_or_patch,
            denoise_wall,
            denoise_cpu_proc,
            gguf_dequant,
            gguf_dequant_cpu_proc,
        ) = self._denoise(positive, negative, noise, latent)

        cond_prepare_explicit = 0.0
        sampler_model_attach = 0.0
        vae_attach = 0.0
        if self.route_label == "backend_explicit":
            cond_prep = self._last_explicit_cond_prep
            cond_prepare_explicit = cond_prep
            sampler_model_attach = unet_gpu_load_or_patch

        decoded, vae_gpu_load, vae_decode = self._decode(samples)
        if self.route_label == "backend_explicit":
            vae_attach = vae_gpu_load

        image_name = f"{self.scenario.name}_{self.route_label}_{run_label}.png"
        image_path = output_dir / image_name
        _ensure_parent(image_path)
        save_start = time.perf_counter()
        _png_save(decoded[0], image_path)
        image_save = time.perf_counter() - save_start

        cleanup_reset = 0.0
        cleanup_result = None
        if self.route_label == "backend_explicit":
            cleanup_result = inference_lifecycle.reset_inference_run_state(
                "headless_backend_explicit_run",
                unload_models=False,
                force_cache=True,
                gc_collect=True,
                target_phase=resources.MemoryPhase.FINALIZE,
                notes={"route": self.route_label, "scenario": self.scenario.name, "run": run_label},
            )
            cleanup_reset = float(cleanup_result.get("duration_s", 0.0))
            end_state = cleanup_result.get("after", {})
        else:
            end_state = self._capture_run_boundary_state("run_end", run_label)

        clip_cache_after = self._clip_cache_entry_count()
        gguf_mmap_released_after = self._gguf_mmap_released()
        total_wall = time.perf_counter() - total_start

        return RunMetrics(
            scenario=self.scenario.name,
            route_label=self.route_label,
            run_label=run_label,
            prompt_hash=_hash_prompt(self.scenario.prompt, self.scenario.negative_prompt),
            quant_model=Path(self.scenario.unet_path).name,
            resolution=f"{self.scenario.width}x{self.scenario.height}",
            steps=self.scenario.steps,
            cfg=self.scenario.cfg,
            sampler=self.scenario.sampler,
            scheduler=self.scenario.scheduler,
            seed=self.scenario.seed,
            batch_size=self.scenario.batch_size,
            process_start=self._process_started,
            cold_model_load_cpu=self._cold_model_load_cpu if run_label == "cold" else 0.0,
            clip_residency_attach=clip_gpu_load if self.route_label == "backend_explicit" else 0.0,
            clip_residency_offload=clip_residency_offload,
            clip_gpu_load=clip_gpu_load,
            clip_encode=clip_encode,
            adm_build=adm_build,
            sampler_model_attach=sampler_model_attach,
            unet_gpu_load_or_patch=unet_gpu_load_or_patch,
            cond_prepare_explicit=cond_prepare_explicit,
            cond_prep=cond_prep,
            denoise_wall=denoise_wall,
            denoise_s_per_it=denoise_wall / max(1, self.scenario.steps),
            denoise_cpu_proc=denoise_cpu_proc,
            gguf_dequant=gguf_dequant,
            gguf_dequant_cpu_proc=gguf_dequant_cpu_proc,
            vae_attach=vae_attach,
            vae_gpu_load=vae_gpu_load,
            vae_decode=vae_decode,
            cleanup_reset=cleanup_reset,
            image_save=image_save,
            total_wall=total_wall,
            image_path=str(image_path),
            warm_state_annotation=self._build_warm_state_annotation(
                run_label,
                start_state,
                end_state,
                clip_cache_before=clip_cache_before,
                clip_cache_after=clip_cache_after,
                cleanup_result=cleanup_result,
                gguf_mmap_released_before=gguf_mmap_released_before,
                gguf_mmap_released_after=gguf_mmap_released_after,
            ),
            notes=self._route_notes(),
        )

    def _run_direct_runtime_once(self, run_label: str, output_dir: Path) -> RunMetrics:
        assert self.direct_runtime is not None

        total_start = time.perf_counter()
        start_state = self._capture_run_boundary_state("run_start", run_label)
        clip_cache_before = self._clip_cache_entry_count()
        gguf_mmap_released_before = self._gguf_mmap_released()
        checkpoint_records = []
        checkpoint_records_path = ""

        if self.checkpoint_enabled:
            prepared_inputs, prep_metrics = self.direct_runtime.prepare_inputs()

            def direct_checkpoint_callback(step: int, x0: torch.Tensor, x: torch.Tensor, total_steps: int, denoised: Optional[torch.Tensor] = None) -> None:
                checkpoint_records.append(_tensor_checkpoint_record(
                    f"step_{step:03d}.x_in",
                    x,
                    step_index=step,
                    summary={"step_index": step, "total_steps": total_steps},
                ))
                checkpoint_records.append(_tensor_checkpoint_record(
                    f"step_{step:03d}.post_cfg_denoised",
                    denoised if denoised is not None else x0,
                    step_index=step,
                    summary={"step_index": step, "total_steps": total_steps},
                ))

            apply_model_trace.reset_apply_model_trace_stats()
            denoise_result = self.direct_runtime.denoise_prepared_inputs(
                prepared_inputs,
                callback=direct_checkpoint_callback,
                disable_pbar=True,
            )
            apply_model_trace_stats = apply_model_trace.consume_apply_model_trace_stats()
            checkpoint_records.append(_tensor_checkpoint_record(
                "final.latent",
                denoise_result.samples,
                summary={"route": self.route_label},
            ))
            images, vae_attach, vae_decode = self.direct_runtime.decode_latent(denoise_result.samples)
            gguf_trace_stats = dict(denoise_result.gguf_trace_stats)
            benchmark = {
                "clip_residency_attach": prep_metrics.get("clip_residency_attach", 0.0),
                "clip_residency_offload": prep_metrics.get("clip_residency_offload", 0.0),
                "clip_encode": prep_metrics.get("clip_encode", 0.0),
                "adm_build": prep_metrics.get("adm_build", 0.0),
                "sampler_model_attach": denoise_result.sampler_model_attach,
                "cond_prepare_explicit": denoise_result.cond_prepare_duration,
                "denoise_wall": denoise_result.denoise_wall,
                "denoise_s_per_it": denoise_result.denoise_wall / max(1, self.scenario.steps),
                "denoise_cpu_proc": denoise_result.denoise_cpu_proc,
                "gguf_dequant": float(denoise_result.gguf_trace_stats.get("dequant_seconds", 0.0)),
                "gguf_dequant_cpu_proc": float(denoise_result.gguf_trace_stats.get("dequant_cpu_process_seconds", 0.0)),
                "vae_attach": vae_attach,
                "vae_decode": vae_decode,
            }
        else:
            apply_model_trace.reset_apply_model_trace_stats()
            direct_result = self.direct_runtime.run()
            apply_model_trace_stats = apply_model_trace.consume_apply_model_trace_stats()
            images = direct_result.images
            benchmark = direct_result.benchmark
            gguf_trace_stats = {}

        self.unet = self.direct_runtime.unet
        self.clip = self.direct_runtime.clip
        self.vae = self.direct_runtime.vae

        image_name = f"{self.scenario.name}_{self.route_label}_{run_label}.png"
        image_path = output_dir / image_name
        _ensure_parent(image_path)
        save_start = time.perf_counter()
        _png_save(images[0].cpu(), image_path)
        image_save = time.perf_counter() - save_start

        if checkpoint_records:
            checkpoint_records_path = str(output_dir / f"{self.scenario.name}_{self.route_label}_{run_label}_checkpoints.json")
            Path(checkpoint_records_path).write_text(
                json.dumps(checkpoint_records, indent=2),
                encoding="utf-8",
            )

        end_state = self._capture_run_boundary_state("run_end", run_label)
        clip_cache_after = self._clip_cache_entry_count()
        gguf_mmap_released_after = self._gguf_mmap_released()
        total_wall = time.perf_counter() - total_start

        clip_residency_attach = float(benchmark.get("clip_residency_attach", 0.0))
        clip_residency_offload = float(benchmark.get("clip_residency_offload", 0.0))
        clip_encode = float(benchmark.get("clip_encode", 0.0))
        adm_build = float(benchmark.get("adm_build", 0.0))
        sampler_model_attach = float(benchmark.get("sampler_model_attach", 0.0))
        cond_prepare_explicit = float(benchmark.get("cond_prepare_explicit", 0.0))
        vae_attach = float(benchmark.get("vae_attach", 0.0))
        vae_decode = float(benchmark.get("vae_decode", 0.0))

        return RunMetrics(
            scenario=self.scenario.name,
            route_label=self.route_label,
            run_label=run_label,
            prompt_hash=_hash_prompt(self.scenario.prompt, self.scenario.negative_prompt),
            quant_model=Path(self.scenario.unet_path).name,
            resolution=f"{self.scenario.width}x{self.scenario.height}",
            steps=self.scenario.steps,
            cfg=self.scenario.cfg,
            sampler=self.scenario.sampler,
            scheduler=self.scenario.scheduler,
            seed=self.scenario.seed,
            batch_size=self.scenario.batch_size,
            process_start=self._process_started,
            cold_model_load_cpu=self._cold_model_load_cpu if run_label == "cold" else 0.0,
            clip_residency_attach=clip_residency_attach,
            clip_residency_offload=clip_residency_offload,
            clip_gpu_load=clip_residency_attach,
            clip_encode=clip_encode,
            adm_build=adm_build,
            sampler_model_attach=sampler_model_attach,
            unet_gpu_load_or_patch=sampler_model_attach,
            cond_prepare_explicit=cond_prepare_explicit,
            cond_prep=cond_prepare_explicit,
            denoise_wall=float(benchmark.get("denoise_wall", 0.0)),
            denoise_s_per_it=float(benchmark.get("denoise_s_per_it", 0.0)),
            denoise_cpu_proc=float(benchmark.get("denoise_cpu_proc", 0.0)),
            gguf_dequant=float(benchmark.get("gguf_dequant", 0.0)),
            gguf_dequant_cpu_proc=float(benchmark.get("gguf_dequant_cpu_proc", 0.0)),
            vae_attach=vae_attach,
            vae_gpu_load=vae_attach,
            vae_decode=vae_decode,
            cleanup_reset=0.0,
            image_save=image_save,
            total_wall=total_wall,
            image_path=str(image_path),
            warm_state_annotation=self._build_warm_state_annotation(
                run_label,
                start_state,
                end_state,
                clip_cache_before=clip_cache_before,
                clip_cache_after=clip_cache_after,
                cleanup_result=None,
                gguf_mmap_released_before=gguf_mmap_released_before,
                gguf_mmap_released_after=gguf_mmap_released_after,
            ),
            checkpoint_records_path=checkpoint_records_path,
            checkpoint_record_count=len(checkpoint_records),
            gguf_trace_stats=gguf_trace_stats,
            apply_model_trace_stats=apply_model_trace_stats,
            notes=self._route_notes(),
        )

    def close(self) -> None:
        if self.direct_runtime is not None:
            self.direct_runtime.close()
        elif self.glass_pipeline is not None:
            self.glass_pipeline.detach_unet_direct()
            self.glass_pipeline.detach_clip_direct()
            if self.vae is not None:
                self.vae.patcher.detach()
        else:
            if self.unet is not None:
                self.unet.detach()
            if self.vae is not None:
                self.vae.patcher.detach()
            if self.clip is not None:
                self.clip.patcher.detach()
        resources.soft_empty_cache(force=True)
        gc.collect()


def scenario_library() -> Dict[str, ScenarioConfig]:
    imagine_root = Path(r"D:\AI\Imagine\models")
    return {
        "historical_q4_bare": ScenarioConfig(
            name="historical_q4_bare",
            unet_path=str(imagine_root / "unet" / "IL_dutch_v30_Q4_K_M.gguf"),
            clip_l_path=str(imagine_root / "clip" / "IL_dutch_v30_clips.safetensors"),
            clip_g_path=str(imagine_root / "clip" / "IL_dutch_v30_clips.safetensors"),
            vae_path=str(imagine_root / "vae" / "sdxl_vae.safetensors"),
            prompt=DEFAULT_POSITIVE_PROMPT,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            width=1024,
            height=1024,
            steps=10,
            cfg=7.0,
            sampler="euler_ancestral",
            scheduler="karras",
            seed=12345,
            quality=QualityConfig(
                sharpness=2.0,
                adaptive_cfg=7.0,
                adm_scale_positive=1.5,
                adm_scale_negative=0.8,
                adm_scaler_end=0.3,
            ),
            notes="Historical P3-M10 style bare GGUF case.",
        ),
        "mission_q4_acceptance": ScenarioConfig(
            name="mission_q4_acceptance",
            unet_path=str(imagine_root / "unet" / "IL_beretMixReal_v100_Q4_K_M.gguf"),
            clip_l_path=str(imagine_root / "clip" / "IL_beretMixReal_v100_clips.safetensors"),
            clip_g_path=str(imagine_root / "clip" / "IL_beretMixReal_v100_clips.safetensors"),
            vae_path=str(imagine_root / "vae" / "sdxl_vae.safetensors"),
            prompt=DEFAULT_POSITIVE_PROMPT,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            width=1024,
            height=1024,
            steps=10,
            cfg=7.0,
            sampler="euler_ancestral",
            scheduler="karras",
            seed=12345,
            notes="Mission acceptance Q4_K_M candidate using current local beretMixReal assets.",
        ),
        "mission_q5_acceptance": ScenarioConfig(
            name="mission_q5_acceptance",
            unet_path=str(imagine_root / "unet" / "IL_beretMixReal_v100_Q5_K_M.gguf"),
            clip_l_path=str(imagine_root / "clip" / "IL_beretMixReal_v100_clips.safetensors"),
            clip_g_path=str(imagine_root / "clip" / "IL_beretMixReal_v100_clips.safetensors"),
            vae_path=str(imagine_root / "vae" / "sdxl_vae.safetensors"),
            prompt=DEFAULT_POSITIVE_PROMPT,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            width=1024,
            height=1024,
            steps=10,
            cfg=7.0,
            sampler="euler_ancestral",
            scheduler="karras",
            seed=12345,
            notes="Mission acceptance Q5_K_M candidate using current local beretMixReal assets.",
        ),
        "comfy_q5_parity": ScenarioConfig(
            name="comfy_q5_parity",
            unet_path=r"G:\ComfyUI\models\diffusion_models\hsUltrahdCG_IllEpic_Q5_K_M.gguf",
            clip_l_path=r"D:\AI\Imagine\models\clip\IL_beretMixReal_v100_clips.safetensors",
            clip_g_path=r"D:\AI\Imagine\models\clip\IL_beretMixReal_v100_clips.safetensors",
            vae_path=r"D:\AI\Imagine\models\vae\sdxl_vae.safetensors",
            prompt=COMFY_POSITIVE_PROMPT,
            negative_prompt=COMFY_NEGATIVE_PROMPT,
            width=832,
            height=1216,
            steps=20,
            cfg=4.0,
            sampler="euler_ancestral",
            scheduler="karras",
            seed=956720277608864,
            notes="Nearest available local Q5-class parity placeholder; exact PowerPuff assets were not found locally.",
        ),
    }


def write_environment_report(environment: EnvironmentSnapshot, output_dir: Path) -> Path:
    destination = output_dir / "environment.json"
    _ensure_parent(destination)
    destination.write_text(json.dumps(asdict(environment), indent=2), encoding="utf-8")
    return destination


def append_metrics_jsonl(metrics: RunMetrics, output_dir: Path) -> Path:
    destination = output_dir / "benchmark_results.jsonl"
    _ensure_parent(destination)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(metrics)) + "\n")
    return destination
