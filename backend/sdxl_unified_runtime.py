"""Nex-owned production home for the unified SDXL runtime spine.

W07c2 uses this module to build the CPU-first artifact boundary. W07c3 should
consume those prepared artifacts for stream-like denoise and decode, without
reassigning ownership of the preparation step.
"""

from __future__ import annotations

import gc
import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from backend import conditioning, lora as backend_lora, loader
from backend.cpu_compiler import CpuArtifactCompiler, SafeOpenHeaderOnly
from backend.lora_artifacts import compute_file_hash
from backend.sdxl_runtime_contract import (
    BaseModelAvailability,
    CompiledUnetArtifact,
    GpuAttachedExecutionState,
    InjectedFeatureArtifact,
    PromptConditioningArtifact,
    SDXL_RUNTIME_SURFACE_CONTRACTS,
    UnifiedSDXLRuntimeProtocol,
    UnifiedSDXLRuntimeSeams,
)


@dataclass(frozen=True)
class UnifiedSDXLRuntimeConfig:
    """Configuration shared by unified SDXL runtime execution modes."""

    model_variant: str
    execution_class: str
    checkpoint_path: str
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
    batch_size: int = 1
    lora_specs: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    pin_base_unet_without_lora: bool = False


@dataclass
class UnifiedSDXLPreparedInputs:
    """Prepared runtime artifacts consumed by the denoise entrypoint."""

    base_model: BaseModelAvailability | None = None
    compiled_unet: CompiledUnetArtifact | None = None
    conditioning: PromptConditioningArtifact | None = None
    injected_features: dict[str, InjectedFeatureArtifact] = field(default_factory=dict)
    gpu_attached_execution_state: GpuAttachedExecutionState | None = None
    payload: Any = None
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class UnifiedSDXLDenoiseResult:
    """Denoise output plus the explicit GPU-attached execution state."""

    samples: torch.Tensor
    execution_state: GpuAttachedExecutionState | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class UnifiedSDXLRuntime(UnifiedSDXLRuntimeProtocol):
    """Unified Nex-owned SDXL runtime spine.

    This class is intentionally narrow in W07c2. It exists to give W07c3 a
    stable production-owned target instead of building directly against the
    benchmark harness.
    """

    route_label = "sdxl_unified_runtime"

    seams = UnifiedSDXLRuntimeSeams(
        task_start_owner="modules.async_worker",
        prompt_conditioning_owner="backend.conditioning",
        compiled_unet_owner="backend.sdxl_unified_runtime",
        denoise_owner="backend.sdxl_unified_runtime",
        decode_owner="backend.sdxl_unified_runtime",
    )

    def __init__(self, config: UnifiedSDXLRuntimeConfig) -> None:
        self.config = config
        self.unet: Any = None
        self.clip: Any = None
        self.vae: Any = None
        self.base_model: BaseModelAvailability | None = None
        self.compiled_unet: CompiledUnetArtifact | None = None
        self.conditioning: PromptConditioningArtifact | None = None
        self.injected_features: dict[str, InjectedFeatureArtifact] = {}
        self.execution_state: GpuAttachedExecutionState | None = None
        self.prepared_inputs: UnifiedSDXLPreparedInputs | None = None
        self._loaded = False
        self._cold_model_load_cpu = 0.0
        self._prepare_metrics: dict[str, float] = {}
        self._resolved_lora_specs: tuple[tuple[str, float], ...] = self._normalize_lora_specs(
            self.config.lora_specs
        )
        self._checkpoint_fingerprint: str | None = None
        self._clip_identity: str = self._build_clip_identity()

    def load_components(self) -> float:
        if self._loaded:
            return 0.0

        checkpoint_path = self._require_checkpoint_path()
        start = time.perf_counter()
        cpu_device = torch.device("cpu")
        self.unet, self.clip, self.vae = loader.load_sdxl_checkpoint(
            checkpoint_path,
            load_device=cpu_device,
            offload_device=cpu_device,
            unet_dtype=torch.float16,
            clip_load_device=cpu_device,
            clip_offload_device=cpu_device,
            vae_offload_device=cpu_device,
        )

        if self.unet is not None:
            self.unet.runtime_release_to_meta = False
        if self.clip is not None and hasattr(self.clip, "clip_layer"):
            self.clip.clip_layer(self.config.clip_layer)

        self._checkpoint_fingerprint = self._fingerprint_source_path(checkpoint_path)
        self.base_model = BaseModelAvailability(
            family="sdxl",
            variant=self.config.model_variant,
            source_path=checkpoint_path,
            fingerprint=self._checkpoint_fingerprint,
            loaded=True,
            reusable=True,
        )
        self._clip_identity = self._build_clip_identity()
        self._cold_model_load_cpu = time.perf_counter() - start
        self._loaded = True
        return self._cold_model_load_cpu

    def prepare_inputs(self) -> tuple[UnifiedSDXLPreparedInputs, dict[str, float]]:
        if self.prepared_inputs is not None:
            return self.prepared_inputs, dict(self._prepare_metrics)

        self.load_components()
        if self.base_model is None or self.clip is None or self.unet is None:
            raise RuntimeError("Unified SDXL runtime failed to load the base CPU components.")

        lora_metrics = self._materialize_lora_stack()

        encode_start = time.perf_counter()
        encoded_prompt_pair = conditioning.encode_prompt_pair_sdxl(
            self.clip,
            self.config.prompt,
            self.config.negative_prompt,
            use_explicit_residency=True,
        )
        conditioning_encode_wall = time.perf_counter() - encode_start

        adm_start = time.perf_counter()
        adm_pair = conditioning.build_sdxl_adm_pair(
            encoded_prompt_pair,
            self.config.width,
            self.config.height,
            target_width=self.config.width,
            target_height=self.config.height,
        )
        adm_build_wall = time.perf_counter() - adm_start

        prompt_stage = conditioning.build_sdxl_text_conditioning_fingerprint(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            model_identity=self.base_model.fingerprint or self.base_model.source_path or self.config.model_variant,
            text_encoder_identity=self._clip_identity,
            clip_patch_uuid=self._lora_signature(),
            clip_layer_idx=self.config.clip_layer,
            lora_artifacts_state=self._resolved_lora_specs,
            route_family_reconciliation_signature=(self.route_label, self.seams.compiled_unet_owner),
            residency_class="cpu",
            route_family=self.route_label,
            execution_family=self.config.execution_class,
            clip_residency_mode="cpu",
        )
        prompt_fingerprint = prompt_stage.digest()
        conditioning_fingerprint = self._hash_payload(
            {
                "positive": encoded_prompt_pair["positive"],
                "negative": encoded_prompt_pair["negative"],
                "adm_pair": adm_pair,
            }
        )
        pooled_fingerprint = self._hash_payload(
            {
                "positive": encoded_prompt_pair["positive"]["pooled"],
                "negative": encoded_prompt_pair["negative"]["pooled"],
            }
        )

        self.conditioning = PromptConditioningArtifact(
            family="sdxl",
            variant=self.config.model_variant,
            prompt_fingerprint=prompt_fingerprint,
            clip_identity=self._clip_identity,
            clip_layer_idx=self.config.clip_layer,
            conditioning_fingerprint=conditioning_fingerprint,
            pooled_fingerprint=pooled_fingerprint,
            reusable=True,
        )

        self.injected_features = {
            "feature_boundary_placeholder": InjectedFeatureArtifact(
                family="sdxl",
                variant=self.config.model_variant,
                block_id="diffusion_boundary",
                timestep_key="unbound",
                context_key="not-prepared-in-w07c2",
                feature_fingerprint=None,
                reusable=True,
            )
        }

        unet_compile_metrics = lora_metrics["unet_compile_metrics"]
        compiled_unet_wall = float(lora_metrics["unet_compile_wall"])

        self.compiled_unet = CompiledUnetArtifact(
            family="sdxl",
            variant=self.config.model_variant,
            execution_class=self._execution_class_label(),
            source_path=self.config.checkpoint_path,
            source_fingerprint=self._checkpoint_fingerprint,
            artifact_fingerprint=self._build_compiled_unet_fingerprint(
                unet_compile_metrics=unet_compile_metrics,
            ),
            pinned_cpu_mb=self._measure_pinned_bytes(self.unet.model) / (1024 * 1024),
            gpu_mb=0.0,
            reusable=True,
        )

        self.execution_state = GpuAttachedExecutionState(
            execution_class=self._execution_class_label(),
            device="cuda",
            active_phase="prepare_inputs",
            attached_component_ids=(),
            stream_budget_mb=0.0,
            headroom_mb=0.0,
        )

        self.prepared_inputs = UnifiedSDXLPreparedInputs(
            base_model=self.base_model,
            compiled_unet=self.compiled_unet,
            conditioning=self.conditioning,
            injected_features=dict(self.injected_features),
            gpu_attached_execution_state=self.execution_state,
            payload={
                "encoded_prompt_pair": encoded_prompt_pair,
                "adm_pair": adm_pair,
                "prompt_fingerprint": prompt_fingerprint,
                "conditioning_fingerprint": conditioning_fingerprint,
                "pooled_fingerprint": pooled_fingerprint,
                "lora_specs": self._resolved_lora_specs,
                "base_model_fingerprint": self.base_model.fingerprint,
                "compiled_unet_fingerprint": self.compiled_unet.artifact_fingerprint,
            },
            metrics={
                "base_model_load_cpu": float(self._cold_model_load_cpu),
                "conditioning_encode_cpu": float(conditioning_encode_wall),
                "conditioning_adm_cpu": float(adm_build_wall),
                "compiled_unet_cpu": float(compiled_unet_wall),
                "lora_spec_count": float(lora_metrics["spec_count"]),
                "clip_patch_count": float(lora_metrics["clip_patch_count"]),
                "unet_patch_count": float(lora_metrics["unet_patch_count"]),
                "clip_host_pinned_bytes": float(lora_metrics["clip_host_pinned_bytes"]),
                "unet_host_pinned_bytes": float(lora_metrics["unet_host_pinned_bytes"]),
                "clip_compile_cpu": float(lora_metrics["clip_compile_wall"]),
                "conditioning_artifact_count": 1.0,
                "injected_feature_count": float(len(self.injected_features)),
            },
        )
        self._prepare_metrics = dict(self.prepared_inputs.metrics)
        return self.prepared_inputs, dict(self._prepare_metrics)

    def denoise_prepared_inputs(
        self,
        prepared_inputs: UnifiedSDXLPreparedInputs,
        *,
        callback: Any = None,
        disable_pbar: bool = True,
    ) -> UnifiedSDXLDenoiseResult:
        raise NotImplementedError(
            "W07c3 must implement stream-like denoise in backend.sdxl_unified_runtime."
        )

    def decode_latent(self, latent: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        raise NotImplementedError(
            "W07c3 must implement the unified decode transition in backend.sdxl_unified_runtime."
        )

    def close(self) -> None:
        self.execution_state = None
        self.prepared_inputs = None
        self.base_model = None
        self.compiled_unet = None
        self.conditioning = None
        self.injected_features = {}
        self.unet = None
        self.clip = None
        self.vae = None
        self._loaded = False
        self._prepare_metrics = {}
        gc.collect()

    def patched_weights_for_block(self, block_id: str) -> Any:
        _ = SDXL_RUNTIME_SURFACE_CONTRACTS["patched_weights_for_block"]
        raise NotImplementedError(
            "W07c2/W07c3 must implement compiled block retrieval in backend.sdxl_unified_runtime."
        )

    def injected_features_for_block(self, block_id: str, timestep: Any, context: Any) -> Any:
        _ = SDXL_RUNTIME_SURFACE_CONTRACTS["injected_features_for_block"]
        raise NotImplementedError(
            "W07c2/W07c3 must implement feature injection retrieval in backend.sdxl_unified_runtime."
        )

    def _require_checkpoint_path(self) -> str:
        checkpoint_path = str(self.config.checkpoint_path or "").strip()
        if not checkpoint_path:
            raise ValueError("Unified SDXL runtime requires config.checkpoint_path for CPU-first preparation.")
        return checkpoint_path

    def _execution_class_label(self) -> str:
        return str(self.config.execution_class or self.route_label)

    def _build_clip_identity(self) -> str:
        checkpoint = self.config.checkpoint_path
        if not checkpoint and self.base_model is not None and self.base_model.source_path:
            checkpoint = self.base_model.source_path
        return f"{self.config.model_variant}:{checkpoint}:{self.config.clip_layer}"

    def _normalize_lora_specs(self, lora_specs: Any) -> tuple[tuple[str, float], ...]:
        normalized: list[tuple[str, float]] = []
        for spec in lora_specs or ():
            if not spec:
                continue
            path, strength = spec
            normalized.append((str(path), float(strength)))
        return tuple(normalized)

    def _fingerprint_source_path(self, source_path: str | None) -> str | None:
        if not source_path:
            return None
        if os.path.isfile(source_path):
            try:
                return compute_file_hash(source_path)
            except Exception:
                pass
        return hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()

    def _lora_signature(self) -> tuple[str, ...]:
        return tuple(f"{path}:{strength:g}" for path, strength in self._resolved_lora_specs)

    def _hash_payload(self, payload: Any) -> str:
        digest = hashlib.sha256()
        digest.update(repr(self._freeze_value(payload)).encode("utf-8"))
        return digest.hexdigest()

    def _freeze_value(self, value: Any) -> Any:
        if hasattr(value, "__dataclass_fields__"):
            from dataclasses import asdict

            return self._freeze_value(asdict(value))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            return (
                "tensor",
                tuple(int(dim) for dim in tensor.shape),
                str(tensor.dtype),
                hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
            )
        if isinstance(value, dict):
            return tuple((str(key), self._freeze_value(item)) for key, item in sorted(value.items(), key=lambda item: str(item[0])))
        if isinstance(value, (list, tuple)):
            return tuple(self._freeze_value(item) for item in value)
        if isinstance(value, set):
            return tuple(sorted(self._freeze_value(item) for item in value))
        return value

    def _measure_pinned_bytes(self, module: Any) -> int:
        if module is None:
            return 0
        total = 0
        for tensor in list(module.parameters()) + list(module.buffers()):
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu" and tensor.is_pinned():
                total += tensor.numel() * tensor.element_size()
        return total

    def _materialize_lora_stack(self) -> dict[str, float]:
        clip_patch_count = 0
        unet_patch_count = 0
        clip_host_pinned_bytes = 0
        unet_host_pinned_bytes = 0

        if not self._resolved_lora_specs:
            unet_compile_start = time.perf_counter()
            unet_compile = self._compile_patcher(
                self.unet,
                pin_model_host=self.config.pin_base_unet_without_lora,
            )
            unet_compile_wall = time.perf_counter() - unet_compile_start
            unet_patch_count = int(unet_compile.get("patch_count", 0))
            unet_host_pinned_bytes = int(unet_compile.get("host_pinned_bytes", 0))
            return {
                "spec_count": 0.0,
                "clip_patch_count": 0.0,
                "unet_patch_count": float(unet_patch_count),
                "clip_host_pinned_bytes": float(clip_host_pinned_bytes),
                "unet_host_pinned_bytes": float(unet_host_pinned_bytes),
                "clip_compile_wall": 0.0,
                "unet_compile_wall": float(unet_compile_wall),
                "clip_compile_metrics": {"status": "noop", "patch_count": 0, "host_pinned_bytes": 0},
                "unet_compile_metrics": unet_compile,
            }

        clip_patch_count = self._apply_lora_specs_to_patcher(
            self.clip.patcher,
            self.clip.patcher.model,
            target_family="clip",
        )
        clip_compile_wall = 0.0
        clip_compile_metrics: dict[str, Any] = {"status": "noop", "patch_count": 0, "host_pinned_bytes": 0}
        if clip_patch_count > 0:
            clip_compile_start = time.perf_counter()
            clip_compile = self._compile_patcher(self.clip.patcher)
            clip_compile_wall = time.perf_counter() - clip_compile_start
            clip_compile_metrics = clip_compile
            clip_host_pinned_bytes = int(clip_compile.get("host_pinned_bytes", 0))

        self._apply_lora_specs_to_patcher(
            self.unet,
            self.unet.model,
            target_family="unet",
        )
        unet_compile_start = time.perf_counter()
        unet_compile = self._compile_patcher(self.unet)
        unet_compile_wall = time.perf_counter() - unet_compile_start
        unet_patch_count = int(unet_compile.get("patch_count", 0))
        unet_host_pinned_bytes = int(unet_compile.get("host_pinned_bytes", 0))

        return {
            "spec_count": float(len(self._resolved_lora_specs)),
            "clip_patch_count": float(clip_patch_count),
            "unet_patch_count": float(unet_patch_count),
            "clip_host_pinned_bytes": float(clip_host_pinned_bytes),
            "unet_host_pinned_bytes": float(unet_host_pinned_bytes),
            "clip_compile_wall": float(clip_compile_wall),
            "unet_compile_wall": float(unet_compile_wall),
            "clip_compile_metrics": clip_compile_metrics,
            "unet_compile_metrics": unet_compile,
        }

    def _apply_lora_specs_to_patcher(self, patcher: Any, model: Any, *, target_family: str) -> int:
        if patcher is None or model is None:
            return 0

        key_map = (
            backend_lora.model_lora_keys_clip(model)
            if target_family == "clip"
            else backend_lora.model_lora_keys_unet(model)
        )
        patch_count = 0
        for lora_path, strength in self._resolved_lora_specs:
            header = SafeOpenHeaderOnly(lora_path)
            patch_dict = backend_lora.load_lora(header, key_map, log_missing=False)
            if not patch_dict:
                continue
            patcher.add_patches(patch_dict, strength)
            patch_count += len(patch_dict)
        return patch_count

    def _compile_patcher(self, patcher: Any, *, pin_model_host: bool = True) -> dict[str, Any]:
        if patcher is None:
            return {"status": "noop", "patch_count": 0, "host_pinned_bytes": 0}

        compile_result = CpuArtifactCompiler.compile_patcher(
            patcher,
            pin_unet_host=pin_model_host,
        )
        host_pinned_bytes = self._measure_pinned_bytes(getattr(patcher, "model", None))
        return {
            **compile_result,
            "host_pinned_bytes": float(max(host_pinned_bytes, int(compile_result.get("host_pinned_bytes", 0)))),
            "patch_count": float(
                compile_result.get("materialized_patch_keys", compile_result.get("patch_count", 0))
            ),
        }

    def _build_compiled_unet_fingerprint(
        self,
        *,
        unet_compile_metrics: dict[str, Any],
    ) -> str:
        digest = hashlib.sha256()
        digest.update((self.base_model.fingerprint or self.config.checkpoint_path or "").encode("utf-8"))
        digest.update(self._execution_class_label().encode("utf-8"))
        digest.update(repr(self._lora_signature()).encode("utf-8"))
        digest.update(repr(self.config.clip_layer).encode("utf-8"))
        digest.update(repr(self.config.batch_size).encode("utf-8"))
        digest.update(repr(self.config.steps).encode("utf-8"))
        digest.update(repr(unet_compile_metrics.get("patch_count", 0)).encode("utf-8"))
        digest.update(repr(unet_compile_metrics.get("host_pinned_bytes", 0)).encode("utf-8"))
        return digest.hexdigest()
