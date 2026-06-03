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
from backend import conditioning, decode, k_diffusion, lora as backend_lora, loader, precision, resources, sampling
from backend.cpu_compiler import CpuArtifactCompiler, SafeOpenHeaderOnly
from backend.lora_artifacts import compute_file_hash
from backend.sdxl_runtime_contract import (
    BaseModelAvailability,
    CompiledUnetArtifact,
    GpuAttachedExecutionState,
    InjectedFeatureArtifact,
    PromptConditioningArtifact,
    SDXL_RUNTIME_SURFACE_CONTRACTS,
    StructuralConditioningArtifact,
    SpatialConditioningArtifact,
    UnifiedSDXLRuntimeProtocol,
    UnifiedSDXLRuntimeSeams,
)
from backend.sdxl_unified_runtime_artifacts import UnifiedSDXLRuntimeArtifactMixin
from backend.sdxl_unified_runtime_execution import UnifiedSDXLRuntimeExecutionMixin


@dataclass(frozen=True)
class UnifiedSDXLRuntimeConfig:
    """Configuration shared by unified SDXL runtime execution modes."""

    model_variant: str
    execution_class: Any
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
    vae_path: str | None = None
    positive_texts: tuple[str, ...] = field(default_factory=tuple)
    negative_texts: tuple[str, ...] = field(default_factory=tuple)
    positive_top_k: int = 1
    negative_top_k: int = 1
    clip_layer: int = -2
    batch_size: int = 1
    lora_specs: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    pin_base_unet_without_lora: bool = False
    streamlike_budget_mb: float = 256.0
    quality: dict[str, float] = field(default_factory=dict)
    source_pixels: Any | None = None
    source_mask: Any | None = None
    spatial_mode: str | None = None
    resolved_spatial_context: Any | None = None
    outpaint_direction: str | None = None
    outpaint_expansion_size: int = 384
    outpaint_pixelate: bool = True
    structural_tasks: dict[str, tuple[tuple[Any, ...], ...]] = field(default_factory=dict)
    controlnet_paths: dict[str, str] = field(default_factory=dict)
    controlnet_quality: dict[str, float] = field(default_factory=dict)
    contextual_tasks: dict[str, tuple[tuple[Any, ...], ...]] = field(default_factory=dict)
    contextual_assets: dict[str, Any] = field(default_factory=dict)
    initial_latent: Any | None = None
    disable_initial_latent: bool = False
    denoise_strength: float | None = None
    runtime_policy: Any | None = None
    original_scheduler_name: str | None = None



@dataclass
class UnifiedSDXLPreparedInputs:
    """Prepared runtime artifacts consumed by the denoise entrypoint."""

    base_model: BaseModelAvailability | None = None
    compiled_unet: CompiledUnetArtifact | None = None
    conditioning: PromptConditioningArtifact | None = None
    structural_conditioning: StructuralConditioningArtifact | None = None
    spatial_conditioning: SpatialConditioningArtifact | None = None
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


class UnifiedSDXLRuntime(
    UnifiedSDXLRuntimeArtifactMixin,
    UnifiedSDXLRuntimeExecutionMixin,
    UnifiedSDXLRuntimeProtocol,
):
    """Unified Nex-owned SDXL runtime spine."""

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
        self.policy: Any = None
        self.base_model: BaseModelAvailability | None = None
        self.compiled_unet: CompiledUnetArtifact | None = None
        self.conditioning: PromptConditioningArtifact | None = None
        self.structural_conditioning: StructuralConditioningArtifact | None = None
        self.spatial_conditioning: SpatialConditioningArtifact | None = None
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
        self._attached_payload: dict[str, Any] | None = None
        self._loaded_controlnets: dict[str, Any] = {}

    def load_components(self) -> float:
        if self._loaded:
            return 0.0

        checkpoint_path = self._require_checkpoint_path()
        vae_path = self._require_optional_vae_path()
        start = time.perf_counter()
        cpu_device = torch.device("cpu")
        from backend.staging_manager import ExecutionClass

        from backend.sdxl_runtime_policy import resolve_sdxl_execution_policy
        self.policy = self.config.runtime_policy or resolve_sdxl_execution_policy(
            architecture="sdxl",
            base_model_name=checkpoint_path,
        )

        exec_class = self._resolved_execution_class()

        is_resident = exec_class in {
            ExecutionClass.SDXL_RESIDENT_T2,
            ExecutionClass.SDXL_GPU_GREEDY_T3PLUS,
        }

        if is_resident:
            cuda_device = resources.get_torch_device()
            # Resident SDXL keeps the checkpoint-weight UNet and VAE authoritative on GPU
            # until process-aware teardown releases them for a real model/process transition.
            self.unet, self.clip, self.vae = loader.load_sdxl_checkpoint(
                checkpoint_path,
                load_device=cuda_device,
                offload_device=cuda_device,
                unet_dtype=torch.float16,
                clip_load_device=cpu_device,
                clip_offload_device=cpu_device,
                vae_load_device=cuda_device,
                vae_offload_device=cuda_device,
                vae_source=vae_path,
            )
        else:
            self.unet, self.clip, self.vae = loader.load_sdxl_checkpoint(
                checkpoint_path,
                load_device=cpu_device,
                offload_device=cpu_device,
                unet_dtype=torch.float16,
                clip_load_device=cpu_device,
                clip_offload_device=cpu_device,
                vae_load_device=cpu_device,
                vae_offload_device=cpu_device,
                vae_source=vae_path,
            )

        if self.unet is not None:
            self.unet.runtime_release_to_meta = False
            # Apply scheduler-specific patch to the UNet
            orig_scheduler = self.config.original_scheduler_name or self.config.scheduler
            if orig_scheduler in ['lcm', 'tcd']:
                from modules import core as modules_core
                self.unet = modules_core.opModelSamplingDiscrete.patch(self.unet, orig_scheduler, False)[0]
            elif orig_scheduler == 'edm_playground_v2.5':
                from modules import core as modules_core
                self.unet = modules_core.opModelSamplingContinuousEDM.patch(self.unet, orig_scheduler, 120.0, 0.002)[0]
        if self.clip is not None:
            self.clip.runtime_policy = self.policy
            if hasattr(self.clip, "clip_layer"):
                self.clip.clip_layer(self.config.clip_layer)
        if self.vae is not None:
            self.vae.runtime_policy = self.policy

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

    def _is_vae_resident(self) -> bool:
        from backend.sdxl_runtime_policy import VAE_POSTURE_GPU_RESIDENT
        from backend.staging_manager import ExecutionClass
        policy = getattr(self, "policy", None)
        if policy is not None:
            return policy.vae_encode_mode == VAE_POSTURE_GPU_RESIDENT

        exec_class = self._resolved_execution_class()
        return exec_class in {
            ExecutionClass.SDXL_RESIDENT_T2,
            ExecutionClass.SDXL_GPU_GREEDY_T3PLUS,
        }

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
            positive_texts=self.config.positive_texts or (self.config.prompt,),
            negative_texts=self.config.negative_texts or (self.config.negative_prompt,),
            positive_top_k=self.config.positive_top_k,
            negative_top_k=self.config.negative_top_k,
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
            adm_scale_positive=float((self.config.quality or {}).get("adm_scaler_positive", 1.5)),
            adm_scale_negative=float((self.config.quality or {}).get("adm_scaler_negative", 0.8)),
        )
        adm_build_wall = time.perf_counter() - adm_start

        prompt_stage = conditioning.build_sdxl_text_conditioning_fingerprint(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            positive_texts=self.config.positive_texts or (self.config.prompt,),
            negative_texts=self.config.negative_texts or (self.config.negative_prompt,),
            positive_top_k=self.config.positive_top_k,
            negative_top_k=self.config.negative_top_k,
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

        injected_features, injected_payload, injected_metrics = self._prepare_injected_feature_artifacts()
        self.injected_features = injected_features

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

        structural_conditioning, structural_payload, structural_metrics = self._prepare_structural_conditioning_artifacts()
        self.structural_conditioning = structural_conditioning
        spatial_conditioning, spatial_payload, spatial_metrics = self._prepare_spatial_conditioning_artifacts()
        self.spatial_conditioning = spatial_conditioning

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
            structural_conditioning=self.structural_conditioning,
            spatial_conditioning=self.spatial_conditioning,
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
                "initial_latent": self.config.initial_latent,
                "denoise_strength": self.config.denoise_strength,
                **structural_payload,

                **spatial_payload,
                **injected_payload,
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
                "structural_artifact_count": 1.0 if structural_conditioning is not None else 0.0,
                "spatial_artifact_count": 1.0 if spatial_conditioning is not None else 0.0,
                "injected_feature_count": float(len(self.injected_features)),
                **structural_metrics,
                **spatial_metrics,
                **injected_metrics,
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
        _ = callback
        _ = disable_pbar
        resources.soft_empty_cache(force=True)
        self.load_components()
        self._validate_prepared_inputs(prepared_inputs)
        self.prepared_inputs = prepared_inputs

        attach_device = self._execution_device()
        budget_bytes = self._clean_unet_budget_bytes(attach_device)
        headroom_mb = self._device_headroom_mb(attach_device)
        unet_attach_start = time.perf_counter()
        self._attach_compiled_unet(attach_device, budget_bytes=budget_bytes)
        unet_attach_wall = time.perf_counter() - unet_attach_start

        conditioning_attach_start = time.perf_counter()
        attached_payload = self._build_attached_payload(prepared_inputs, attach_device)
        conditioning_attach_wall = time.perf_counter() - conditioning_attach_start
        self._attached_payload = attached_payload

        state = self._transition_execution_state(
            prepared_inputs,
            active_phase="diffusion",
            attached_component_ids=self._build_attached_component_ids(prepared_inputs),
            device=attach_device,
            stream_budget_mb=float(budget_bytes) / (1024 * 1024),
            headroom_mb=headroom_mb,
        )

        denoise_start = time.perf_counter()
        denoise_cpu_start = time.process_time()
        try:
            with torch.inference_mode(), precision.autocast_context(attach_device):
                samples = self._run_prepared_denoise(
                    attached_payload,
                    device=attach_device,
                    callback=callback,
                    disable_pbar=disable_pbar,
                )
        finally:
            self._park_compiled_unet_before_decode()
        denoise_wall = time.perf_counter() - denoise_start
        denoise_cpu_proc = time.process_time() - denoise_cpu_start
        latent_cpu = samples.detach().cpu()

        metrics = {
            "execution_device": 1.0 if attach_device.type == "cuda" else 0.0,
            "unet_attach_cpu": float(unet_attach_wall),
            "conditioning_attach_cpu": float(conditioning_attach_wall),
            "prepared_conditioning_reused": 1.0,
            "prepared_unet_reused": 1.0,
            "prepared_structural_reused": 1.0 if prepared_inputs.structural_conditioning is not None else 0.0,
            "prepared_spatial_reused": 1.0 if prepared_inputs.spatial_conditioning is not None else 0.0,
            "denoise_mask_attached": 1.0 if attached_payload.get("denoise_mask") is not None else 0.0,
            "attached_component_count": float(len(state.attached_component_ids)),
            "stream_budget_mb": float(state.stream_budget_mb),
            "headroom_mb": float(state.headroom_mb),
            "denoise_wall": float(denoise_wall),
            "denoise_cpu_proc": float(denoise_cpu_proc),
            "cond_prepare_explicit": float(attached_payload.get("cond_prepare_duration", 0.0)),
        }
        return UnifiedSDXLDenoiseResult(
            samples=latent_cpu,
            execution_state=state,
            metrics=metrics,
        )

    def decode_latent(self, latent: torch.Tensor, tiled: bool = False) -> tuple[torch.Tensor, float, float]:
        self.load_components()
        decode_device = self._execution_device()
        self._park_compiled_unet_before_decode()

        attach_start = time.perf_counter()
        self._attach_vae(decode_device)
        vae_attach = time.perf_counter() - attach_start

        self._transition_execution_state(
            self.prepared_inputs,
            active_phase="decode",
            attached_component_ids=self._build_decode_component_ids(),
            device=decode_device,
            stream_budget_mb=0.0,
            headroom_mb=self._device_headroom_mb(decode_device),
        )

        decode_start = time.perf_counter()
        try:
            with torch.inference_mode():
                decoded_patch = decode.decode_preloaded_vae(self.vae, latent, tiled=tiled)
                images = self._compose_decoded_images(decoded_patch)
        finally:
            if not self._is_vae_resident():
                self._detach_component(getattr(self.vae, "patcher", None))
            self._attached_payload = None
            self._transition_execution_state(
                self.prepared_inputs,
                active_phase="finalize",
                attached_component_ids=(),
                device=decode_device,
                stream_budget_mb=0.0,
                headroom_mb=self._device_headroom_mb(decode_device),
            )
        vae_decode = time.perf_counter() - decode_start
        return images, vae_attach, vae_decode

    def close(self) -> None:
        self._park_compiled_unet_before_decode()
        self._detach_component(getattr(self.vae, "patcher", None))
        self._detach_component(getattr(self.clip, "patcher", None))
        self._attached_payload = None
        self.execution_state = None
        self.prepared_inputs = None
        self.base_model = None
        self.compiled_unet = None
        self.conditioning = None
        self.structural_conditioning = None
        self.spatial_conditioning = None
        self.injected_features = {}
        self.unet = None
        self.clip = None
        self.vae = None
        self._unload_controlnets()
        self._loaded = False
        self._prepare_metrics = {}
        self._checkpoint_fingerprint = None
        gc.collect()
        resources.soft_empty_cache(force=True)

    def patched_weights_for_block(self, block_id: str) -> Any:
        _ = SDXL_RUNTIME_SURFACE_CONTRACTS["patched_weights_for_block"]
        if self.compiled_unet is None or self.unet is None:
            raise RuntimeError("Unified SDXL runtime has no compiled UNet artifact available for block retrieval.")

        # W07 keeps the unified runtime on a narrowed full-frame execution shape.
        # Expose the already-compiled execution surface for the requested block id
        # instead of asking callers to rebuild LoRA or base checkpoint patch state.
        execution_unet = self.unet
        if self._attached_payload is not None:
            execution_unet = self._attached_payload.get("execution_unet") or execution_unet

        return {
            "block_id": str(block_id),
            "artifact_fingerprint": self.compiled_unet.artifact_fingerprint,
            "execution_unet": execution_unet,
            "execution_state": self.execution_state,
            "attached": bool(self._attached_payload is not None),
        }

    def injected_features_for_block(self, block_id: str, timestep: Any, context: Any) -> Any:
        _ = SDXL_RUNTIME_SURFACE_CONTRACTS["injected_features_for_block"]
        _ = timestep
        _ = context
        payload = self._attached_payload or (self.prepared_inputs.payload if self.prepared_inputs is not None else {})
        contextual_tasks = (payload or {}).get("contextual_tasks") or {}
        if str(block_id) != "attn2":
            return None
        return contextual_tasks or None

    def _require_checkpoint_path(self) -> str:
        checkpoint_path = str(self.config.checkpoint_path or "").strip()
        if not checkpoint_path:
            raise ValueError("Unified SDXL runtime requires config.checkpoint_path for CPU-first preparation.")
        return checkpoint_path

    def _execution_class_label(self) -> str:
        return str(self.config.execution_class or self.route_label)

    def _resolved_execution_class(self):
        from backend.staging_manager import ExecutionClass

        exec_class = self.config.execution_class
        if isinstance(exec_class, str):
            normalized_exec_class = exec_class.rsplit(".", 1)[-1]
            try:
                return ExecutionClass[normalized_exec_class]
            except KeyError:
                exec_class = None

        if exec_class is not None:
            return exec_class

        policy_exec_class = getattr(self.policy, "execution_class", None)
        if isinstance(policy_exec_class, str):
            normalized_exec_class = policy_exec_class.rsplit(".", 1)[-1]
            try:
                return ExecutionClass[normalized_exec_class]
            except KeyError:
                return policy_exec_class
        return policy_exec_class

    def _require_optional_vae_path(self) -> str | None:
        value = str(self.config.vae_path or "").strip()
        return value or None

    def _execution_device(self) -> torch.device:
        try:
            return resources.get_torch_device()
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_clip_identity(self) -> str:
        clip_identity = self.config.checkpoint_path
        if not clip_identity and self.base_model is not None and self.base_model.source_path:
            clip_identity = self.base_model.source_path
        return f"{self.config.model_variant}:{clip_identity}:{self.config.clip_layer}"

    def _clean_unet_budget_bytes(self, device: torch.device) -> int:
        if device.type != "cuda":
            return 0
        if self.config.streamlike_budget_mb <= 0:
            return 0
        return max(64, int(self.config.streamlike_budget_mb)) * 1024 * 1024

    def _device_headroom_mb(self, device: torch.device) -> float:
        try:
            return float(resources.get_free_memory(device)) / (1024 * 1024)
        except Exception:
            return 0.0

    def _attach_compiled_unet(self, device: torch.device, *, budget_bytes: int = 0) -> None:
        if self.unet is None:
            raise RuntimeError("Compiled UNet is not loaded.")
        model_size = int(self.unet.model_size())
        lowvram_model_memory = 0 if budget_bytes <= 0 or budget_bytes >= model_size else int(budget_bytes)
        self.unet.patch_model(device_to=device, lowvram_model_memory=lowvram_model_memory)

    def _attach_vae(self, device: torch.device) -> None:
        if self.vae is None:
            raise RuntimeError("VAE is not loaded.")
        self.vae.patcher.patch_model(device_to=device, lowvram_model_memory=0)

    def _detach_component(self, component: Any) -> None:
        if component is None:
            return
        detach = getattr(component, "detach", None)
        if callable(detach):
            try:
                detach()
            except Exception:
                pass

    def _park_compiled_unet_before_decode(self) -> None:
        self._detach_component(self.unet)

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
        digest.update(repr(self.config.scheduler).encode("utf-8"))
        digest.update(repr(self.config.original_scheduler_name).encode("utf-8"))
        digest.update(repr(unet_compile_metrics.get("patch_count", 0)).encode("utf-8"))
        digest.update(repr(unet_compile_metrics.get("host_pinned_bytes", 0)).encode("utf-8"))
        return digest.hexdigest()
