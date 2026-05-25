from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from backend import patching, resources
from backend.flux import text_conditioning as tc
from backend.gguf.ops import GGMLOps
from backend.gguf.patcher import GGUFModelPatcher
from ldm_patched.modules import model_management

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None


_RESIDENT_ENCODER_CACHE: dict[tuple[str, str, str | None, str | None, str | None, str], tc.FluxPromptTextEncoder] = {}


def _make_stream_trace_template() -> dict[str, Any]:
    return {
        "materialize_calls": 0,
        "materialize_bytes": 0,
        "materialize_wall": 0.0,
        "materialize_cpu_proc": 0.0,
        "peak_single_tensor_bytes": 0,
        "peak_active_layer_bytes": 0,
        "linear_calls": 0,
        "embedding_calls": 0,
        "linear_compute_wall": 0.0,
        "linear_compute_cpu_proc": 0.0,
        "embedding_compute_wall": 0.0,
        "embedding_compute_cpu_proc": 0.0,
        "forward_memory_samples": [],
        "forward_peak_rss_mb": 0.0,
        "forward_peak_uss_mb": 0.0,
    }


_STREAM_TRACE_STATS = _make_stream_trace_template()
_STREAM_TRACE_OPTIONS = {
    "forward_memory_sampling": False,
}


def _capture_process_memory_snapshot() -> dict[str, float] | None:
    if psutil is None:
        return None
    try:
        process = psutil.Process()
        info = process.memory_info()
        full = None
        try:
            full = process.memory_full_info()
        except Exception:
            full = None
        return {
            "rss_mb": float(getattr(info, "rss", 0)) / (1024 * 1024),
            "vms_mb": float(getattr(info, "vms", 0)) / (1024 * 1024),
            "pagefile_mb": float(getattr(info, "pagefile", 0)) / (1024 * 1024),
            "private_mb": float(getattr(info, "private", getattr(info, "private_bytes", 0))) / (1024 * 1024),
            "uss_mb": float(getattr(full, "uss", 0)) / (1024 * 1024) if full is not None else 0.0,
        }
    except Exception:
        return None


def _append_load_phase_snapshot(phase_snapshots: list[dict[str, Any]], phase: str, phase_start: float) -> None:
    snapshot: dict[str, Any] = {
        "phase": phase,
        "elapsed_wall": time.perf_counter() - phase_start,
    }
    memory = _capture_process_memory_snapshot()
    if memory is not None:
        snapshot["memory"] = memory
    phase_snapshots.append(snapshot)


def _resident_encoder_key(
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    load_device: str | torch.device | None = None,
    offload_device: str | torch.device | None = None,
    t5_loader_policy: str | None = None,
) -> tuple[str, str, str | None, str | None, str | None, str]:
    return (
        str(Path(clip_l_path)),
        str(Path(t5_path)),
        str(Path(embedding_directory)) if embedding_directory is not None else None,
        str(torch.device(load_device)) if load_device is not None else None,
        str(torch.device(offload_device)) if offload_device is not None else None,
        _normalize_t5_loader_policy(t5_loader_policy),
    )


def clear_flux_prompt_text_encoder_cache() -> None:
    cached = list(_RESIDENT_ENCODER_CACHE.values())
    _RESIDENT_ENCODER_CACHE.clear()
    for encoder in cached:
        try:
            del encoder
        except Exception:
            pass
    try:
        resources.soft_empty_cache(force=True)
    except Exception:
        pass


def _normalize_t5_loader_policy(policy: str | None) -> str:
    value = str(policy or "eager").strip().lower().replace("-", "_").replace(" ", "_")
    if value not in {"eager", "direct_safetensors_iterative", "lazy_safetensors_runtime", "stream_safetensors_runtime"}:
        raise ValueError(f"Unsupported T5 loader policy: {policy!r}.")
    if value == "lazy_safetensors_runtime":
        return "stream_safetensors_runtime"
    return value


def reset_stream_trace_stats() -> None:
    _STREAM_TRACE_STATS.clear()
    _STREAM_TRACE_STATS.update(_make_stream_trace_template())


def set_stream_trace_forward_sampling(enabled: bool) -> None:
    _STREAM_TRACE_OPTIONS["forward_memory_sampling"] = bool(enabled)


def consume_stream_trace_stats() -> dict[str, Any]:
    snapshot = dict(_STREAM_TRACE_STATS)
    snapshot["forward_memory_sampling"] = bool(_STREAM_TRACE_OPTIONS["forward_memory_sampling"])
    reset_stream_trace_stats()
    return snapshot


def _record_materialize_stats(*, tensor_bytes: int, wall_seconds: float, cpu_process_seconds: float) -> None:
    _STREAM_TRACE_STATS["materialize_calls"] += 1
    _STREAM_TRACE_STATS["materialize_bytes"] += int(tensor_bytes)
    _STREAM_TRACE_STATS["materialize_wall"] += float(wall_seconds)
    _STREAM_TRACE_STATS["materialize_cpu_proc"] += float(cpu_process_seconds)
    _STREAM_TRACE_STATS["peak_single_tensor_bytes"] = max(_STREAM_TRACE_STATS["peak_single_tensor_bytes"], int(tensor_bytes))


def _record_active_layer_bytes(bytes_used: int) -> None:
    _STREAM_TRACE_STATS["peak_active_layer_bytes"] = max(_STREAM_TRACE_STATS["peak_active_layer_bytes"], int(bytes_used))


def _record_compute_stats(op_name: str, *, wall_seconds: float, cpu_process_seconds: float) -> None:
    key_prefix = "embedding" if op_name == "embedding" else "linear"
    _STREAM_TRACE_STATS[f"{key_prefix}_calls"] += 1
    _STREAM_TRACE_STATS[f"{key_prefix}_compute_wall"] += float(wall_seconds)
    _STREAM_TRACE_STATS[f"{key_prefix}_compute_cpu_proc"] += float(cpu_process_seconds)


def _record_forward_memory_sample(op_name: str, stage: str, *, active_bytes: int, call_index: int) -> None:
    if not _STREAM_TRACE_OPTIONS["forward_memory_sampling"]:
        return
    memory = _capture_process_memory_snapshot()
    if memory is None:
        return
    _STREAM_TRACE_STATS["forward_peak_rss_mb"] = max(
        float(_STREAM_TRACE_STATS["forward_peak_rss_mb"]),
        float(memory.get("rss_mb", 0.0)),
    )
    _STREAM_TRACE_STATS["forward_peak_uss_mb"] = max(
        float(_STREAM_TRACE_STATS["forward_peak_uss_mb"]),
        float(memory.get("uss_mb", 0.0)),
    )
    _STREAM_TRACE_STATS["forward_memory_samples"].append(
        {
            "op": op_name,
            "stage": stage,
            "call_index": int(call_index),
            "active_mb": float(active_bytes) / (1024 * 1024),
            "rss_mb": float(memory.get("rss_mb", 0.0)),
            "uss_mb": float(memory.get("uss_mb", 0.0)),
            "pagefile_mb": float(memory.get("pagefile_mb", 0.0)),
            "private_mb": float(memory.get("private_mb", 0.0)),
        }
    )


def _map_t5_source_key(source_key: str, state_entries: dict[str, Any] | None = None) -> str | None:
    target_key = source_key
    if state_entries is not None and target_key in state_entries:
        return target_key
    alias_map = {
        "encoder.embed_tokens.weight": "shared.weight",
    }
    mapped = alias_map.get(source_key, source_key)
    if state_entries is None:
        return mapped
    return mapped if mapped in state_entries else None


def _safetensors_header_stats(path: Path) -> tuple[int, int]:
    from backend.cpu_compiler import SafeOpenHeaderOnly

    header = SafeOpenHeaderOnly(str(path))
    total_bytes = 0
    for value in header.values():
        dtype = tc._normalize_checkpoint_dtype(value)
        if dtype is None:
            continue
        element_size = torch.empty((), dtype=dtype).element_size()
        numel = 1
        for dim in getattr(value, "shape", ()) or ():
            numel *= int(dim)
        total_bytes += int(numel) * int(element_size)
    return len(header), total_bytes


class LazySafetensorsLayer(torch.nn.Module):
    comfy_cast_weights = True

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        matched = False
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if suffix == "weight":
                self.weight = value
                matched = True
            elif suffix == "bias":
                self.bias = value
                matched = True
            else:
                unexpected_keys.append(key)
        if not matched:
            missing_keys.append(prefix + "weight")

    @staticmethod
    def _materialize(value: Any, *, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor | None:
        if value is None:
            return None
        load_start = time.perf_counter()
        load_cpu_start = time.process_time()
        if hasattr(value, "load") and callable(value.load):
            value = value.load()
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor-like lazy weight, got {type(value).__name__}.")
        tensor_bytes = int(value.numel() * value.element_size())
        _record_materialize_stats(
            tensor_bytes=tensor_bytes,
            wall_seconds=time.perf_counter() - load_start,
            cpu_process_seconds=time.process_time() - load_cpu_start,
        )
        if dtype is None:
            return value.to(device=device)
        return value.to(device=device, dtype=dtype)


class LazySafetensorsOps(tc.base_ops.manual_cast):
    class Linear(LazySafetensorsLayer, tc.base_ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward(self, input):
            weight = self._materialize(self.weight, device=input.device, dtype=input.dtype)
            bias = self._materialize(self.bias, device=input.device, dtype=input.dtype) if self.bias is not None else None
            active_bytes = int(weight.numel() * weight.element_size())
            if bias is not None:
                active_bytes += int(bias.numel() * bias.element_size())
            _record_active_layer_bytes(active_bytes)
            call_index = int(_STREAM_TRACE_STATS["linear_calls"]) + 1
            _record_forward_memory_sample("linear", "post_materialize", active_bytes=active_bytes, call_index=call_index)
            compute_start = time.perf_counter()
            compute_cpu_start = time.process_time()
            output = torch.nn.functional.linear(input, weight, bias)
            _record_compute_stats(
                "linear",
                wall_seconds=time.perf_counter() - compute_start,
                cpu_process_seconds=time.process_time() - compute_cpu_start,
            )
            _record_forward_memory_sample("linear", "post_compute", active_bytes=active_bytes, call_index=call_index)
            return output

    class Embedding(LazySafetensorsLayer, tc.base_ops.manual_cast.Embedding):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.num_embeddings = num_embeddings
            self.embedding_dim = embedding_dim
            self.padding_idx = padding_idx
            self.max_norm = max_norm
            self.norm_type = norm_type
            self.scale_grad_by_freq = scale_grad_by_freq
            self.sparse = sparse
            self.weight = None
            self.bias = None

        def forward(self, input, out_dtype=None):
            weight_dtype = None
            if out_dtype is not None:
                weight_dtype = out_dtype
            weight = self._materialize(self.weight, device=input.device, dtype=weight_dtype)
            active_bytes = int(weight.numel() * weight.element_size())
            _record_active_layer_bytes(active_bytes)
            call_index = int(_STREAM_TRACE_STATS["embedding_calls"]) + 1
            _record_forward_memory_sample("embedding", "post_materialize", active_bytes=active_bytes, call_index=call_index)
            compute_start = time.perf_counter()
            compute_cpu_start = time.process_time()
            out = torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            _record_compute_stats(
                "embedding",
                wall_seconds=time.perf_counter() - compute_start,
                cpu_process_seconds=time.process_time() - compute_cpu_start,
            )
            _record_forward_memory_sample("embedding", "post_compute", active_bytes=active_bytes, call_index=call_index)
            if out_dtype is not None:
                out = out.to(dtype=out_dtype)
            return out


def _build_lazy_t5_state_dict(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from backend.cpu_compiler import SafeOpenHeaderOnly

    header = SafeOpenHeaderOnly(str(path))
    lazy_state_dict: dict[str, Any] = {}
    duplicate_source_keys: list[str] = []
    for source_key, value in header.items():
        target_key = _map_t5_source_key(source_key, None)
        if target_key is None:
            continue
        if target_key in lazy_state_dict:
            duplicate_source_keys.append(source_key)
            continue
        lazy_state_dict[target_key] = value
    return lazy_state_dict, {
        "custom_operations": LazySafetensorsOps,
        "lazy_safetensors_runtime": True,
        "lazy_duplicate_source_keys": duplicate_source_keys,
    }


def _load_t5_direct_safetensors_into_model(
    *,
    cond_stage_model: tc.FluxClipModel,
    t5_path: Path,
    target_device: torch.device,
) -> dict[str, Any]:
    key_count, raw_bytes = _safetensors_header_stats(t5_path)
    state_entries = cond_stage_model.t5xxl.transformer.state_dict()
    loaded_keys: set[str] = set()
    unexpected: list[str] = []

    start = time.perf_counter()
    with safe_open(str(t5_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            target_key = _map_t5_source_key(key, state_entries)
            if target_key is None:
                unexpected.append(key)
                continue
            target_tensor = state_entries[target_key]
            source_tensor = handle.get_tensor(key)
            copy_tensor = source_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            target_tensor.copy_(copy_tensor)
            loaded_keys.add(target_key)
    load_wall = time.perf_counter() - start
    missing = [key for key in state_entries.keys() if key not in loaded_keys]
    if missing:
        raise ValueError(
            f"Missing T5 keys while directly loading {t5_path}: {missing[:8]}"
            + (" ..." if len(missing) > 8 else "")
        )
    if unexpected:
        raise ValueError(
            f"Unexpected T5 keys while directly loading {t5_path}: {unexpected[:8]}"
            + (" ..." if len(unexpected) > 8 else "")
        )
    return {
        "t5_loader_policy": "direct_safetensors_iterative",
        "t5_direct_assign": True,
        "t5_direct_assign_key_count": int(key_count),
        "t5_direct_assign_mb": float(raw_bytes) / (1024 * 1024),
        "t5_iterative_load_wall": float(load_wall),
        "t5_source_kind": "safetensors_header_iterative_assign",
        "t5_full_state_dict_materialized": False,
    }


def load_flux_prompt_text_encoder(
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    load_device: str | torch.device | None = None,
    offload_device: str | torch.device | None = None,
    t5_loader_policy: str | None = None,
) -> tc.FluxPromptTextEncoder:
    phase_snapshots: list[dict[str, Any]] = []
    phase_start = time.perf_counter()
    clip_l_path = Path(clip_l_path)
    t5_path = Path(t5_path)
    clip_l_sd, clip_l_options = tc._load_text_encoder_state_dict(clip_l_path)
    _append_load_phase_snapshot(phase_snapshots, "clip_l_state_dict_loaded", phase_start)
    t5_loader_policy = _normalize_t5_loader_policy(t5_loader_policy)
    direct_t5 = t5_loader_policy == "direct_safetensors_iterative"
    lazy_t5 = t5_loader_policy in {"lazy_safetensors_runtime", "stream_safetensors_runtime"}
    if (direct_t5 or lazy_t5) and t5_path.suffix.lower() != ".safetensors":
        raise ValueError(f"{t5_loader_policy} T5 loader policy requires a .safetensors checkpoint.")

    t5_sd = None
    t5_options: dict[str, Any] = {}
    phase_start = time.perf_counter()
    if lazy_t5:
        t5_sd, t5_options = _build_lazy_t5_state_dict(t5_path)
        detected_t5_dtype = tc._detect_t5_dtype(t5_sd)
    elif not direct_t5:
        t5_sd, t5_options = tc._load_text_encoder_state_dict(t5_path)
        detected_t5_dtype = tc._detect_t5_dtype(t5_sd)
    else:
        from backend.cpu_compiler import SafeOpenHeaderOnly

        detected_t5_dtype = tc._detect_t5_dtype(SafeOpenHeaderOnly(str(t5_path)))
    _append_load_phase_snapshot(phase_snapshots, "t5_source_prepared", phase_start)

    load_device = torch.device(load_device) if load_device is not None else torch.device("cpu")
    offload_device = torch.device(offload_device) if offload_device is not None else torch.device("cpu")
    dtype = model_management.text_encoder_dtype(load_device)
    model_options: dict[str, Any] = {}
    model_options.update(clip_l_options)
    model_options.update(t5_options)
    initial_device = torch.device("cpu")
    model_options["initial_device"] = initial_device

    phase_start = time.perf_counter()
    cond_stage_model = tc.FluxClipModel(
        dtype_t5=detected_t5_dtype,
        device=initial_device,
        dtype=dtype,
        model_options=model_options,
    )
    tokenizer = tc.FluxTokenizer(embedding_directory=embedding_directory)
    patcher_cls = GGUFModelPatcher if model_options.get("custom_operations") is GGMLOps else patching.NexModelPatcher
    patcher = patcher_cls(cond_stage_model, load_device=load_device, offload_device=offload_device)
    _append_load_phase_snapshot(phase_snapshots, "encoder_module_constructed", phase_start)

    phase_start = time.perf_counter()
    missing, unexpected = cond_stage_model.load_sd(clip_l_sd)
    if missing:
        tc.logger.debug("Flux CLIP-L missing keys: %s", missing)
    if unexpected:
        tc.logger.debug("Flux CLIP-L unexpected keys: %s", unexpected)
    _append_load_phase_snapshot(phase_snapshots, "clip_l_weights_applied", phase_start)

    load_metadata = {
        "clip_l_path": str(clip_l_path),
        "t5_path": str(t5_path),
        "t5_loader_policy": t5_loader_policy,
        "t5_detected_dtype": str(detected_t5_dtype) if detected_t5_dtype is not None else None,
        "t5_direct_assign": False,
        "t5_direct_assign_key_count": 0,
        "t5_direct_assign_mb": 0.0,
        "t5_iterative_load_wall": 0.0,
        "t5_source_kind": "eager_state_dict",
        "t5_full_state_dict_materialized": True,
        "t5_lazy_runtime": False,
        "t5_stream_runtime": False,
    }
    if direct_t5:
        phase_start = time.perf_counter()
        load_metadata.update(
            _load_t5_direct_safetensors_into_model(
                cond_stage_model=cond_stage_model,
                t5_path=t5_path,
                target_device=initial_device,
            )
        )
        _append_load_phase_snapshot(phase_snapshots, "t5_weights_applied_direct", phase_start)
    elif lazy_t5:
        assert t5_sd is not None
        phase_start = time.perf_counter()
        missing, unexpected = cond_stage_model.load_sd(t5_sd)
        if missing:
            tc.logger.debug("Flux T5 lazy runtime missing keys: %s", missing)
        if unexpected:
            tc.logger.debug("Flux T5 lazy runtime unexpected keys: %s", unexpected)
        key_count, raw_bytes = _safetensors_header_stats(t5_path)
        load_metadata.update(
            {
                "t5_direct_assign": False,
                "t5_direct_assign_key_count": int(key_count),
                "t5_direct_assign_mb": float(raw_bytes) / (1024 * 1024),
                "t5_iterative_load_wall": 0.0,
                "t5_source_kind": "safetensors_lazy_runtime",
                "t5_full_state_dict_materialized": False,
                "t5_lazy_runtime": True,
                "t5_stream_runtime": t5_loader_policy == "stream_safetensors_runtime",
                "t5_lazy_duplicate_source_keys": list(t5_options.get("lazy_duplicate_source_keys", [])),
            }
        )
        _append_load_phase_snapshot(phase_snapshots, "t5_weights_applied_lazy", phase_start)
    else:
        assert t5_sd is not None
        phase_start = time.perf_counter()
        missing, unexpected = cond_stage_model.load_sd(t5_sd)
        if missing:
            tc.logger.debug("Flux T5 missing keys: %s", missing)
        if unexpected:
            tc.logger.debug("Flux T5 unexpected keys: %s", unexpected)
        _append_load_phase_snapshot(phase_snapshots, "t5_weights_applied_eager", phase_start)

    phase_start = time.perf_counter()
    encoder = tc.FluxPromptTextEncoder(cond_stage_model=cond_stage_model, tokenizer=tokenizer, patcher=patcher)
    _append_load_phase_snapshot(phase_snapshots, "encoder_ready", phase_start)
    load_metadata["load_phase_snapshots"] = phase_snapshots
    setattr(encoder, "_nex_load_metadata", load_metadata)
    return encoder


def get_flux_prompt_text_encoder(
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    load_device: str | torch.device | None = None,
    offload_device: str | torch.device | None = None,
    keep_resident: bool = False,
    t5_loader_policy: str | None = None,
) -> tc.FluxPromptTextEncoder:
    t5_loader_policy = _normalize_t5_loader_policy(t5_loader_policy)
    if not keep_resident:
        return load_flux_prompt_text_encoder(
            clip_l_path=clip_l_path,
            t5_path=t5_path,
            embedding_directory=embedding_directory,
            load_device=load_device,
            offload_device=offload_device,
            t5_loader_policy=t5_loader_policy,
        )

    key = _resident_encoder_key(
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        load_device=load_device,
        offload_device=offload_device,
        t5_loader_policy=t5_loader_policy,
    )
    cached = _RESIDENT_ENCODER_CACHE.get(key)
    if cached is not None:
        load_metadata = dict(getattr(cached, "_nex_load_metadata", {}) or {})
        load_metadata["resident_cache_hit"] = True
        setattr(cached, "_nex_load_metadata", load_metadata)
        return cached

    clear_flux_prompt_text_encoder_cache()
    encoder = load_flux_prompt_text_encoder(
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        load_device=load_device,
        offload_device=offload_device,
        t5_loader_policy=t5_loader_policy,
    )
    load_metadata = dict(getattr(encoder, "_nex_load_metadata", {}) or {})
    load_metadata["resident_cache_hit"] = False
    setattr(encoder, "_nex_load_metadata", load_metadata)
    _RESIDENT_ENCODER_CACHE[key] = encoder
    return encoder


def encode_flux_prompt_conditioning(
    prompt: str,
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    load_device: str | torch.device | None = None,
    offload_device: str | torch.device | None = None,
    keep_resident: bool = False,
    t5_loader_policy: str | None = None,
) -> tc.FluxEmptyConditioning:
    prompt_text = str(prompt or "").strip()
    if prompt_text == "":
        raise ValueError("Flux prompt conditioning requires a non-empty prompt.")

    encoder = get_flux_prompt_text_encoder(
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        load_device=load_device,
        offload_device=offload_device,
        keep_resident=keep_resident,
        t5_loader_policy=t5_loader_policy,
    )
    try:
        cross_attn, pooled_output = encoder.encode(prompt_text)
        return tc.FluxEmptyConditioning(
            cross_attn=cross_attn.to(device="cpu"),
            pooled_output=pooled_output.to(device="cpu"),
            metadata={
                "prompt": prompt_text,
                "clip_l_path": str(clip_l_path),
                "t5_path": str(t5_path),
                "t5_format": "gguf" if str(t5_path).lower().endswith(".gguf") else "safetensors",
                "generator": "tools/flux_text_conditioning_experiments.py",
                "conditioning_kind": "prompt",
                "transport": "memory",
                "text_encoder_resident": bool(keep_resident),
                "load_device": str(torch.device(load_device)) if load_device is not None else "cpu",
                "offload_device": str(torch.device(offload_device)) if offload_device is not None else "cpu",
                "t5_loader_policy": _normalize_t5_loader_policy(t5_loader_policy),
            },
        )
    finally:
        if not keep_resident:
            del encoder
            gc.collect()
            try:
                resources.soft_empty_cache()
            except Exception:
                pass


def save_flux_prompt_conditioning_cache(
    prompt: str,
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    output_path: str | Path,
    embedding_directory: str | Path | None = None,
    load_device: str | torch.device | None = None,
    offload_device: str | torch.device | None = None,
    keep_resident: bool = False,
    t5_loader_policy: str | None = None,
) -> tc.FluxEmptyConditioning:
    conditioning = encode_flux_prompt_conditioning(
        prompt,
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        load_device=load_device,
        offload_device=offload_device,
        keep_resident=keep_resident,
        t5_loader_policy=t5_loader_policy,
    )
    return tc.save_flux_empty_conditioning_cache(
        output_path,
        cross_attn=conditioning.cross_attn,
        pooled_output=conditioning.pooled_output,
        metadata=dict(conditioning.metadata, transport="pt_cache"),
    )
