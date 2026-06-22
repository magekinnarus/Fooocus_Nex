from __future__ import annotations

import time
from typing import Any
import torch
from backend import resources

class FluxAsyncLayerPrefetchScheduler:
    def __init__(
        self,
        *,
        prefetch_depth: int = 1,
        max_prefetch_bytes: int | None = None,
        vram_guard_bytes: int | None = None,
        vram_guard_margin_bytes: int = 0,
        prefetch_scan_ahead: int = 1,
        bandwidth_limit_mb_s: float | None = None,
    ) -> None:
        self.prefetch_depth = max(0, int(prefetch_depth))
        self.max_prefetch_bytes = int(max_prefetch_bytes) if max_prefetch_bytes is not None else None
        self.vram_guard_bytes = int(vram_guard_bytes) if vram_guard_bytes is not None else None
        self.vram_guard_margin_bytes = max(0, int(vram_guard_margin_bytes))
        self.prefetch_scan_ahead = max(1, int(prefetch_scan_ahead))
        self.bandwidth_limit_mb_s = (
            float(bandwidth_limit_mb_s)
            if bandwidth_limit_mb_s is not None and float(bandwidth_limit_mb_s) > 0.0
            else None
        )
        self._hooks: list[Any] = []
        self._ordered_modules: list[Any] = []
        self._module_indices: dict[int, int] = {}
        self._module_names: dict[int, str] = {}
        self._prefetched: dict[tuple[int, str, str, str], tuple[torch.Tensor, torch.Tensor | None, Any, dict[str, Any] | None]] = {}
        self._streams: dict[str, list[Any]] = {}
        self._stats: dict[str, Any] = {}
        self.reset_run(clear_prefetched=True)

    def attach(self, model: Any, device: torch.device | None = None) -> int:
        self.detach()
        diffusion_model = getattr(model, "diffusion_model", model)
        ordered: list[Any] = []
        module_names: dict[int, str] = {}
        for name, module in diffusion_model.named_modules():
            if module is diffusion_model:
                continue
            if not hasattr(module, "comfy_cast_weights"):
                continue
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor) or weight.device.type != "cpu":
                continue
            ordered.append(module)
            module_names[id(module)] = str(name or module.__class__.__name__)

        self._ordered_modules = ordered
        self._module_indices = {id(module): index for index, module in enumerate(ordered)}
        self._module_names = module_names
        for index, module in enumerate(ordered):
            setattr(module, "_nex_streaming_scheduler", self)
            setattr(module, "_nex_streaming_scheduler_index", index)
            self._hooks.append(module.register_forward_pre_hook(self._build_prefetch_hook(index)))

        self._stats["module_count"] = len(ordered)

        # Eagerly allocate streams when device is specified and has CUDA capability
        if device is not None and self.is_enabled_for(device):
            device_key = self._device_key(device)
            if device_key not in self._streams:
                try:
                    self._streams[device_key] = [
                        torch.cuda.Stream(device=device, priority=0),
                        torch.cuda.Stream(device=device, priority=0),
                    ]
                except Exception as e:
                    import logging
                    logging.warning(
                        f"[Nex-Streaming] Failed to eagerly allocate CUDA streams on {device}: {e}"
                    )
        return len(ordered)

    def detach(self) -> None:
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks.clear()
        for module in self._ordered_modules:
            try:
                if getattr(module, "_nex_streaming_scheduler", None) is self:
                    delattr(module, "_nex_streaming_scheduler")
            except Exception:
                pass
            try:
                if hasattr(module, "_nex_streaming_scheduler_index"):
                    delattr(module, "_nex_streaming_scheduler_index")
            except Exception:
                pass
        self._ordered_modules = []
        self._module_indices = {}
        self._module_names = {}
        self._streams.clear()
        self.reset_run(clear_prefetched=True)

    def reset_run(self, *, clear_prefetched: bool = True) -> None:
        if clear_prefetched:
            self._prefetched.clear()
        self._stats = {
            "enabled": bool(torch.cuda.is_available()),
            "prefetch_depth": int(self.prefetch_depth),
            "max_prefetch_bytes": int(self.max_prefetch_bytes or 0),
            "vram_guard_bytes": int(self.vram_guard_bytes or 0),
            "vram_guard_margin_bytes": int(self.vram_guard_margin_bytes),
            "prefetch_scan_ahead": int(self.prefetch_scan_ahead),
            "bandwidth_limit_mb_s": float(self.bandwidth_limit_mb_s or 0.0),
            "module_count": len(self._ordered_modules),
            "prefetch_enqueued": 0,
            "prefetch_hits": 0,
            "prefetch_misses": 0,
            "prefetch_bytes": 0,
            "prefetch_copy_wall_s": 0.0,
            "prefetch_copy_cuda_ms": 0.0,
            "prefetch_throttle_cuda_ms": 0.0,
            "prefetch_throttle_events": 0,
            "prefetch_skipped_size": 0,
            "prefetch_skipped_vram": 0,
            "prefetch_scan_considered": 0,
            "direct_copy_bytes": 0,
            "direct_copy_cuda_ms": 0.0,
            "direct_throttle_cuda_ms": 0.0,
            "direct_throttle_events": 0,
            "direct_copy_calls": 0,
            "direct_copy_stream_uses": 0,
            "bandwidth_throttle_cuda_ms": 0.0,
            "bandwidth_throttle_events": 0,
            "stream_waits": 0,
            "sync_calls": 0,
            "module_profiles": {},
        }

    def snapshot(self) -> dict[str, Any]:
        snapshot = dict(self._stats)
        snapshot["streams"] = {device: len(streams) for device, streams in self._streams.items()}
        snapshot["prefetched_entries"] = len(self._prefetched)
        profiles = list(self._stats.get("module_profiles", {}).values())
        snapshot["top_direct_modules"] = sorted(
            profiles,
            key=lambda item: float(item.get("direct_wall_s", 0.0)),
            reverse=True,
        )[:12]
        snapshot["top_prefetch_modules"] = sorted(
            profiles,
            key=lambda item: int(item.get("prefetch_hits", 0)),
            reverse=True,
        )[:12]
        snapshot["top_vram_skipped_modules"] = sorted(
            profiles,
            key=lambda item: int(item.get("prefetch_skipped_vram", 0)),
            reverse=True,
        )[:12]
        return snapshot

    def is_enabled_for(self, device: torch.device | None) -> bool:
        return bool(
            torch.cuda.is_available()
            and device is not None
            and isinstance(device, torch.device)
            and device.type == "cuda"
            and self._ordered_modules
        )

    def _device_key(self, device: torch.device) -> str:
        return str(device)

    def _stream_for_index(self, device: torch.device, module_index: int):
        if not self.is_enabled_for(device):
            return None
        device_key = self._device_key(device)
        streams = self._streams.get(device_key)
        if streams is None:
            try:
                streams = [torch.cuda.Stream(device=device, priority=0), torch.cuda.Stream(device=device, priority=0)]
                self._streams[device_key] = streams
            except Exception as e:
                import logging
                logging.warning(
                    f"[Nex-Streaming] Failed to lazily create CUDA streams on {device}: {e}. "
                    f"Falling back to default stream."
                )
                return None
        return streams[int(module_index) % len(streams)]

    def stream_for_module(self, module: Any, *, device: torch.device):
        module_index = int(self._module_indices.get(id(module), 0))
        stream = self._stream_for_index(device, module_index)
        if stream is not None:
            self._stats["direct_copy_stream_uses"] += 1
        return stream

    def sync_stream(self, device: torch.device, stream: Any) -> None:
        if stream is None or not self.is_enabled_for(device):
            return
        torch.cuda.current_stream(device).wait_stream(stream)
        self._stats["sync_calls"] += 1
        self._stats["stream_waits"] += 1

    def consume_prefetched(
        self,
        module: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
        bias_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        key = (id(module), str(device), str(dtype), str(bias_dtype))
        entry = self._prefetched.pop(key, None)
        if entry is None:
            self._stats["prefetch_misses"] += 1
            return None
        weight, bias, stream, copy_token = entry
        self.sync_stream(device, stream)
        self._settle_timed_copy(copy_token)
        self._stats["prefetch_hits"] += 1
        profile = self._profile(module)
        profile["prefetch_hits"] += 1
        return weight, bias

    def _build_prefetch_hook(self, module_index: int):
        def _hook(module: Any, args: tuple[Any, ...]) -> None:
            if not args:
                return
            input_tensor = args[0]
            if not isinstance(input_tensor, torch.Tensor):
                return
            device = input_tensor.device
            if not self.is_enabled_for(device):
                return
            selected_indices = self._select_prefetch_indices(module_index)
            enqueued = 0
            for next_index in selected_indices:
                next_module = self._ordered_modules[next_index]
                if self._prefetch_module(
                    next_module,
                    device=device,
                    dtype=input_tensor.dtype,
                    bias_dtype=input_tensor.dtype,
                    module_index=next_index,
                ):
                    enqueued += 1
                    if enqueued >= self.prefetch_depth:
                        break
        return _hook

    def _select_prefetch_indices(self, module_index: int) -> list[int]:
        if self.prefetch_depth <= 0:
            return []
        start_index = int(module_index) + 1
        if start_index >= len(self._ordered_modules):
            return []
        end_index = min(len(self._ordered_modules), start_index + int(self.prefetch_scan_ahead))
        candidates: list[tuple[tuple[int, int, int], int]] = []
        for next_index in range(start_index, end_index):
            self._stats["prefetch_scan_considered"] += 1
            module = self._ordered_modules[next_index]
            priority = self._module_prefetch_priority(module, next_index)
            candidates.append((priority, next_index))
        candidates.sort(reverse=True)
        return [index for _priority, index in candidates]

    def _prefetch_module(
        self,
        self_module: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
        bias_dtype: torch.dtype,
        module_index: int,
    ) -> bool:
        if not self.is_enabled_for(device):
            return False
        key = (id(self_module), str(device), str(dtype), str(bias_dtype))
        if key in self._prefetched:
            return False

        weight = getattr(self_module, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.device.type != "cpu":
            return False

        stream = self._stream_for_index(device, module_index)
        if stream is None:
            return False

        projected_bytes = self._project_prefetch_bytes(
            weight=weight,
            bias=getattr(self_module, "bias", None),
            dtype=dtype,
            bias_dtype=bias_dtype,
        )
        if self.max_prefetch_bytes is not None and projected_bytes > self.max_prefetch_bytes:
            self._stats["prefetch_skipped_size"] += 1
            self._profile(self_module)["prefetch_skipped_size"] += 1
            return False
        if self.vram_guard_bytes is not None and self._would_exceed_vram_guard(
            device=device,
            projected_bytes=projected_bytes,
        ):
            self._stats["prefetch_skipped_vram"] += 1
            self._profile(self_module)["prefetch_skipped_vram"] += 1
            return False

        non_blocking = bool(resources.device_supports_non_blocking(device))
        has_weight_function = len(getattr(self_module, "weight_function", []) or []) > 0
        copy_start = time.perf_counter()
        copy_token = self._begin_timed_copy(
            kind="prefetch",
            device=device,
            stream=stream,
            transfer_bytes=projected_bytes,
        )
        weight_copy = resources.cast_to(weight, dtype=dtype, device=device, non_blocking=non_blocking, copy=has_weight_function, stream=stream)
        if has_weight_function:
            with torch.cuda.stream(stream):
                for function in self_module.weight_function:
                    weight_copy = function(weight_copy)

        bias = getattr(self_module, "bias", None)
        bias_copy = None
        if isinstance(bias, torch.Tensor):
            has_bias_function = len(getattr(self_module, "bias_function", []) or []) > 0
            bias_copy = resources.cast_to(bias, dtype=bias_dtype, device=device, non_blocking=non_blocking, copy=has_bias_function, stream=stream)
            if has_bias_function:
                with torch.cuda.stream(stream):
                    for function in self_module.bias_function:
                        bias_copy = function(bias_copy)

        self._end_timed_copy(copy_token, stream=stream)
        self._prefetched[key] = (weight_copy, bias_copy, stream, copy_token)
        self._stats["prefetch_enqueued"] += 1
        self._stats["prefetch_bytes"] += int(projected_bytes)
        self._stats["prefetch_copy_wall_s"] += float(time.perf_counter() - copy_start)
        profile = self._profile(self_module)
        profile["prefetch_enqueued"] += 1
        profile["prefetch_bytes"] += int(projected_bytes)
        return True

    def _project_prefetch_bytes(
        self,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        dtype: torch.dtype,
        bias_dtype: torch.dtype,
    ) -> int:
        total = int(weight.numel()) * int(torch.tensor([], dtype=dtype).element_size())
        if isinstance(bias, torch.Tensor):
            total += int(bias.numel()) * int(torch.tensor([], dtype=bias_dtype).element_size())
        return total

    def estimate_module_transfer_bytes(
        self,
        module: Any,
        *,
        dtype: torch.dtype,
        bias_dtype: torch.dtype,
    ) -> int:
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.Tensor):
            return 0
        return self._project_prefetch_bytes(
            weight=weight,
            bias=getattr(module, "bias", None),
            dtype=dtype,
            bias_dtype=bias_dtype,
        )

    def _target_copy_ms(self, transfer_bytes: int) -> float:
        if self.bandwidth_limit_mb_s is None or self.bandwidth_limit_mb_s <= 0.0:
            return 0.0
        if transfer_bytes <= 0:
            return 0.0
        return (float(transfer_bytes) / (float(self.bandwidth_limit_mb_s) * 1024.0 * 1024.0)) * 1000.0

    def _apply_bandwidth_throttle(self, *, device: torch.device, delay_ms: float) -> float:
        if delay_ms <= 0.0:
            return 0.0
        if not isinstance(device, torch.device) or device.type != "cuda" or not torch.cuda.is_available():
            return 0.0
        try:
            clock_rate_khz = int(torch.cuda.get_device_properties(device).clock_rate)
        except Exception:
            return 0.0
        if clock_rate_khz <= 0:
            return 0.0
        cycles = int(max(1, round(float(delay_ms) * float(clock_rate_khz))))
        try:
            with torch.cuda.device(device):
                torch.cuda._sleep(cycles)
            return float(delay_ms)
        except Exception:
            return 0.0

    def _module_prefetch_priority(self, module: Any, module_index: int) -> tuple[int, int, int]:
        label = self._module_label(module)
        score = 0
        if "txt_attn.qkv" in label:
            score += 9
        if "txt_mod.lin" in label:
            score += 8
        if "single_blocks." in label and ".linear2" in label:
            score += 7
        if "single_blocks." in label and ".modulation.lin" in label:
            score += 5
        if "img_attn.qkv" in label:
            score += 4
        if ".linear2" in label:
            score += 3
        if ".img_mlp.0" in label or ".txt_mlp.0" in label:
            score += 2
        if ".proj" in label:
            score += 1

        later_bias = int(module_index)
        transfer_bytes = self.estimate_module_transfer_bytes(
            module,
            dtype=torch.float16,
            bias_dtype=torch.float16,
        )
        size_bias = int(transfer_bytes // (16 * 1024 * 1024))
        return (score, later_bias, size_bias)

    def record_module_wall(
        self,
        module: Any,
        *,
        path: str,
        wall_s: float,
    ) -> None:
        profile = self._profile(module)
        if path == "prefetch":
            profile["prefetch_consumes"] += 1
            profile["prefetch_wall_s"] += float(max(0.0, wall_s))
        elif path == "direct":
            profile["direct_calls"] += 1
            profile["direct_wall_s"] += float(max(0.0, wall_s))

    def begin_direct_copy(
        self,
        module: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
        bias_dtype: torch.dtype,
        stream: Any,
    ) -> dict[str, Any] | None:
        transfer_bytes = self.estimate_module_transfer_bytes(
            module,
            dtype=dtype,
            bias_dtype=bias_dtype,
        )
        if transfer_bytes <= 0:
            return None
        return self._begin_timed_copy(
            kind="direct",
            device=device,
            stream=stream,
            transfer_bytes=transfer_bytes,
        )

    def end_direct_copy(self, token: dict[str, Any] | None, *, stream: Any) -> None:
        self._end_timed_copy(token, stream=stream)

    def settle_direct_copy(self, token: dict[str, Any] | None) -> None:
        self._settle_timed_copy(token)

    def _begin_timed_copy(
        self,
        *,
        kind: str,
        device: torch.device,
        stream: Any,
        transfer_bytes: int,
    ) -> dict[str, Any] | None:
        token: dict[str, Any] = {
            "kind": str(kind),
            "bytes": int(max(0, transfer_bytes)),
            "device": device,
            "start_event": None,
            "end_event": None,
        }
        if (
            stream is not None
            and isinstance(device, torch.device)
            and device.type == "cuda"
            and torch.cuda.is_available()
        ):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            token["start_event"] = start_event
            token["end_event"] = end_event
        return token

    def _end_timed_copy(self, token: dict[str, Any] | None, *, stream: Any) -> None:
        if not token or stream is None:
            return
        end_event = token.get("end_event")
        if end_event is not None:
            end_event.record(stream)

    def _settle_timed_copy(self, token: dict[str, Any] | None) -> None:
        if not token:
            return
        kind = str(token.get("kind", ""))
        transfer_bytes = int(token.get("bytes", 0))
        device = token.get("device")
        elapsed_ms = 0.0
        start_event = token.get("start_event")
        end_event = token.get("end_event")
        if start_event is not None and end_event is not None:
            try:
                end_event.synchronize()
            except Exception:
                pass
            try:
                elapsed_ms = float(start_event.elapsed_time(end_event))
            except Exception:
                elapsed_ms = 0.0
        target_ms = self._target_copy_ms(transfer_bytes)
        throttle_ms = max(0.0, float(target_ms) - float(elapsed_ms))
        applied_throttle_ms = 0.0
        if isinstance(device, torch.device):
            applied_throttle_ms = self._apply_bandwidth_throttle(device=device, delay_ms=throttle_ms)
        if kind == "prefetch":
            self._stats["prefetch_copy_cuda_ms"] += elapsed_ms
            self._stats["prefetch_throttle_cuda_ms"] += applied_throttle_ms
            if applied_throttle_ms > 0.0:
                self._stats["prefetch_throttle_events"] += 1
        elif kind == "direct":
            self._stats["direct_copy_calls"] += 1
            self._stats["direct_copy_bytes"] += transfer_bytes
            self._stats["direct_copy_cuda_ms"] += elapsed_ms
            self._stats["direct_throttle_cuda_ms"] += applied_throttle_ms
            if applied_throttle_ms > 0.0:
                self._stats["direct_throttle_events"] += 1
        if applied_throttle_ms > 0.0:
            self._stats["bandwidth_throttle_cuda_ms"] += applied_throttle_ms
            self._stats["bandwidth_throttle_events"] += 1

    def _module_label(self, module: Any) -> str:
        module_id = id(module)
        name = self._module_names.get(module_id)
        index = int(self._module_indices.get(module_id, -1))
        if name:
            return f"{index}:{name}"
        return f"{index}:{module.__class__.__name__}"

    def _profile(self, module: Any) -> dict[str, Any]:
        profiles = self._stats.setdefault("module_profiles", {})
        module_id = id(module)
        profile = profiles.get(module_id)
        if profile is None:
            profile = {
                "label": self._module_label(module),
                "index": int(self._module_indices.get(module_id, -1)),
                "class_name": module.__class__.__name__,
                "prefetch_enqueued": 0,
                "prefetch_hits": 0,
                "prefetch_consumes": 0,
                "prefetch_bytes": 0,
                "prefetch_wall_s": 0.0,
                "prefetch_skipped_size": 0,
                "prefetch_skipped_vram": 0,
                "direct_calls": 0,
                "direct_wall_s": 0.0,
            }
            profiles[module_id] = profile
        return profile

    def _would_exceed_vram_guard(self, *, device: torch.device, projected_bytes: int) -> bool:
        if self.vram_guard_bytes is None:
            return False
        try:
            current_allocated = int(torch.cuda.memory_allocated(device))
            projected_usage = current_allocated + int(projected_bytes) + int(self.vram_guard_margin_bytes)
            if projected_usage > int(self.vram_guard_bytes):
                return True
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
            required_free_bytes = int(projected_bytes) + int(self.vram_guard_margin_bytes)
            return int(free_bytes) < required_free_bytes
        except Exception:
            return False
