import logging
import os
import sys

import torch
from einops import rearrange
from torch import Tensor

import ldm_patched.modules.model_management

logger = logging.getLogger(__name__)

try:
    import xformers
    import xformers.ops
except Exception:
    xformers = None


_FLUX_ATTENTION_ENV_KEY = "NEX_FLUX_ATTENTION_BACKEND"
_FLUX_ATTENTION_VALID_BACKENDS = {"auto", "sdpa", "xformers", "xformers_only"}
_FLUX_ATTENTION_RUNTIME_BACKENDS: dict[tuple[str, str], str] = {}
_FLUX_ATTENTION_LOGGED_KEYS: set[tuple[str, str]] = set()
_BROKEN_XFORMERS = False

if xformers is not None:
    try:
        x_vers = str(getattr(xformers, "__version__", "") or "")
        # Mirror the existing cross-attention guard for the xformers releases
        # known to fail on some CUDA launch configurations.
        _BROKEN_XFORMERS = (
            x_vers.startswith("0.0.21")
            or x_vers.startswith("0.0.22")
            or x_vers.startswith("0.0.23")
        )
    except Exception:
        pass


def _normalize_flux_attention_backend(value: str | None) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "default": "auto",
        "pytorch": "sdpa",
        "pytorch_sdp": "sdpa",
        "prefer_xformers": "xformers",
        "xformers_prefer": "xformers",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in _FLUX_ATTENTION_VALID_BACKENDS:
        logger.warning(
            "[Flux Attention] Unknown backend policy %r; falling back to auto.",
            value,
        )
        return "auto"
    return normalized


def _resolve_requested_flux_attention_backend() -> str:
    env_value = os.getenv(_FLUX_ATTENTION_ENV_KEY)
    if env_value:
        return _normalize_flux_attention_backend(env_value)

    argv = [str(arg or "").strip().lower() for arg in sys.argv[1:]]
    for arg in argv:
        if arg.startswith("--flux-attention-backend="):
            return _normalize_flux_attention_backend(arg.split("=", 1)[1])
    if "--flux-attention-xformers-only" in argv:
        return "xformers_only"
    if "--flux-attention-xformers" in argv:
        return "xformers"
    if "--flux-attention-sdpa" in argv:
        return "sdpa"
    if "--flux-attention-auto" in argv:
        return "auto"
    return "auto"


def _flux_attention_cache_key(device: torch.device | None) -> tuple[str, str]:
    requested = _resolve_requested_flux_attention_backend()
    return requested, str(device or "")


def _log_flux_attention_selection(
    cache_key: tuple[str, str],
    *,
    selected: str,
    reason: str | None = None,
) -> None:
    if cache_key in _FLUX_ATTENTION_LOGGED_KEYS:
        return
    _FLUX_ATTENTION_LOGGED_KEYS.add(cache_key)
    requested, device_label = cache_key
    if reason is None:
        logger.info(
            "[Flux Attention] policy=%s selected=%s device=%s",
            requested,
            selected,
            device_label,
        )
    else:
        logger.info(
            "[Flux Attention] policy=%s selected=%s device=%s reason=%s",
            requested,
            selected,
            device_label,
            reason,
        )


def _select_flux_attention_backend(
    q: Tensor,
    mask,
) -> str:
    cache_key = _flux_attention_cache_key(q.device)
    cached = _FLUX_ATTENTION_RUNTIME_BACKENDS.get(cache_key)
    if cached is not None:
        return cached

    requested, _device_label = cache_key

    if requested in {"auto", "sdpa"}:
        selected = "sdpa"
        reason = "sdpa_default" if requested == "auto" else "forced_sdpa"
    elif requested in {"xformers", "xformers_only"}:
        if mask is not None:
            if requested == "xformers_only":
                raise RuntimeError("Flux xformers-only backend does not support attention masks.")
            selected = "sdpa"
            reason = "mask_requires_sdpa_fallback"
        elif not ldm_patched.modules.model_management.xformers_enabled():
            if requested == "xformers_only":
                raise RuntimeError("Flux xformers-only backend requested but xformers is unavailable.")
            selected = "sdpa"
            reason = "xformers_unavailable"
        elif xformers is None:
            if requested == "xformers_only":
                raise RuntimeError("Flux xformers-only backend requested but xformers could not be imported.")
            selected = "sdpa"
            reason = "xformers_import_failed"
        else:
            selected = "xformers"
            reason = "requested_xformers"
    else:
        selected = "sdpa"
        reason = "unexpected_policy_fallback"

    _FLUX_ATTENTION_RUNTIME_BACKENDS[cache_key] = selected
    _log_flux_attention_selection(cache_key, selected=selected, reason=reason)
    return selected


def _flux_attention_sdpa(q: Tensor, k: Tensor, v: Tensor, mask=None) -> Tensor:
    out = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    )
    return rearrange(out, "b h n d -> b n (h d)")


def _flux_attention_xformers(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    if xformers is None:
        raise RuntimeError("xformers backend requested but xformers is not importable.")
    if _BROKEN_XFORMERS and (int(q.shape[0]) * int(q.shape[1])) > 65535:
        raise RuntimeError("xformers version is known-broken for the current batch*heads size.")

    # xformers expects [B, tokens, heads, dim] for the memory-efficient path,
    # while Flux keeps the tensors in [B, heads, tokens, dim].
    q_x = q.transpose(1, 2).contiguous()
    k_x = k.transpose(1, 2).contiguous()
    v_x = v.transpose(1, 2).contiguous()
    out = xformers.ops.memory_efficient_attention(q_x, k_x, v_x, attn_bias=None)
    return rearrange(out, "b n h d -> b n (h d)")


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask=None) -> Tensor:
    q_shape = q.shape
    k_shape = k.shape

    if pe is not None:
        q = q.to(dtype=pe.dtype).reshape(*q.shape[:-1], -1, 1, 2)
        k = k.to(dtype=pe.dtype).reshape(*k.shape[:-1], -1, 1, 2)
        q = (pe[..., 0] * q[..., 0] + pe[..., 1] * q[..., 1]).reshape(*q_shape).type_as(v)
        k = (pe[..., 0] * k[..., 0] + pe[..., 1] * k[..., 1]).reshape(*k_shape).type_as(v)

    selected = _select_flux_attention_backend(q, mask)
    if selected == "xformers":
        try:
            return _flux_attention_xformers(q, k, v)
        except (ImportError, AttributeError, NotImplementedError, RuntimeError) as exc:
            cache_key = _flux_attention_cache_key(q.device)
            requested, _device_label = cache_key
            if requested == "xformers_only":
                raise
            _FLUX_ATTENTION_RUNTIME_BACKENDS[cache_key] = "sdpa"
            logger.warning(
                "[Flux Attention] xformers failed; falling back to SDPA for this process. error=%s: %s",
                type(exc).__name__,
                exc,
            )
            _log_flux_attention_selection(cache_key, selected="sdpa", reason="xformers_runtime_fallback")

    return _flux_attention_sdpa(q, k, v, mask=mask)


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    if (
        ldm_patched.modules.model_management.is_device_mps(pos.device)
        or ldm_patched.modules.model_management.is_intel_xpu()
        or ldm_patched.modules.model_management.is_directml_enabled()
    ):
        device = torch.device("cpu")
    else:
        device = pos.device

    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor):
    xq_ = xq.to(dtype=freqs_cis.dtype).reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.to(dtype=freqs_cis.dtype).reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)
