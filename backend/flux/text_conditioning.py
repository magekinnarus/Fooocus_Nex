from __future__ import annotations

import gc
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import T5TokenizerFast

import backend.patching as patching
import ldm_patched.modules.model_management as model_management
import ldm_patched.modules.ops as base_ops
import ldm_patched.modules.sd1_clip as sd1_clip
import ldm_patched.modules.utils as comfy_utils
from backend import resources
from backend.flux.flux_fill_pipeline import FluxEmptyConditioning, save_flux_empty_conditioning_cache
from backend.gguf.loader import gguf_clip_loader
from backend.gguf.ops import GGMLOps
from backend.gguf.patcher import GGUFModelPatcher
from ldm_patched.ldm.modules.attention import optimized_attention_for_device

logger = logging.getLogger(__name__)

_T5_FIXED_LENGTH = 256
_CLIP_L_KEY = "text_model.encoder.layers.1.mlp.fc1.weight"
_T5_KEY = "encoder.block.23.layer.1.DenseReluDense.wi_1.weight"
_T5_KEY_OLD = "encoder.block.23.layer.1.DenseReluDense.wi.weight"
_RESIDENT_ENCODER_CACHE: dict[tuple[str, str, str | None, str], "FluxPromptTextEncoder"] = {}


def flux_t5_tokenizer_path() -> Path:
    return Path(__file__).with_name("t5_tokenizer")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _pick_t5_ops(model_options: dict[str, Any] | None) -> Any:
    custom_ops = (model_options or {}).get("custom_operations")
    return custom_ops if custom_ops is not None else base_ops.manual_cast


def _normalize_t5_loader_policy(policy: str | None, *, t5_path: str | Path | None = None) -> str:
    if policy is None or str(policy).strip() == "":
        if t5_path is not None and Path(t5_path).suffix.lower() == ".safetensors":
            return "stream_safetensors_runtime"
        return "eager"
    value = str(policy).strip().lower().replace("-", "_").replace(" ", "_")
    if value not in {"eager", "stream_safetensors_runtime"}:
        raise ValueError(f"Unsupported T5 loader policy: {policy!r}.")
    return value


class FixedLengthT5Tokenizer:
    def __init__(
        self,
        tokenizer_path: str | Path | None = None,
        *,
        fixed_length: int = _T5_FIXED_LENGTH,
    ) -> None:
        tokenizer_path = Path(tokenizer_path) if tokenizer_path is not None else flux_t5_tokenizer_path()
        self.tokenizer = T5TokenizerFast.from_pretrained(str(tokenizer_path))
        self.fixed_length = int(fixed_length)
        empty = self.tokenizer("")["input_ids"]
        self.start_token = None
        self.end_token = int(empty[0])
        self.pad_token = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False, **_: Any) -> list[list[tuple[int, float] | tuple[int, float, int]]]:
        parsed_weights = sd1_clip.token_weights(sd1_clip.escape_important(text), 1.0)
        token_stream: list[tuple[int, float, int]] = []
        word_id = 1

        for weighted_segment, weight in parsed_weights:
            words = sd1_clip.unescape_important(weighted_segment).replace("\n", " ").split(" ")
            for word in (piece for piece in words if piece != ""):
                token_ids = self.tokenizer(word)["input_ids"][:-1]
                for token_id in token_ids:
                    if len(token_stream) >= self.fixed_length - 1:
                        break
                    token_stream.append((int(token_id), float(weight), word_id))
                if len(token_stream) >= self.fixed_length - 1:
                    break
                word_id += 1
            if len(token_stream) >= self.fixed_length - 1:
                break

        token_stream.append((self.end_token, 1.0, 0))
        while len(token_stream) < self.fixed_length:
            token_stream.append((self.pad_token, 1.0, 0))

        if return_word_ids:
            return [token_stream]
        return [[(token_id, weight) for token_id, weight, _ in token_stream]]


class T5LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, *, dtype=None, device=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = f"{prefix}weight"
        value = state_dict.get(key)
        if value is None:
            missing_keys.append(key)
            return
        if hasattr(value, "load") and callable(value.load):
            value = value.load()
        if not isinstance(value, torch.Tensor):
            error_msgs.append(f"{key} expected tensor-like value, got {type(value).__name__}.")
            return
        self.weight.data.copy_(value.to(device=self.weight.device, dtype=self.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return base_ops.cast_to_input(self.weight, x) * x


_ACTIVATIONS = {
    "gelu_pytorch_tanh": lambda tensor: torch.nn.functional.gelu(tensor, approximate="tanh"),
    "relu": torch.nn.functional.relu,
}


class T5DenseActDense(torch.nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, ff_activation: str, *, dtype=None, device=None, operations=base_ops.manual_cast):
        super().__init__()
        self.wi = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = operations.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.act = _ACTIVATIONS[ff_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.act(self.wi(x)))


class T5DenseGatedActDense(torch.nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, ff_activation: str, *, dtype=None, device=None, operations=base_ops.manual_cast):
        super().__init__()
        self.wi_0 = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wi_1 = operations.Linear(model_dim, ff_dim, bias=False, dtype=dtype, device=device)
        self.wo = operations.Linear(ff_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.act = _ACTIVATIONS[ff_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wo(self.act(self.wi_0(x)) * self.wi_1(x))


class T5LayerFF(torch.nn.Module):
    def __init__(self, model_dim: int, ff_dim: int, ff_activation: str, gated_act: bool, *, dtype=None, device=None, operations=base_ops.manual_cast):
        super().__init__()
        dense_cls = T5DenseGatedActDense if gated_act else T5DenseActDense
        self.DenseReluDense = dense_cls(model_dim, ff_dim, ff_activation, dtype=dtype, device=device, operations=operations)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.DenseReluDense(self.layer_norm(x))


class T5Attention(torch.nn.Module):
    def __init__(self, model_dim: int, inner_dim: int, num_heads: int, relative_attention_bias: bool, *, dtype=None, device=None, operations=base_ops.manual_cast):
        super().__init__()
        self.q = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.k = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.v = operations.Linear(model_dim, inner_dim, bias=False, dtype=dtype, device=device)
        self.o = operations.Linear(inner_dim, model_dim, bias=False, dtype=dtype, device=device)
        self.num_heads = num_heads
        self.relative_attention_bias = None
        if relative_attention_bias:
            self.relative_attention_num_buckets = 32
            self.relative_attention_max_distance = 128
            self.relative_attention_bias = operations.Embedding(self.relative_attention_num_buckets, self.num_heads, device=device, dtype=dtype)

    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor, *, bidirectional: bool = True, num_buckets: int = 32, max_distance: int = 128) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / torch.log(torch.tensor(max_distance / max_exact, device=relative_position.device))
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1))
        return relative_buckets + torch.where(is_small, relative_position, relative_position_if_large)

    def compute_bias(self, query_length: int, key_length: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        buckets = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(buckets, out_dtype=dtype)
        return values.permute(2, 0, 1).unsqueeze(0).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        past_bias: torch.Tensor | None = None,
        optimized_attention=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        if self.relative_attention_bias is not None:
            past_bias = self.compute_bias(x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)

        if past_bias is not None:
            mask = mask + past_bias if mask is not None else past_bias

        out = optimized_attention(q, k * ((k.shape[-1] / self.num_heads) ** 0.5), v, self.num_heads, mask)
        return self.o(out), past_bias


class T5LayerSelfAttention(torch.nn.Module):
    def __init__(self, model_dim: int, inner_dim: int, num_heads: int, relative_attention_bias: bool, *, dtype=None, device=None, operations=base_ops.manual_cast):
        super().__init__()
        self.SelfAttention = T5Attention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype=dtype, device=device, operations=operations)
        self.layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        past_bias: torch.Tensor | None = None,
        optimized_attention=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        output, past_bias = self.SelfAttention(self.layer_norm(x), mask=mask, past_bias=past_bias, optimized_attention=optimized_attention)
        return x + output, past_bias


class T5Block(torch.nn.Module):
    def __init__(
        self,
        model_dim: int,
        inner_dim: int,
        ff_dim: int,
        ff_activation: str,
        gated_act: bool,
        num_heads: int,
        relative_attention_bias: bool,
        *,
        dtype=None,
        device=None,
        operations=base_ops.manual_cast,
    ):
        super().__init__()
        self.layer = torch.nn.ModuleList(
            [
                T5LayerSelfAttention(model_dim, inner_dim, num_heads, relative_attention_bias, dtype=dtype, device=device, operations=operations),
                T5LayerFF(model_dim, ff_dim, ff_activation, gated_act, dtype=dtype, device=device, operations=operations),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        past_bias: torch.Tensor | None = None,
        optimized_attention=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, past_bias = self.layer[0](x, mask=mask, past_bias=past_bias, optimized_attention=optimized_attention)
        return self.layer[1](x), past_bias


class T5Stack(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        inner_dim: int,
        ff_dim: int,
        ff_activation: str,
        gated_act: bool,
        num_heads: int,
        relative_attention: bool,
        *,
        dtype=None,
        device=None,
        operations=base_ops.manual_cast,
    ):
        super().__init__()
        self.block = torch.nn.ModuleList(
            [
                T5Block(
                    model_dim,
                    inner_dim,
                    ff_dim,
                    ff_activation,
                    gated_act,
                    num_heads,
                    relative_attention_bias=((not relative_attention) or (index == 0)),
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
                for index in range(num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(model_dim, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        mask = None
        if attention_mask is not None:
            mask = 1.0 - attention_mask.to(x.dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])).expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )
            mask = mask.masked_fill(mask.to(torch.bool), -torch.finfo(x.dtype).max)

        optimized_attention = optimized_attention_for_device(x.device, mask=attention_mask is not None, small_input=True)
        past_bias = None
        for block in self.block:
            x, past_bias = block(x, mask=mask, past_bias=past_bias, optimized_attention=optimized_attention)
        return self.final_layer_norm(x), None


class T5(torch.nn.Module):
    def __init__(self, config_dict: dict[str, Any], dtype, device, operations) -> None:
        super().__init__()
        model_dim = config_dict["d_model"]
        inner_dim = config_dict["d_kv"] * config_dict["num_heads"]
        self.encoder = T5Stack(
            config_dict["num_layers"],
            model_dim,
            inner_dim,
            config_dict["d_ff"],
            config_dict["dense_act_fn"],
            config_dict["is_gated_act"],
            config_dict["num_heads"],
            config_dict["model_type"] != "umt5",
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.shared = operations.Embedding(config_dict["vocab_size"], model_dim, device=device, dtype=dtype)
        self.dtype = dtype

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, embeddings) -> None:
        self.shared = embeddings

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, embeds: torch.Tensor | None = None, **kwargs):
        if input_ids is None:
            x = embeds
        else:
            x = self.shared(input_ids, out_dtype=kwargs.get("dtype", torch.float32))
        if self.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            x = torch.nan_to_num(x)
        return self.encoder(x, attention_mask=attention_mask)


class T5XXLTextEncoder(torch.nn.Module, sd1_clip.ClipTokenWeightEncoder):
    def __init__(self, *, device="cpu", dtype=None, model_options: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = _load_json(Path(__file__).with_name("t5_config_xxl.json"))
        operations = _pick_t5_ops(model_options)
        self.transformer = T5(config, dtype, device, operations)
        self.special_tokens = {"end": 1, "pad": 0}

    def set_clip_options(self, options: dict[str, Any]) -> None:
        return None

    def reset_clip_options(self) -> None:
        return None

    def encode(self, tokens):
        embedding_weight = self.transformer.get_input_embeddings().weight
        device = getattr(embedding_weight, "device", torch.device("cpu"))
        if not isinstance(device, torch.device):
            device = torch.device(device)
        input_ids = torch.LongTensor(tokens).to(device)
        attention_mask = (input_ids != self.special_tokens["pad"]).long()
        encoded, _ = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return encoded.float(), None

    def load_sd(self, sd):
        return self.transformer.load_state_dict(sd, strict=False)


class FluxClipModel(torch.nn.Module):
    def __init__(self, *, dtype_t5=None, device="cpu", dtype=None, model_options: dict[str, Any] | None = None):
        super().__init__()
        model_options = model_options or {}
        dtype_t5 = model_management.pick_weight_dtype(dtype_t5, dtype, device)
        self.clip_l = sd1_clip.SDClipModel(device=device, dtype=dtype)
        self.t5xxl = T5XXLTextEncoder(device=device, dtype=dtype_t5, model_options=model_options)
        self.dtypes = {dtype, dtype_t5}

    def set_clip_options(self, options: dict[str, Any]) -> None:
        return None

    def reset_clip_options(self) -> None:
        return None

    def clip_layer(self, layer_idx: int) -> None:
        self.clip_l.clip_layer(layer_idx)

    def reset_clip_layer(self) -> None:
        self.clip_l.reset_clip_layer()

    def encode_token_weights(self, token_weight_pairs):
        t5_out, _ = self.t5xxl.encode_token_weights(token_weight_pairs["t5xxl"])
        _, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs["l"])
        return t5_out, l_pooled

    def load_sd(self, sd):
        if _CLIP_L_KEY in sd:
            return self.clip_l.load_sd(sd)
        return self.t5xxl.load_sd(sd)


class FluxTokenizer:
    def __init__(self, embedding_directory=None):
        self.clip_l = sd1_clip.SDTokenizer(embedding_directory=embedding_directory)
        self.t5xxl = FixedLengthT5Tokenizer()

    def tokenize_with_weights(self, text: str, return_word_ids: bool = False, **kwargs):
        return {
            "l": self.clip_l.tokenize_with_weights(text, return_word_ids),
            "t5xxl": self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs),
        }

    def state_dict(self) -> dict[str, Any]:
        return {}


@dataclass
class FluxPromptTextEncoder:
    cond_stage_model: FluxClipModel
    tokenizer: FluxTokenizer
    patcher: Any

    def tokenize(self, text: str):
        return self.tokenizer.tokenize_with_weights(text)

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenize(text)
        load_device = getattr(self.patcher, "load_device", None)
        if getattr(load_device, "type", None) != "cpu":
            resources.load_models_gpu([self.patcher], force_full_load=True)
        try:
            with torch.inference_mode():
                cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
            return cond, pooled
        finally:
            try:
                resources.eject_model(self.patcher)
            except Exception:
                detach = getattr(self.patcher, "detach", None)
                if callable(detach):
                    detach()


def _resident_encoder_key(
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    t5_loader_policy: str | None = None,
) -> tuple[str, str, str | None, str]:
    return (
        str(Path(clip_l_path)),
        str(Path(t5_path)),
        str(Path(embedding_directory)) if embedding_directory is not None else None,
        _normalize_t5_loader_policy(t5_loader_policy, t5_path=t5_path),
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


def _load_text_encoder_state_dict(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.suffix.lower() == ".gguf":
        return gguf_clip_loader(str(path)), {"custom_operations": GGMLOps}
    return comfy_utils.load_torch_file(str(path), safe_load=True), {}


def _normalize_checkpoint_dtype(value: Any) -> torch.dtype | None:
    dtype = getattr(value, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None and isinstance(value, torch.dtype):
        return value

    dtype_text = str(dtype if dtype is not None else value).strip()
    mapping = {
        "F16": torch.float16,
        "F32": torch.float32,
        "BF16": torch.bfloat16,
        "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
        "F8_E4M3FN": getattr(torch, "float8_e4m3fn", None),
        "F8_E5M2": getattr(torch, "float8_e5m2", None),
        "torch.float16": torch.float16,
        "torch.float32": torch.float32,
        "torch.bfloat16": torch.bfloat16,
        "float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "float8_e5m2": getattr(torch, "float8_e5m2", None),
        "torch.float8_e4m3fn": getattr(torch, "float8_e4m3fn", None),
        "torch.float8_e5m2": getattr(torch, "float8_e5m2", None),
    }
    return mapping.get(dtype_text)


def _detect_t5_dtype(state_dict: dict[str, Any]) -> torch.dtype | None:
    for key in ("encoder.final_layer_norm.weight", _T5_KEY, _T5_KEY_OLD):
        tensor = state_dict.get(key)
        if tensor is not None:
            detected = _normalize_checkpoint_dtype(tensor)
            if detected is not None:
                return detected
    return None


def _map_t5_source_key(source_key: str) -> str:
    if source_key == "encoder.embed_tokens.weight":
        return "shared.weight"
    return source_key


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
        if hasattr(value, "load") and callable(value.load):
            value = value.load()
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor-like lazy weight, got {type(value).__name__}.")
        if dtype is None:
            return value.to(device=device)
        return value.to(device=device, dtype=dtype)


class LazySafetensorsOps(base_ops.manual_cast):
    class Linear(LazySafetensorsLayer, base_ops.manual_cast.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            torch.nn.Module.__init__(self)
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.bias = None

        def forward(self, input):
            weight = self._materialize(self.weight, device=input.device, dtype=input.dtype)
            bias = self._materialize(self.bias, device=input.device, dtype=input.dtype) if self.bias is not None else None
            return torch.nn.functional.linear(input, weight, bias)

    class Embedding(LazySafetensorsLayer, base_ops.manual_cast.Embedding):
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
            weight_dtype = out_dtype if out_dtype is not None else None
            weight = self._materialize(self.weight, device=input.device, dtype=weight_dtype)
            out = torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            if out_dtype is not None:
                out = out.to(dtype=out_dtype)
            return out


def _build_lazy_t5_state_dict(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from backend.cpu_compiler import SafeOpenHeaderOnly

    header = SafeOpenHeaderOnly(str(path))
    lazy_state_dict: dict[str, Any] = {}
    duplicate_source_keys: list[str] = []
    for source_key, value in header.items():
        target_key = _map_t5_source_key(source_key)
        if target_key in lazy_state_dict:
            duplicate_source_keys.append(source_key)
            continue
        lazy_state_dict[target_key] = value
    return lazy_state_dict, {
        "custom_operations": LazySafetensorsOps,
        "lazy_safetensors_runtime": True,
        "lazy_duplicate_source_keys": duplicate_source_keys,
    }


def load_flux_prompt_text_encoder(
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    t5_loader_policy: str | None = None,
) -> FluxPromptTextEncoder:
    clip_l_path = Path(clip_l_path)
    t5_path = Path(t5_path)
    clip_l_sd, clip_l_options = _load_text_encoder_state_dict(clip_l_path)
    t5_loader_policy = _normalize_t5_loader_policy(t5_loader_policy, t5_path=t5_path)
    if t5_loader_policy == "stream_safetensors_runtime":
        if t5_path.suffix.lower() != ".safetensors":
            raise ValueError("stream_safetensors_runtime requires a .safetensors T5 checkpoint.")
        t5_sd, t5_options = _build_lazy_t5_state_dict(t5_path)
    else:
        t5_sd, t5_options = _load_text_encoder_state_dict(t5_path)

    # Flux prompt conditioning is intentionally CPU-scoped so Colab/local policy
    # decisions are about RAM fit instead of opportunistic GPU residency.
    load_device = torch.device("cpu")
    offload_device = torch.device("cpu")
    dtype = model_management.text_encoder_dtype(load_device)
    model_options = {}
    model_options.update(clip_l_options)
    model_options.update(t5_options)
    initial_device = torch.device("cpu")
    model_options["initial_device"] = initial_device

    cond_stage_model = FluxClipModel(
        dtype_t5=_detect_t5_dtype(t5_sd),
        device=initial_device,
        dtype=dtype,
        model_options=model_options,
    )
    tokenizer = FluxTokenizer(embedding_directory=embedding_directory)
    patcher_cls = GGUFModelPatcher if model_options.get("custom_operations") is GGMLOps else patching.NexModelPatcher
    patcher = patcher_cls(cond_stage_model, load_device=load_device, offload_device=offload_device)

    missing, unexpected = cond_stage_model.load_sd(clip_l_sd)
    if missing:
        logger.debug("Flux CLIP-L missing keys: %s", missing)
    if unexpected:
        logger.debug("Flux CLIP-L unexpected keys: %s", unexpected)

    missing, unexpected = cond_stage_model.load_sd(t5_sd)
    if missing:
        logger.debug("Flux T5 missing keys: %s", missing)
    if unexpected:
        logger.debug("Flux T5 unexpected keys: %s", unexpected)

    encoder = FluxPromptTextEncoder(cond_stage_model=cond_stage_model, tokenizer=tokenizer, patcher=patcher)
    setattr(
        encoder,
        "_nex_load_metadata",
        {
            "clip_l_path": str(clip_l_path),
            "t5_path": str(t5_path),
            "t5_loader_policy": t5_loader_policy,
            "t5_detected_dtype": str(_detect_t5_dtype(t5_sd)) if _detect_t5_dtype(t5_sd) is not None else None,
            "t5_source_kind": "safetensors_lazy_runtime" if t5_loader_policy == "stream_safetensors_runtime" else "eager_state_dict",
            "t5_full_state_dict_materialized": t5_loader_policy != "stream_safetensors_runtime",
            "t5_stream_runtime": t5_loader_policy == "stream_safetensors_runtime",
            "t5_lazy_runtime": t5_loader_policy == "stream_safetensors_runtime",
            "t5_lazy_duplicate_source_keys": list(t5_options.get("lazy_duplicate_source_keys", [])),
        },
    )
    return encoder


def get_flux_prompt_text_encoder(
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    keep_resident: bool = False,
    t5_loader_policy: str | None = None,
) -> FluxPromptTextEncoder:
    t5_loader_policy = _normalize_t5_loader_policy(t5_loader_policy, t5_path=t5_path)
    if not keep_resident:
        return load_flux_prompt_text_encoder(
            clip_l_path=clip_l_path,
            t5_path=t5_path,
            embedding_directory=embedding_directory,
            t5_loader_policy=t5_loader_policy,
        )

    key = _resident_encoder_key(
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        t5_loader_policy=t5_loader_policy,
    )
    cached = _RESIDENT_ENCODER_CACHE.get(key)
    if cached is not None:
        return cached

    clear_flux_prompt_text_encoder_cache()
    encoder = load_flux_prompt_text_encoder(
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        t5_loader_policy=t5_loader_policy,
    )
    _RESIDENT_ENCODER_CACHE[key] = encoder
    return encoder


def encode_flux_prompt_conditioning(
    prompt: str,
    *,
    clip_l_path: str | Path,
    t5_path: str | Path,
    embedding_directory: str | Path | None = None,
    keep_resident: bool = False,
    t5_loader_policy: str | None = None,
) -> FluxEmptyConditioning:
    prompt_text = str(prompt or "").strip()
    if prompt_text == "":
        raise ValueError("Flux prompt conditioning requires a non-empty prompt.")

    encoder = get_flux_prompt_text_encoder(
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        keep_resident=keep_resident,
        t5_loader_policy=t5_loader_policy,
    )
    load_metadata = dict(getattr(encoder, "_nex_load_metadata", {}) or {})
    try:
        cross_attn, pooled_output = encoder.encode(prompt_text)
        return FluxEmptyConditioning(
            cross_attn=cross_attn.to(device="cpu"),
            pooled_output=pooled_output.to(device="cpu"),
            metadata={
                "prompt": prompt_text,
                "clip_l_path": str(clip_l_path),
                "t5_path": str(t5_path),
                "t5_format": "gguf" if str(t5_path).lower().endswith(".gguf") else "safetensors",
                "generator": "backend/flux/text_conditioning.py",
                "conditioning_kind": "prompt",
                "transport": "memory",
                "text_encoder_resident": bool(keep_resident),
                "t5_loader_policy": _normalize_t5_loader_policy(t5_loader_policy, t5_path=t5_path),
                "loader_metadata": load_metadata,
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
    keep_resident: bool = False,
    t5_loader_policy: str | None = None,
) -> FluxEmptyConditioning:
    conditioning = encode_flux_prompt_conditioning(
        prompt,
        clip_l_path=clip_l_path,
        t5_path=t5_path,
        embedding_directory=embedding_directory,
        keep_resident=keep_resident,
        t5_loader_policy=t5_loader_policy,
    )
    return save_flux_empty_conditioning_cache(
        output_path,
        cross_attn=conditioning.cross_attn,
        pooled_output=conditioning.pooled_output,
        metadata=dict(conditioning.metadata, transport="pt_cache"),
    )
