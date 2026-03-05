# Decision Record: Nex Module Design Principles

**ID:** DR-001
**Date:** 2026-02-14
**Status:** Proposed (Pending Director Approval)
**Triggered By:** P3-M02 mission findings — the "Entanglement Trap"

---

## Context

P3-M02 proved that wrapping `ldm_patched` is possible but painful. The friction came from specific anti-patterns in the ComfyUI-derived code. Before building more modules, we need to capture *what went wrong* and codify *what our modules should look like* so every future mission starts from clear structural rules, not ad-hoc discovery.

## The Anti-Patterns (What `ldm_patched` Does Wrong)

These are the specific entanglement patterns observed during P3-M02. Each one has a corresponding design principle.

| # | Anti-Pattern | Example | Pain Caused |
|---|-------------|---------|-------------|
| **AP1** | **Implicit config dicts** | `model_base.SDXL.__init__` expects `model_config.unet_config` with undocumented keys like `adm_in_channels`, `manual_cast_dtype` | Couldn't instantiate models without reverse-engineering what keys were expected |
| **AP2** | **Deep inheritance** | `SDXL` → `BaseModel` → `nn.Module`, `BASE` → many methods that feed back into `model_base` | Changing one class risks breaking 8+ subclasses; understanding behavior requires reading 3 files |
| **AP3** | **Circular imports** | `supported_models_base` imports `model_base`; `model_base` uses patterns that assume `supported_models_base.BASE` | Extracting one module always drags in the other |
| **AP4** | **Global state** | `model_management.py` holds global device/memory state; `model_base.memory_required()` calls into it | Can't test model loading without initializing global memory manager |
| **AP5** | **Multi-model dispatch** | `detect_unet_config` (572 lines), `model_config_from_unet_config` iterating 30+ classes | 95% of detection code is irrelevant for SDXL |
| **AP6** | **Bundled responsibilities** | `ModelPatcher` does device placement AND weight patching AND LoRA AND hooks AND memory tracking (1,398 lines) | Can't use device management without pulling in the entire patching system |
| **AP7** | **Untyped interfaces** | Functions pass `kwargs` dicts through 4+ levels; `extra_conds(**kwargs)` relies on callers knowing which keys exist | Debugging requires tracing call stacks to find who sets `device`, `noise`, `cross_attn` etc. |

## The Design Principles (What Nex Modules Must Follow)

### DP1: Data is Separate from Logic

**Anti-pattern countered:** AP1, AP5

```
backend/defs/sdxl.py    ← Pure data. Configs, prefixes, constants. No imports.
backend/loader.py       ← Pure logic. Functions that use the data.
```

This pattern (already established in P3-M02) must extend to all modules:

| Module | Data File | Logic File |
|--------|-----------|------------|
| Loader | `defs/sdxl.py` (configs, prefixes) | `loader.py` |
| Conditioning | `defs/sdxl.py` (CLIP config, tokenizer params) | `conditioning.py` |
| Sampling | `defs/sdxl.py` (scheduler params, sigma ranges) | `sampling.py` |
| Memory | `defs/device.py` (thresholds, strategies) | `memory.py` |

**Rule:** `defs/` files must have **zero imports** from `ldm_patched` or any other module. They are pure Python data.

### DP2: Functions Over Class Hierarchies

**Anti-pattern countered:** AP2, AP3

Where the task is a transformation (input → output), prefer **plain functions** over classes:

```python
# ✅ Good: function with explicit inputs/outputs
def encode_sdxl_prompt(clip_l_model, clip_g_model, text: str) -> tuple[Tensor, Tensor]:
    """Returns (cross_attn_embedding, pooled_output)"""
    ...

# ❌ Bad: class that hides state and requires specific initialization sequence
class ConditioningPipeline:
    def __init__(self, clip_model, tokenizer, layer_idx=None, ...):
        self.clip = clip_model
        ...
    def encode(self, text):
        ...
```

**When to use classes:** Only when the object has *genuine state that persists across calls* (e.g., a loaded model). Even then, keep them shallow — no inheritance deeper than `nn.Module`.

### DP3: Explicit Inputs, No Kwargs Tunneling

**Anti-pattern countered:** AP7

Every function declares its full signature. No `**kwargs` passed through layers.

```python
# ✅ Good: every parameter visible
def encode_adm(
    pooled_output: Tensor,
    height: int, width: int,
    crop_h: int, crop_w: int,
    target_height: int, target_width: int
) -> Tensor:
    ...

# ❌ Bad: kwargs soup
def encode_adm(**kwargs):
    clip_pooled = sdxl_pooled(kwargs, self.noise_augmentor)
    width = kwargs.get("width", 768)
    ...
```

### DP4: No Global State — Dependency Injection

**Anti-pattern countered:** AP4

Modules must not import or reference `model_management` global state. Device information is passed explicitly.

```python
# ✅ Good: caller decides device
def load_sdxl_unet(source, load_device, offload_device, dtype=None):
    ...

# ❌ Bad: function queries global device state
def load_sdxl_unet(source):
    device = model_management.get_torch_device()  # global state
    ...
```

### DP5: Single-Responsibility Modules

**Anti-pattern countered:** AP6

Each `backend/` file does **one thing**. If a module grows beyond ~300 lines or starts handling two concerns, split it.

```
backend/
├── defs/
│   ├── sdxl.py          ← SDXL constants and configs
│   └── device.py        ← Device/memory thresholds
├── loader.py            ← Model loading only
├── conditioning.py      ← Text encoding only
├── sampling.py          ← Denoising loop only
├── memory.py            ← Device placement only
├── decode.py            ← VAE decode only
└── patching.py          ← LoRA/adapter application only
```

### DP6: Typed Return Values

**Anti-pattern countered:** AP7

Functions return named structures (dataclasses or named tuples), not anonymous dicts or bare tuples.

```python
# ✅ Good: caller knows exactly what they get
@dataclass
class SDXLComponents:
    unet: dict[str, Tensor]
    clip_l: dict[str, Tensor]
    clip_g: dict[str, Tensor]
    vae: dict[str, Tensor]

def extract_sdxl_components(ckpt_path: str) -> SDXLComponents:
    ...

# ❌ Bad: caller has to guess the dict keys
def extract_sdxl_components(ckpt_path: str) -> dict:
    return {"unet": {...}, "clip_l": {...}, ...}
```

### DP7: `ldm_patched` Is Runtime-Only, Never API

**Anti-pattern countered:** AP2, AP3, AP6

`ldm_patched` modules may be used *internally* by `backend/` modules for their runtime functionality (UNet forward pass, CLIP encoding, VAE decode). But `ldm_patched` types must **never appear in `backend/` function signatures**.

```python
# ✅ Good: backend API uses only standard types
def load_sdxl_unet(source: str | dict, ...) -> ModelPatcherWrapper:
    ...

# ❌ Bad: backend API exposes ldm_patched internals
def load_sdxl_unet(source, ...) -> model_patcher.ModelPatcher:
    ...  # leaks ldm_patched type into API
```

**Note:** This principle is aspirational — the current loader returns `ModelPatcher` directly. As decontamination progresses, we wrap or replace these with backend-owned types.

---

## Tracking Template: Dependency Inventory

For every new `backend/` module, the author must fill out this table in the work order:

| `ldm_patched` Dependency | What We Actually Use | SDXL-Essential? | Decontamination Target? |
|--------------------------|---------------------|-----------------|------------------------|
| `model_base.SDXL` | Model class + `encode_adm` | ✅ | Yes (later) |
| `model_patcher.ModelPatcher` | Device mgmt + LoRA patching | ✅ | Yes (later) |
| `supported_models_base.BASE` | `__init__` config setup | ❌ shim only | Yes (next) |
| ... | ... | ... | ... |

This table becomes the input for future decontamination missions — we'll know exactly which `ldm_patched` primitives are used across all modules, and which ones are just dragged in as transitive dead weight.

---

## How This Integrates With Mission Workflow

1. **Mission brief** references this document
2. **Work orders** include the Dependency Inventory table
3. **Work reports** note any violations or exceptions
4. **Mission report** updates the inventory with actual vs. planned deps
5. **Future decontamination missions** use the accumulated inventory as their scope

---

## Module Boundary Map (Living Document)

This section will be updated as each module is built. It defines what each module owns and its inputs/outputs.

### `loader.py` (P3-M02 — Complete)
- **Owns:** Checkpoint splitting, component instantiation, GGUF dispatch
- **Input:** File path or state dict
- **Output:** Wrapped model objects (UNet patcher, CLIP container, VAE container)
- **`ldm_patched` deps:** `model_base.SDXL`, `ModelPatcher`, `BASE`, `SDXLClipModel`, `AutoencoderKL`

### `conditioning.py` (Not yet built)
- **Will own:** Tokenization, dual CLIP encoding, ADM conditioning
- **Input:** Text string + CLIP container
- **Output:** `(cross_attn_embedding: Tensor, pooled_output: Tensor, adm_conditioning: Tensor)`
- **Expected `ldm_patched` deps:** `SDXLClipModel.encode_token_weights`, `Timestep` embedder

### `sampling.py` (Not yet built)
- **Will own:** CFG guidance, k-diffusion loop, scheduler
- **Input:** UNet patcher, conditioning tensors, noise, sigmas
- **Output:** Denoised latent tensor
- **Expected `ldm_patched` deps:** `sampling_function`, `cfg_function`, scheduler functions

### `decode.py` (Not yet built)
- **Will own:** VAE decoding, tiled fallback, memory-aware batching
- **Input:** VAE container, latent tensor
- **Output:** Image tensor (B, H, W, 3) float32 [0,1]
- **Expected `ldm_patched` deps:** `AutoencoderKL.decode`

### `memory.py` (Not yet built)
- **Will own:** Device placement, load/offload scheduling, VRAM estimation
- **Input:** Model patcher, device preferences
- **Output:** Managed placement state
- **Expected `ldm_patched` deps:** Possibly minimal — this may be the first fully decontaminated module

### `patching.py` (Not yet built)
- **Will own:** LoRA application, adapter management
- **Input:** Model patcher, LoRA state dicts
- **Output:** Patched model
- **Expected `ldm_patched` deps:** `calculate_weight`, `ModelPatcher.patch_weight` — highest entanglement
