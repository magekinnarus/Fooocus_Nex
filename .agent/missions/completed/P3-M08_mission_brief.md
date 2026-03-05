# Mission Brief: P3-M08 — CLIP Pipeline Extraction (SD1.5 → SDXL)
**ID:** P3-M08
**Phase:** 3
**Date Issued:** 2026-02-18
**Status:** Draft
**Depends On:** P3-M07 (lessons learned from SD1.5 debugging)
**Work List:** `.agent/missions/active/P3-M08_work_list.md`

## Required Reading
- `.agent/summaries/01_Project_Vision.md`
- `.agent/summaries/02_Architecture_and_Strategy.md`
- `.agent/summaries/03_Roadmap.md`
- `.agent/rules/01_Global_Context_Rules.md`
- `.agent/rules/02_Documentation_and_Mission_Standards.md`
- `.agent/reference/P3_ldm_patched_clip_extraction_map.md` ← **Primary technical reference**
- `.agent/reference/P3_technical_discovery_precision.md`

## Objective

Extract the CLIP text encoding pipeline from `ldm_patched` into a **self-contained** `backend/clip.py` module with **zero ldm_patched imports**. This unblocks SD1.5 inference (currently broken due to invisible precision/normalization traps in the entangled ldm_patched chain) and prepares for SDXL CLIP isolation.

**Why this mission exists:** Multiple sessions of debugging SD1.5 inference revealed that the thin-wrapper approach (backend modules calling into ldm_patched) breaks down because ldm_patched embeds hidden precision casting, layer normalization, and model lifecycle assumptions across 6 interconnected files. The only sustainable fix is surgical extraction of the ~300 lines of actual CLIP logic.

### Core Design Decisions (Director-approved)
1. **Unified Key Normalization:** One normalization function per CLIP variant (`normalize_clip_l_keys`, `normalize_clip_g_keys`) that auto-detects and strips any known prefix. Both bundled-checkpoint and standalone-file loading use the **same code path**. No separate loading logic.
2. **HF Naming Convention:** Model class parameter names use HuggingFace-standard naming (not ComfyUI's internal `clip_l.*`/`clip_g.*` convention). Prefix stripping happens once at the loading boundary.
3. **Optional Projection:** `text_projection` and `logit_scale` are optional, controlled by `use_projection` flag — not universally defined and silently ignored like ComfyUI. This is Flux-forward: Flux CLIP-L needs projection (like SDXL CLIP-G), while SD1.5 and SDXL CLIP-L do not.

## Scope

### In Scope
- **`backend/clip.py` [NEW]** — self-contained CLIP tokenizer + encoder module
  - SD1.5 CLIP-L support (Phase A / W01)
  - SDXL CLIP-L + CLIP-G support (Phase B / W02)
  - Explicit FP32 precision chain (no reliance on ops.manual_cast)
  - Inline attention with FP32 safety guard
  - Token weighting (prompt emphasis with parentheses)
  - Unified key normalization (same load path for bundled checkpoints and standalone files)
  - Optional `text_projection`/`logit_scale` via `use_projection` flag (Flux-forward)
- **`app.py` [MODIFY]** — update to use `backend/clip.py` instead of ldm_patched CLIP
- **`backend/loader.py` [MODIFY]** — update SD1.5/SDXL load functions to use extracted CLIP
- **`tools/model_extractor.py` [MODIFY]** — update clip extraction to output Nex bundled format (HF keys with `clip_l.`/`clip_g.` namespace, OpenCLIP→HF transform at extraction time)
- **Verification scripts** — parity check between ldm_patched CLIP and extracted CLIP

### Out of Scope
- UNet extraction (already works via model_patcher)
- VAE extraction (already works via backend/decode.py)
- Textual inversion embeddings (future mission)
- LoRA CLIP patching (future mission, P3-M08 scope)
- model_patcher refactoring

## Reference Files
- `Fooocus_Nex/ldm_patched/modules/clip_model.py` — source for CLIPTextModel_ architecture (~80 lines needed)
- `Fooocus_Nex/ldm_patched/modules/sd1_clip.py` — source for SDTokenizer, SDClipModel, token weighting (~200 lines needed)
- `Fooocus_Nex/ldm_patched/modules/ops.py` — source for cast_bias_weight logic (~40 lines needed)
- `Fooocus_Nex/ldm_patched/ldm/modules/attention.py` — source for FP32 attention guard (~20 lines needed)
- `Fooocus_Nex/ldm_patched/modules/sd1_clip_config.json` — CLIP-L model config
- `Fooocus_Nex/ldm_patched/modules/clip_config_bigg.json` — CLIP-G model config
- `.agent/reference/P3_ldm_patched_clip_extraction_map.md` — line-by-line extraction guide

## Constraints
- **Zero ldm_patched imports** in the final `backend/clip.py`
- Dependencies limited to: `torch`, `transformers.CLIPTokenizer`, `backend.resources`
- Must preserve the FP32 precision chain (embeddings → encoder → output all FP32 compute)
- Must match ldm_patched CLIP output tensor values exactly (parity test)
- Tokenizer data files (`sd1_tokenizer/`) can be copied to `backend/` or referenced from ldm_patched
- **Unified loading:** Same `normalize_clip_*_keys()` function handles both bundled checkpoints and standalone files — no separate code paths
- **No universal projection:** `text_projection`/`logit_scale` only registered when `use_projection=True` — avoids ComfyUI's ghost-parameter problem

## Deliverables
- [ ] **`backend/clip.py`** — self-contained CLIP module with SD1.5 + SDXL support
- [ ] **Updated `backend/loader.py`** — uses `backend/clip.py` for CLIP instantiation
- [ ] **Updated `app.py`** — uses extracted CLIP, generates correct SD1.5 image
- [ ] **Parity test script** — proves extracted CLIP produces identical outputs to ldm_patched CLIP
- [ ] **Proof of work** — SD1.5 generated image (recognizable, not noise/blue/black)

## Success Criteria
1. `backend/clip.py` contains zero imports from `ldm_patched`
2. Parity test: extracted CLIP output matches ldm_patched CLIP output (max absolute diff < 1e-5)
3. SD1.5 inference produces a recognizable image using extracted CLIP
4. No NaN or Inf values in conditioning tensors
5. No regression in SDXL inference after W02 integration
6. The Director can read `backend/clip.py` top-to-bottom and understand the full text encoding pipeline
7. Same CLIP weights loaded from bundled checkpoint and standalone file produce identical model state dicts
8. `text_projection`/`logit_scale` are only present in models where `use_projection=True`

## Work Orders
To be registered in `P3-M08_work_list.md` by CM Role1:
- `P3-M08-W01` — Extract SD1.5 CLIP pipeline into `backend/clip.py` with unified key normalization, parity test, fix `app.py` for SD1.5 inference
- `P3-M08-W02` — Extend `backend/clip.py` for SDXL (CLIP-G key normalization + OpenCLIP transforms, dual encoding, `use_projection=True` for CLIP-G), parity test
- `P3-M08-W03` — Integration: update `backend/loader.py` and `app.py` to use unified loading path, full inference validation for both SD1.5 and SDXL

## Notes
- This mission was triggered by the multi-session SD1.5 debugging that revealed 3 specific traps in ldm_patched: (1) ghost `text_projection`/`logit_scale` layers, (2) invisible `ops.manual_cast` casting chain, (3) `layer_norm_hidden_state` confusion with Clip Skip. Details in `.agent/reference/P3_technical_discovery_precision.md`.
- The extraction map in `.agent/reference/P3_ldm_patched_clip_extraction_map.md` provides line-by-line guidance on what to copy, modify, and skip from each ldm_patched file.
- `clip_model.py` is 90% clean PyTorch — the only impurity is one import from `attention.py` that needs inlining.
- SDXL CLIP (W02) is architecturally identical to SD1.5 CLIP (same `CLIPTextModel_` class, different config). The extraction generalizes naturally.
- `model_patcher.py` is NOT being extracted in this mission. The CLIP patcher wrapper in `backend/loader.py` will continue to use `model_patcher.ModelPatcher` for device management. CLIP model weights themselves are managed by the new module.
