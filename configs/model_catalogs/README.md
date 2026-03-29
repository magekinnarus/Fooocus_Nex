# Model Catalogs

This folder holds normalized model-catalog JSON files for M06.

The app-facing goal is:
- keep source catalogs separate from runtime code
- support multiple catalog files at once
- normalize different provider formats into one runtime view
- keep thumbnails repo-owned and stable across Colab sessions

## Current Draft Templates

- `civitai_catalog.template.json`
- `huggingface_catalog.template.json`

## Current Runtime Seed

- `m06_runtime_seed.catalog.json`

These are draft normalized templates generated from the Director's existing source files and are intended to serve as starter examples for user-created catalogs.

## Source Catalogs vs Runtime Catalogs

Every catalog should declare its upstream provider when it is created.

For M06, the provider layer is currently:
- CivitAI catalogs
- HuggingFace catalogs

Private or personal catalogs are still expected to fall under one of those providers. In other words:
- `private` / `personal` describe ownership and maintenance
- `source_provider` describes the actual download source and token behavior

Examples:
- a personal CivitAI catalog should still use `source_provider: "civitai"`
- a private HuggingFace catalog should still use `source_provider: "huggingface"`

The app should normalize them into one unified runtime index, but users can still maintain them as separate JSON files.

## Recommended Asset Layout

For clarity, model assets should be organized by model root first, then architecture, then subtype where that subtype is meaningful.

Checkpoint layout:
- `checkpoints/sd15/base/`
- `checkpoints/sdxl/base/`
- `checkpoints/sdxl/pony/`
- `checkpoints/sdxl/illustrious/`
- `checkpoints/sdxl/noob/`

GGUF quantized assets are SDXL-only in this project:
- quantized UNet files go under `unet/`
- extracted CLIP files go under `clip/`
- there is no `unet/sd15/base/`

Recommended UNet layout:
- `unet/sdxl/base/`
- `unet/sdxl/pony/`
- `unet/sdxl/illustrious/`
- `unet/sdxl/noob/`

Recommended CLIP layout:
- `clip/sdxl/base/`
- `clip/sdxl/pony/`
- `clip/sdxl/illustrious/`
- `clip/sdxl/noob/`

LoRAs follow the same convention as checkpoints and UNet, except there is no Noob LoRA bucket:
- `loras/sd15/base/`
- `loras/sdxl/base/`
- `loras/sdxl/pony/`
- `loras/sdxl/illustrious/`

Embeddings are architecture-scoped but do not use Pony / Illustrious / Noob subtype buckets:
- `embeddings/sd15/`
- `embeddings/sdxl/`

VAE is also architecture-scoped only:
- `vae/sd15/`
- `vae/sdxl/`

For quantized SDXL workflows, the same SDXL VAE family may be shared across multiple models, including fp16 and other variants.

The `loras/sdxl/noob/` example from earlier drafts should be considered obsolete.

These subfolders are for organization, not hard compatibility walls. By default:
- SDXL, Pony, Illustrious, and Noob checkpoints / UNets / CLIP assets should resolve to `compatibility_family: "sdxl"`
- Pony and Illustrious LoRAs should still be treated as SDXL-family assets for runtime compatibility
- SD15 assets should resolve to `compatibility_family: "sd15"`

## Recommended Multi-Catalog Layout

Preset/example catalogs can live in this repo folder.

User/private catalogs should live in a writable catalog folder and be loaded alongside preset catalogs.

Recommended runtime naming:
- active catalogs: `*.catalog.json`
- starter references: `*.template.json`, `*.example.json`

The runtime loader should only ingest `*.catalog.json` files so templates and examples remain available without automatically appearing in the app.

Conceptually, the runtime view is built from:
1. one or more catalog JSON files
2. filesystem scanning of installed models
3. active download-job state

## Important Identity Rules

Do not use filenames alone as the canonical identity.

Each normalized entry should distinguish between:
- `id`: unique entry identity
- `architecture`: broad runtime family such as `sdxl` or `sd15`
- `sub_architecture`: organization subtype such as `base`, `pony`, `illustrious`, `noob`, or `null` when no subtype split is needed
- `compatibility_family`: runtime compatibility group, for example all SDXL subtypes map to `sdxl`
- `asset_group_key`: shared family identity across variants
- `thumbnail_key`: shared visual identity
- source identity such as `source_provider`, `source_model_id`, `source_version_id`

This allows:
- installed models to be removed from the Available view
- checkpoint and extracted UNet variants to share one thumbnail
- Colab session downloads to disappear without breaking thumbnail lookup

## Thumbnail Strategy

Thumbnails are repo-owned assets, not runtime-fetched dependencies.

Recommended behavior:
- use `thumbnail_library_relative` when present
- otherwise resolve by `thumbnail_key`
- if no specific thumbnail exists, fall back to `0001_default.jpg`

### Naming Convention

Human-friendly thumbnail filenames should use:
- `{code}_{slug}.jpg`

Examples:
- `0001_default.jpg`
- `0002_stoiqoNewreality.jpg`
- `0003_eventHorizon.jpg`
- `0004_homoveritas.jpg`

The actual binding should still come from JSON fields such as `thumbnail_key` and `thumbnail_library_relative`, not just filename matching.

## CivitAI Download Convention

For the Director's current CivitAI workflow, downloads are token-authenticated through `.env`.

Expected pattern:
- base URL: `https://civitai.com/api/download/models/`
- final URL: `https://civitai.com/api/download/models/{id}?token={CIVITAI_TOKEN}`

Normalized CivitAI entries should therefore typically declare:
- `source_provider: "civitai"`
- `token_required: true`
- `token_env: "CIVITAI_TOKEN"`

## Legacy Filename Prefixes

Legacy prefixes such as:
- `SDXL_`
- `PONY_`
- `IL_`
- `Noob_`

should be treated as import hints, not part of the long-term canonical naming scheme.

Normalized entries may preserve the original imported filename for traceability, but runtime taxonomy should come from metadata and folder structure rather than name prefixes.

## Key Fields in the Draft Schema

Common fields in the current draft templates include:
- `id`
- `alias`
- `name`
- `source_file_name` or `source_key`
- `display_name`
- `model_type`
- `architecture`
- `sub_architecture`
- `compatibility_family`
- `root_key`
- `relative_path`
- `asset_group_key`
- `thumbnail_key`
- `thumbnail_library_relative`
- `source_provider`
- `source_model_id`
- `source_version_id`
- `catalog_source`
- `storage_tier`
- `visibility`
- `preset_managed`
- `token_required`
- `token_env`
- `sources`

## Planned Runtime Consumers

- `modules/model_download/catalog.py`
- `modules/model_download/policy.py`
- `modules/model_download/resolver.py`
- `modules/model_download/transport.py`
- `modules/model_download/orchestrator.py`
- future M06 runtime index / manager modules

