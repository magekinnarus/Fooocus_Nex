# Changelog

All notable changes to Nexfocus are recorded in this file.

This project has not yet published a tagged product release. Work accumulates
under `Unreleased` until the controlled release process creates a
versioned entry. Corrective changes may add further `Unreleased` entries at any
time.

## Unreleased

### Added

- Full public documentation and launch-asset package: README, FEATURES,
  USAGE, PROMPT_PRESETS, INSTALL, CONTRIBUTING, and this changelog, plus
  Windows (`launch.bat`) and Linux (`launch.sh`) launchers and a safe
  `.env_template` for optional credentials.
- A walkthrough video published as a public GitHub Release
  asset and linked from the README.
- An official Colab notebook as the supported cloud entry point, linked from
  the README.
- A GIMP plug-in guide for layer exchange between Nexfocus and external
  editing.

### Changed

- Documentation is split into focused root-level files instead of one
  comprehensive README. Hotkey and practical recovery guidance now lives in
  `USAGE.md`; installation remains in `INSTALL.md`.
- The README now leads with a compact feature-walkthrough doorway and routes
  deeper tours to `FEATURES.md`.
- Model catalogues were aligned across the committed Hugging Face, CivitAI,
  and GitHub main catalogues. SD 1.5 main-catalogue entries were removed, and
  six unavailable CivitAI entries were retired: `stoiqo`,
  `realHallucinations`, `sinful_il`, `biwa`, `crystalVAESDXL`, and
  `ponyStandardVAE`.
- Catalogue thumbnails were migrated from 256x256 PNG files to native 400x250
  JPG assets.
- The Windows and Linux launchers verify Python 3.10+, the virtual
  environment, PyTorch with CUDA, xformers, and uv before launch, and only
  Aria2 is installed automatically.

### Fixed

- Corrected README and navigation links to match the final document set,
  including routing hotkey and recovery guidance to `USAGE.md`.

### Removed

- Deprecated upstream `troubleshoot.md` and its references; recovery guidance
  now lives in `USAGE.md`.

### Internal Architecture and Engineering

- SDXL full-quality FP16 streaming on the 3 GB GTX 1050 reference machine,
  including the unified FP16 safetensors pipeline and transient worker
  lifecycles.
- Flux Fill streaming on the same 3 GB GPU through its dedicated native FP8
  streaming runtime.
- FP16 T5-XXL disk paging so Flux prompt conditioning fits Colab Free's
  12.7 GB system-RAM ceiling without defaulting to quantized text encoding.
- One Nex-owned SDXL pipeline contract backed by posture-specific worker
  assemblies for streaming and resident execution.
- CPU- and GPU-pinned text-encoder workers and shadow-copy avoidance so
  placement follows the active environment rather than one universal policy.
- Session-aware in-app model management (browser, downloads, presets) that
  preserves ephemeral Colab sessions.
- User-first pipeline engineering: UNet-only LoRA patching, custom overlay
  masking, staging palette and comparison viewer, GIMP layer exchange, and
  metadata round-trip.
- Production hardening: automated test contract, normal and debug logging,
  explicit compatibility bridges for retained legacy settings, queue-frozen
  execution plans, component-level fingerprints and invalidation, and
  controlled process transitions.

## How to Add an Entry

Add new entries to the most specific subsection under `Unreleased` that
matches the change. Keep user-visible changes in `Added`, `Changed`, `Fixed`,
or `Removed`; keep purely internal work under `Internal Architecture and
Engineering`. Do not invent version numbers. When the
maintainer-controlled release process tags the first release, move the
accumulated entries into a versioned section.
