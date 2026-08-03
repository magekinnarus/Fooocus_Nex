# Contributing to Nexfocus

Nexfocus is a maintained image-generation application with a deliberately
small ownership surface. The project accepts practical, narrowly scoped
contributions: fixes, documentation corrections, workflow improvements, and
clear issue reports with reproduction evidence.

## What to Contribute

- **Bug reports** with the exact environment, the steps to reproduce, and the
  relevant console output.
- **Feature and workflow requests** that describe a real user problem rather
  than only a desired control.
- **Documentation corrections** to the public documents at the repository
  root.
- **Code changes** that fix a defect, improve memory or runtime behavior, or
  add a workflow, while staying within the existing architecture.

## Reporting Issues

Open an issue with:

- operating system, GPU, VRAM, system RAM, and Python version;
- installation method (local install or Colab) and the exact runtime posture
  or model family involved;
- the smallest set of steps that reproduces the problem;
- the expected versus observed behavior; and
- the relevant console block. For reproducible failures, run with
  `--debug-mode` and include the diagnostic lines it produces.

Never include API keys, tokens, the contents of your `.env` file, or other
credentials in an issue. The `.env_template` file documents the supported
credential names; keep real values local.

## Environment and Installation

Follow [INSTALL.md](INSTALL.md) to set up a working installation before
changing code. The launchers (`launch.bat`, `launch.sh`) verify Python 3.10+,
the virtual environment, PyTorch with CUDA, xformers, and uv, and install only
Aria2 automatically. Do not add launcher steps that install other tooling
without a strong reason.

Use the repository virtual environment for development and validation. Plain
system Python is not a supported validation interpreter because its dependency
set is incompatible with the project.

## Repository Conventions

- Keep changes narrowly scoped to the module or document the change is about.
  Do not mix unrelated refactors into a bug fix.
- Match the existing patterns in the file you touch. Public documentation
  stays split into focused root-level files; do not expand the README into a
  comprehensive manual.
- `USAGE.md` owns hotkey and practical recovery guidance. Do not create a
  separate hotkey file.
- `INSTALL.md` remains the installation guide. Audit and correct it rather
  than replacing it.
- Write UTF-8 text. Do not introduce mojibake or non-ASCII punctuation into
  existing ASCII files.
- Preserve line-ending rules in `.gitattributes` for `launch.bat` and
  `launch.sh`.

## Tests and the Maintained Test Contract

Nexfocus protects its behavior with a maintained test suite under `tests/`.
The suite is a local ownership contract, not a public continuous-integration
gate: it runs through the repository virtual environment and may warm local
model or cache assets, so some tests take minutes.

Run the full suite from the project root:

```powershell
.\venv\Scripts\python.exe -m pytest tests\ -q
```

```bash
venv/bin/python -m pytest tests -q
```

The suite is CPU-safe for the covered seams, but model-backed smoke tests may
download or load local support assets. Recorded dependency and hygiene
warnings are not treated as failures.

### Focused Selection

For a change to a specific surface, run the smallest focused set that covers
it. The core ownership replay protects contextual pipeline ownership, loader
residency, SDXL Assembly request/slot/eligibility behavior, frozen workflow
admission, and super-upscale residency:

```powershell
.\venv\Scripts\python.exe -m pytest `
  tests\test_contextual_pipeline_ownership.py `
  tests\test_loader_clip_residency.py `
  tests\test_sdxl_assembly_w07.py `
  tests\test_sdxl_assembly_w08.py `
  tests\test_sdxl_assembly_w09.py `
  tests\test_super_upscale_residency.py -q
```

Run the broader suite before submitting when your change touches shared
fixtures, compatibility bridges, runtime policy, or any behavior another
workflow depends on.

### When a Test Fails

Reproduce the exact failing node, identify the current production owner and
frozen-plan boundary, and assign one disposition: product repair, fixture or
assertion repair, narrow retirement with a documented reason, or a documented
external precondition. Do not edit the suite command to hide a failure.

## Documentation Validation

Documentation changes must be self-consistent:

- every Markdown link and image target resolves from the document's own
  location;
- file names and paths match the repository's actual case and layout;
- images use the `.jpg`/`.png` assets already committed under `assets/` where
  possible;
- no credentials, private paths, or internal project terminology leak into
  public documents; and
- `git diff --check` reports no whitespace errors.

## Generated Assets, Models, Credentials, and Large Files

- Do not commit model files (`.safetensors`, `.ckpt`, `.pth`, `.gguf`, and
  similar), downloaded archives, or generated outputs. The `.gitignore` keeps
  these out of the repository.
- Do not commit `.env`, logs, or private thumbnails. Use `.env_template` for
  credential documentation and keep real values local.
- Large media belongs in GitHub Releases, not in Git. The walkthrough master
  is intentionally ignored, and its public copy is served from a Release
  asset. Ask the maintainers before adding any large binary to the repository.
- If a contribution depends on a generated asset (screenshot, video frame, or
  composite), prefer committed source material and describe how it was made.

## Pull Requests

A good pull request:

- describes what changed and why, with the issue it addresses;
- keeps the diff minimal and scoped to one concern;
- reports the focused tests run and any broader regression results;
- includes documentation updates in the same change when behavior or wording
  changes;
- confirms the walkthrough media and Colab entry point were not duplicated or
  committed;
- and leaves the release process alone: no version tags, no final release
  claims, and no public push without the maintainer-controlled release step.

Reviewers check scope, test evidence, documentation consistency, and whether
the change respects the ownership boundary below.

## Compatibility Bridges and Owned Implementation

The project draws a clear line between mechanism and policy. Inherited or
framework code is mechanism: it may translate, dispatch, mirror, or report a
decision. Nex-owned code is policy: it owns pipeline family choice, runtime
family, artifact lifecycle, warm-state meaning, cache invalidation, and
process transitions.

Compatibility bridges may adapt retained legacy settings to the current
runtime, but they must not silently infer a second answer from mutable task or
UI state, partial runtime identity, or legacy goals. If a behavior feels like
it is being decided in two places, the fix is to consolidate ownership, not to
add another layer that guesses. Prefer moving behavior into the owned
implementation over growing a compatibility bridge.
