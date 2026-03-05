# Work List: P3-M10 — Fooocus Backend Integration
**Mission:** P3-M10
**Date Created:** 2026-02-24

| # | Work Order | Description | Status | Depends On |
|---|-----------|-------------|--------|------------|
| 1 | P3-M10-W01 | Replace `default_pipeline.py` and `core.py` internals to route through `backend/` | [x] | — |
| 2 | P3-M10-W02 | Replace `patch.py` monkey-patches #1–5 and integrate UNet forward precision casting | [x] | W01 verified |
| 3 | P3-M10-W03 | Strip dead features from `async_worker.py`, clean up residual imports, integration testing | [x] | W02 verified |
| 4 | P3-M10-W04 | Fix integration bugs: consecutive generation hangs, memory/RAM leaks, and UI buttons | [x] | W03 verified |

## Sequencing Rationale
W01 establishes the plumbing by rewiring the middle layer (`default_pipeline.py` / `core.py`) to call `backend/` instead of `ldm_patched`. This gives the monkey-patches in `patch.py` a clean surface to land on.

W02 removes the monkey-patches themselves — replacing `ldm_patched` overrides with native backend calls. This is the highest-risk step because it changes the actual inference path.

W03 is cleanup: stripping dead features (enhancement, wildcards), removing stale imports, and conducting end-to-end integration testing through the Fooocus UI.
