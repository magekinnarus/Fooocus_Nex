# Work List: P3-M11 — Codebase Cleanup & Dead Code Removal
**Mission:** P3-M11
**Date Created:** 2026-02-25

| # | Work Order | Description | Status | Depends On |
|---|-----------|-------------|--------|------------|
| 1 | P3-M11-W01 | Dead Code Removal — delete unused modules/ files and 25 ldm_patched/contrib/ nodes | ✅ Done | — |
| 2 | P3-M11-W02 | Bridge File Consolidation — move gguf/, consolidate lora, evaluate ops | ✅ Done | W01 |
| 3 | P3-M11-W03 | Backend ldm_patched Import Reduction — remove unsupported model types, inline checkpoint_pickle | ✅ Done | W02 |
| 4 | P3-M11-W04 | UI Cleanup & Backend Wiring — fix double-load, remove dead UI, wire sampler/scheduler | ✅ Done | W01 |

## Sequencing Rationale
W01 removes dead code first — zero-risk deletions that reduce noise and prevent confusion during the more complex consolidation work. Completed with 2 rollbacks; see report.

W02 consolidates bridge files, which involves moving code between layers. This is medium-risk because import paths change across multiple files. **Note:** W01 already deleted `sample_hijack.py` and ported its schedulers to `backend/schedulers.py`. W02 scope should be updated to reflect this.

W03 is the final polish — reducing the backend's remaining `ldm_patched` imports. This depends on W02 because some backend imports may shift after bridge consolidation (e.g., `backend/lora.py` changes if `modules/lora.py` is absorbed).

W04 cleans the Gradio UI of dead features and wires sampler/scheduler dropdowns to the backend. It depends only on W01 and can be executed independently of W02/W03. Should be completed before M12 refactoring begins.
