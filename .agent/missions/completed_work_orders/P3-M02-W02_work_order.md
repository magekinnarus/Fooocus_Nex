# Work Order: P3-M02-W02

**Mission:** P3-M02 (Component-First Loader)
**Phase:** 3
**Work Order ID:** P3-M02-W02
**Status:** Ready
**Depends On:** P3-M02-W01
**Date Issued:** 2026-02-14
**Philosophy:** Logic uses Definition

## Objective
Implement `extract_sdxl_components` in `Fooocus_Nex/backend/loader.py` using the definition from `defs/sdxl.py`.

## Logic
1.  Import `Fooocus_Nex.backend.defs.sdxl as sdxl_def`.
2.  Use `sdxl_def.PREFIXES` for the splitting logic.
3.  Implement the same "Atomic Splitting" and "Key Normalization" logic as previously discussed (W02 v2).

## Distinction
*   This Work Order implements the **Process** (iterating, creating dicts).
*   The **Data** (what keys to look for) is imported.
*   *Benefit*: If future models just need different prefixes, the logic remains extracting-ish (or we make a generic extractor), but for now, strict SDXL logic using externalized config is the sweet spot.

## Validation
*   Extractor works as before, but constants are loaded from `defs`.
