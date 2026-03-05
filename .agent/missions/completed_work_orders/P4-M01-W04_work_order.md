# Work Order: P4-M01-W04 — SUPERSEDED
**ID:** P4-M01-W04
**Mission:** P4-M01
**Status:** SUPERSEDED
**Superseded By:** P4-M03 (Extended API Endpoints — not yet created)

## Reason for Supersession
The W02 architectural pivot (bypassing `modules/pipeline/` to build directly on `backend/`)
invalidated this work order's approach. It referenced `modules/default_pipeline.py`,
`modules/config.py`, `modules/pipeline/image_input.py`, and `TaskState` — all of which
the W02 pivot abandoned.

The work described here (model management, interruption, vary/upscale) has been deferred
to P4-M03 (Extended API Endpoints), which will be created after the frontend (P4-M02)
validates the API contract.

## Original Scope (for reference)
- Model Management: `POST /switch-model`, enhanced `GET /models`
- Generation Interruption: `POST /interrupt`
- Vary/Upscale: `POST /vary`, `POST /upscale`
- Error Handling & Polish
- API Documentation

See git history or `.agent/archive/` for the full original work order.
