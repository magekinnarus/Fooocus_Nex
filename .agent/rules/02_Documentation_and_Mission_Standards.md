# Documentation and Mission Standards

**Version:** 1.5
**Effective Date:** 2026-02-13
**Scope:** All agents working in this repository

## 1. Purpose
This document defines shared principles for documentation, mission tracking, numbering, and status management.
The goal is traceability with low overhead in a multi-session, multi-model workflow.

## 2. Operating Model (Multi-Session Aware)
This project may run across many short sessions with different agents.
The Director may request document creation or updates at session start, session end, or both.

Principle:
- Keep documentation lightweight and handoff-friendly.
- Preserve continuity through mission IDs and work-list status, not through session ownership.
- Close the loop between agent analysis and real practice through Director/Gopher testing feedback.

## 3. Rule Precedence
If instructions conflict, follow this order:
1. `.agent/rules/01_Global_Context_Rules.md`
2. Active mission brief
3. This document
4. Workflow-specific guidance

## 4. Role Matrix (Canonical)
Use this section as the single source of truth for role boundaries.

| Role | Primary Responsibility | Must Produce | Must Not Do |
|------|------------------------|--------------|-------------|
| Director/Gopher (User) | Approve plans, decide priorities, run practical tests, and relay field insights | Approval decisions, mission direction, test outcomes, fetched data, trial-and-error insights | Assume agent-only analysis is sufficient without practical validation feedback |
| PM (`Nex_pm`) | Project-level analysis and mission definition | Mission briefs, `{mission_id}_work_list.md`, `{work_order_id}_work_order.md`, roadmap docs, mission-level review decisions | Code implementation, work reports, issue reports |
| CM Role1 (Mission Analyst) | Mission readiness review and tracking | Annotated/Reviewed `{work_order_id}_work_order.md`, mission closure package, `{mission_id}_mission_report.md` | Code implementation |
| CM Role2 (Implementor) | Execute approved work orders | `{work_order_id}_work_report.md` or `{work_order_id}_issue_report.md`, code changes | Work outside approved work-order scope |

Notes:
- PM creates the initial work-list and work-orders.
- CM Role1 owns mission state tracking via `work_list` execution, reading work-orders to inject mandatory technical references for Role 2.
- PM reviews CM Role1 mission reports and decides next mission with Director.
- Director/Gopher provides hands-on trial feedback that agents must treat as first-class input for replanning.

## 5. Canonical ID Convention (Required)
All new work uses this structure:

- Mission ID: `P{phase}-M{mission}`
- Work Order ID: `P{phase}-M{mission}-W{work_order}`
- Optional issue variant: `P{phase}-M{mission}-W{work_order}-ISSUE{n}`

Formatting:
- `phase`: integer or decimal (`3`, `3.5`)
- `mission`: two digits minimum (`M01`, `M12`)
- `work_order`: two digits minimum (`W01`, `W02`)

## 6. Document Types
### Core artifacts (default set)
- `_mission_brief.md`
- `_work_list.md`
- `_work_order.md`
- `_work_report.md`
- `_issue_report.md`
- `_mission_report.md`

### Conditional artifacts (create only when needed)
- `_reference_trace.md`
Use when the mission depends on code-path tracing, reverse engineering, or extraction mapping.

- `_benchmark_report.md`
Use when the mission includes performance comparison, optimization, or timing-based acceptance criteria.

- `_decision_record.md`
Use when a decision changes architecture direction, conventions, scope boundaries, or supersedes prior assumptions.

## 7. Directory Structure
- Active mission docs: `.agent/missions/active/`
- Completed mission docs: `.agent/missions/completed/`
- Completed work-order docs: `.agent/missions/completed_work_orders/`
- Shared strategy docs: `.agent/summaries/`
- Rules: `.agent/rules/`
- Reference traces: `.agent/reference/`
- Benchmarks (optional): `.agent/benchmarks/`
- Decisions (optional): `.agent/decisions/`
- Archived legacy docs: `.agent/archive/`

## 8. Minimal Metadata (No Session Ownership Requirement)
Mission and work-order files should include:
- `ID`
- `Phase`
- `Date Issued` (or `Date Completed` for final reports)
- `Status`
- `Depends On` (or `None`)
- `Reference Material` (Link to Mission Brief and key source files)

Work-list file should include:
- `Mission ID`
- `Date Opened`
- `Last Updated`
- `Overall Status`
- `Brief Reference`
- `Reference Material` (Link to Mission Brief)

## 9. Work-List Principle (Source of Truth)
Each mission should maintain one work-list file:
- `.agent/missions/active/{mission_id}_work_list.md`

Work-list should track:
- work-order IDs and statuses
- dependencies
- issue/report links
- replanning notes when scope changes

In multi-session operation, update timing is flexible.
At minimum, update it before handoff or closure.

## 10. Lifecycle Baseline
A mission is considered well-closed when:
1. Mission brief exists
2. Work-list exists
3. Executed work orders have report or issue records
4. Mission report summarizes outcome
5. Closed artifacts (including all work orders and work reports) are moved from `active/` to completed folders ONLY when the entire mission is closed. Do not move individual work orders as they finish.
6. Roadmap reflects current status

## 11. Roadmap Synchronization Principle
Keep `.agent/summaries/03_Roadmap.md` aligned with actual mission state.
Use practical statuses (`In Progress`, `Partial`, `Complete`, `Blocked`) to reduce drift.

## 12. Legacy Handling Principle
Legacy IDs (`001`, `002`, `003`) remain valid history.
New work should use canonical `P*-M*-W*` IDs.

If legacy files are referenced, prefer notes/mapping over disruptive renames unless Director asks.

## 13. Reference Integrity Principle
Prefer workspace-relative paths.
If a path is missing or moved, either:
- update the reference, or
- mark it as `Missing` with a short note.

## 14. Scope and Pragmatism Principle
Documentation should support execution, not slow it down.
Avoid extra files unless they improve handoff clarity or decision traceability.

## 15. Enforcement Style
These standards are guidance-first.
If the Director requests an exception for workflow efficiency, follow the Director and note the exception in the mission work-list or mission report.

## 16. Session Handoff Document (Required)

When a session ends mid-work, encounters unresolvable problems, or is escalated to a different agent/session, the active agent **must** produce a handoff note before closing. This ensures the next session can resume without re-investigation.

### When to Write
- Session ends with work still in progress
- Agent encounters a blocker it cannot resolve and escalates
- Director explicitly requests handoff

### Where to Write
Append a `## Session Handoff` section to the active work report:
- `.agent/missions/active/{work_order_id}_work_report.md`

If no work report exists yet, create one with at minimum the handoff section.

### Required Content

```markdown
## Session Handoff
**Date:** YYYY-MM-DD
**Mission:** {mission_id}
**Work Order:** {work_order_id}
**Status at Handoff:** IN PROGRESS | BLOCKED | ESCALATED

### Completed Tasks
- [ ] Task description — brief outcome

### Pending Tasks
- [ ] Task description — what remains

### File Change Manifest
| Action   | File Path (workspace-relative)       | Notes                  |
|----------|--------------------------------------|------------------------|
| EDITED   | `path/to/file.py`                    | Brief description      |
| DELETED  | `path/to/removed_file.py`            | Reason                 |
| MOVED    | `old/path.py` → `new/path.py`       | Why moved              |
| CREATED  | `path/to/new_file.py`               | Purpose                |

### Blockers / Context for Next Session
- Description of any unresolved issues, errors, or decisions needed
```

### Principle
The handoff note is **not** a full report — keep it minimal and actionable. Its purpose is to eliminate re-investigation time for the next agent.
