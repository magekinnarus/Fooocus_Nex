---
description: Workflow for code implementation agents (analysis and implementation roles)
---

# Code Manager Workflow

## Overview
This workflow defines two roles for mission execution.
The Director may split planning and implementation across different sessions and agents.
PM provides mission briefs, initial work lists, and work orders. CM Role 1 reviews these, injects necessary technical context/references, tracks progress, and owns mission closure.

Required references:
- `.agent/rules/02_Documentation_and_Mission_Standards.md`

## Naming Convention (Required)
- Mission ID: `P{phase}-M{mission}`
- Work Order ID: `P{phase}-M{mission}-W{work_order}`

Core files:
- `{mission_id}_mission_brief.md`
- `{mission_id}_work_list.md`
- `{work_order_id}_work_order.md`
- `{work_order_id}_work_report.md` or `{work_order_id}_issue_report.md`
- `{mission_id}_mission_report.md`

Conditional files:
- `{id}_reference_trace.md`
- `{id}_benchmark_report.md`
- `{id}_decision_record.md`

## Role 1: Mission Analyst

Input:
- Approved mission brief (from PM)
- Initial mission work-list (from PM)
- Initial work-order documents (from PM)

Output:
- Reviewed and annotated work-order documents (with added mandatory references)
- Issue reports or clarification notes (if technical feasibility is questioned)
- Mission closure package and mission report

### Instructions
1. Read the mission brief, work list, and all work orders provided by the PM.
2. Review relevant source code to digest the technical work required for each order.
3. Identify and raise any potential implementation issues or architectural conflicts.
4. Inject/add mandatory reading references (e.g., specific guideline docs like `.agent/summaries/05_Local_Environment_Guidelines.md` or architecture traces) into the relevant work orders so Role 2 has full context.
5. Keep the work-list aligned with status, dependencies, and blockers during execution.
6. On closure, prepare the mission report and archive artifacts.

## Role 2: Implementor

Input:
- Approved work-order docs
- Current mission work-list

Output:
- Implementation plan (must be approved before coding)
- Code changes (executed in approved steps)
- Work report or issue report
- Session handoff note (if session ends mid-work)

### Instructions
1. Read the approved work-order and all mandatory references listed in it.
2. **Draft an implementation plan** before writing any code:
   - Break the work order into discrete, numbered steps.
   - For each step, list the files to modify, the specific changes, and the expected outcome.
   - Present the plan to the Director for approval.
   - **Do not proceed with coding until the Director approves the plan.**
   - If the Director requests changes to the plan, revise and re-submit.
3. **Execute steps one at a time with Director approval:**
   - Implement only the current approved step.
   - Present the result of each step to the Director (files changed, what was done, any issues).
   - **Wait for Director approval before proceeding to the next step.**
   - If a step reveals unexpected complexity, pause and discuss with the Director before continuing.
4. Validate against mission success criteria at each step boundary.
5. Write completion report or issue report:
   - `.agent/missions/active/{work_order_id}_work_report.md`
   - `.agent/missions/active/{work_order_id}_issue_report.md`
   - *Note: Leave all completed work orders and reports in the `active/` directory. Do not move them to `completed_work_orders/` yet.*
6. Update work-list at handoff points (minimum).
7. If the session ends mid-work or is escalated, produce a **Session Handoff** note per Section 16 of `02_Documentation_and_Mission_Standards.md`.

## Handoff and Closure
1. PM assigns `{mission_id}_mission_brief.md`, `{mission_id}_work_list.md`, and initial work orders.
2. Analyst (Role 1) reviews these, adds mandatory references, and checks technical feasibility.
3. Implementor (Role 2) drafts an implementation plan and obtains Director approval before coding.
4. Implementor executes approved steps one at a time, with Director sign-off at each step.
5. If a session ends mid-work, the active agent writes a **Session Handoff** note (see `02_Documentation_and_Mission_Standards.md` Section 16).
6. Work-list is updated at handoff or closure.
7. Mission closes with `{mission_id}_mission_report.md`.
8. Archive mission documentation (performed by Role 1 upon mission closure):
   - Move mission-level files (brief, work-list, mission-report: `{mission_id}_*`) from `active/` to `completed/`.
   - Move work-order level files (orders, reports, issues: `{mission_id}-W*`) from `active/` to `completed_work_orders/`.
   - *Caution: Misusing wildcards (e.g., `{mission_id}*`) will incorrectly mix these folders.*

## Key Locations
- Active: `.agent/missions/active/`
- Completed: `.agent/missions/completed/`
- Completed work orders: `.agent/missions/completed_work_orders/`
- Rules: `.agent/rules/`
- Strategy docs: `.agent/summaries/`
