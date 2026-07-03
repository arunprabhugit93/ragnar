# Matron Implementation Roadmap

## Phase 1 - Skeleton

- Create role registry loader.
- Create approval broker with allow/read, require_approval/outward, deny/destructive policy classes.
- Create append-only audit event writer.
- Create LangGraph workflow for the minimal build flow.
- Wire placeholder adapters for memory, context graph, execution, and tracing.

Minimal crew:

- `conductor`
- `delivery_architect`
- `backend_engineer`
- `qa_engineer`
- `integrator`

## Phase 2 - Durable Role Instances

- Add Letta adapter.
- Create one Letta agent per role contract.
- Give every role a private memory namespace.
- Add shared memory lookup from Graphiti.
- Add memory write policy: roles can write only to their own private namespace and approved shared artifact types.

## Phase 3 - Real Execution

- Add isolated git branch/worktree manager.
- Add execution tools for repo search, file read, patch apply, test run, build run.
- Add OpenHands adapter for long-running code tasks.
- Add task artifact contracts: plan, branch diff, test report, PR draft.

## Phase 4 - Watch Team

- Add registrar, live-ops monitor, researcher, correspondence drafter, and portfolio editor.
- Connect project registry, metrics/logs, web research, inbox draft tools, and briefing generation.

## Phase 5 - Governance Hardening

- Add Langfuse traces and evals.
- Add approval UI/API.
- Add policy tests for every outward action.
- Add replay and resume from checkpoints.
- Add role-level tool permission tests.

