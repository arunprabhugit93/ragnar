# Matron Architecture

## Runtime Layers

1. Role Registry
   - Defines responsibilities, memory scopes, tool scopes, permissions, and handoff contracts.
   - Source: `matron/roles/matron_roles.yaml`.

2. Orchestration Runtime
   - Use LangGraph as the first workflow spine.
   - Owns run state, task state, retries, checkpoints, parallel branches, and human gates.

3. Durable Role Agent Layer
   - Use Letta for long-lived role instances.
   - Each role gets private memory, model config, skills, and tool bindings.

4. Shared Context Layer
   - Use Graphiti for portfolio, project, architecture, incident, decision, and dependency context.
   - Shared context is retrieved by policy, not copied wholesale into every role.

5. Execution Adapter Layer
   - Start with Claude/Codex CLI adapters if convenient.
   - Add OpenHands as the stronger execution runtime for code work.

6. Approval Broker
   - Central policy engine for outward actions.
   - Blocks send, push, PR open, merge, deploy, spend, destructive SQL, external webhook enablement, and production mutation.

7. Audit and Observability
   - Append-only audit log for actions and approvals.
   - Langfuse for LLM/tool traces, costs, latency, errors, evals, and replay.

## First Build Flow

```text
owner objective
  -> conductor triage
  -> delivery architect plan
  -> backend/frontend/flow build in isolated worktrees
  -> qa ship gate
  -> integrator assemble PR
  -> owner approval for open PR / merge / deploy
```

## State Model

Minimum durable tables/entities:

- `role_contract`
- `agent_instance`
- `agent_memory_namespace`
- `agent_skill_binding`
- `agent_tool_permission`
- `run`
- `task`
- `handoff`
- `artifact`
- `approval_request`
- `audit_event`
- `worktree_assignment`

## Non-Negotiables

- Role prompts are not enough.
- Each role instance must have private durable memory.
- Shared context must be structured and retrievable.
- Tool permissions must be enforced outside the LLM.
- Agents cannot approve their own work.
- Engineers work in isolated branches/worktrees.
- Outward actions require owner approval.

