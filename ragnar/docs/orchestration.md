# Ragnar Orchestration

Ragnar uses LangGraph as a strict orchestration spine.

The rule is:

- LangGraph controls phase order, retries, checkpoint state, and approval gates.
- Letta role agents provide durable role intelligence inside bounded nodes.
- The approval broker enforces outward-action policy outside the LLM.
- pgvector and Graphiti provide retrieved context; raw history is not copied into every prompt.

## Current Graph

```text
START
  -> intake_objective
  -> conductor_triage
  -> architect_plan
  -> dispatch_build_roles
  -> backend_engineer?
  -> frontend_engineer?
  -> workflow_engineer?
  -> qa_gate
  -> integrator_prepare
  -> approval_gate
  -> final_report
  -> END
```

Build-role routing is deterministic. The conductor node selects the relevant build roles from the objective, then the graph allows only the fixed role sequence above.

## Strict But Intelligent

Strict:

- no node can skip QA
- no node can merge, deploy, send, spend, or mutate production
- approval checks happen in `ApprovalBroker`, not in prompts
- every outward proposed action becomes an approval request
- each role node is restricted to actions granted in `ragnar_roles.yaml`

Intelligent:

- the conductor chooses relevant build roles from the objective
- every role node receives role-scoped context-query instructions
- future Letta calls can replace the deterministic role packets without changing graph gates
- Graphiti can add time-aware facts before each role node acts

## Run

```sh
cd ragnar
. .venv/bin/activate
ragnar-orchestrate "build a frontend settings page with backend API and webhook integration"
```

By default the CLI uses SQLite checkpoints at `.ragnar/orchestrator.sqlite`. For a temporary in-memory run:

```sh
ragnar-orchestrate "build a frontend settings page with backend API and webhook integration" --no-checkpoint
```

For full graph state:

```sh
ragnar-orchestrate "build a frontend settings page with backend API and webhook integration" --json
```

The first implementation is intentionally a dry-run orchestrator. It proves routing, policy, artifact contracts, and approval blocking before we let execution adapters modify files or open pull requests.

## Provider-Free Runtime Pieces

These pieces now work without OpenAI, Anthropic, or other model provider keys:

- local approval ledger at `.ragnar/approvals.jsonl`
- local run traces at `.ragnar/runs/`
- bounded local QA command execution
- role runtime envelopes bound to the Letta manifest when present
- best-effort memory retrieval that degrades cleanly if pgvector or Graphiti are offline

## Memory Modes

The CLI defaults to pgvector best-effort retrieval:

```sh
ragnar-orchestrate "fix auth API and login screen"
```

Memory mode can be controlled per run:

```sh
ragnar-orchestrate "fix auth API" --memory-mode off
ragnar-orchestrate "fix auth API" --memory-mode pgvector
ragnar-orchestrate "fix auth API" --memory-mode all
```

`graphiti` and `all` require the local Graphiti MCP service. If it is not running or its provider keys are missing, the graph records the memory error as context and continues.

## QA Commands

QA commands are explicit and local. They are captured in the `qa_verdict` artifact:

```sh
ragnar-orchestrate "fix auth API" --qa-command "python -m compileall -q src"
```

Multiple commands can be passed by repeating `--qa-command`.

## Approval Flow

When the graph proposes an outward action, it records a pending approval and blocks:

```sh
ragnar-orchestrate "build frontend settings page and open PR"
```

Approve the pending action:

```sh
ragnar-approval approve <run_id> integrator open_pull_request
```

Then rerun with the same run ID:

```sh
ragnar-orchestrate "build frontend settings page and open PR" --run-id <run_id>
```

The approval broker remains outside the LLM. The graph will only treat the action as approved if the local approval ledger has an owner approval for the same `run_id`, `role_id`, and `action`.
