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
- build roles receive isolated Git worktree reports before execution
- role file scopes and command allowlists are enforced outside the LLM

Intelligent:

- the conductor chooses relevant build roles from the objective
- every role node receives role-scoped context-query instructions and workspace policy
- future Letta calls can replace the deterministic role packets without changing graph gates
- Graphiti can add time-aware facts before each role node acts

## Run

```sh
cd ragnar
. .venv/bin/activate
ragnar-orchestrate "build a frontend settings page with backend API and webhook integration"
```

## Terminal Chat

Use `ragnar-chat` when you want to develop from a terminal prompt:

```sh
cd ragnar
. .venv/bin/activate
ragnar-chat --memory-mode off --no-worktrees
```

Then type an objective:

```text
ragnar> build backend API and frontend settings page
```

Useful chat commands:

```text
/help
/approvals last
/approve last integrator open_pull_request
/rerun
/json on
/quit
```

For one command without opening the REPL:

```sh
ragnar-chat --once "build backend API" --memory-mode off --no-worktrees
```

By default the CLI uses SQLite checkpoints at `.ragnar/orchestrator.sqlite`. For a temporary in-memory run:

```sh
ragnar-orchestrate "build a frontend settings page with backend API and webhook integration" --no-checkpoint
```

For full graph state:

```sh
ragnar-orchestrate "build a frontend settings page with backend API and webhook integration" --json
```

The first implementation still does not let agents freely edit code. It now prepares the safety substrate for that: isolated worktrees, role file-scope policy, command allowlists, diff reports, QA command capture, and approval blocking.

## Provider-Free Runtime Pieces

These pieces now work without OpenAI, Anthropic, or other model provider keys:

- provider/model config in `ragnar.yaml`
- role invocation/result contracts using `ragnar-contract/v1`
- local approval ledger at `.ragnar/approvals.jsonl`
- local run traces at `.ragnar/runs/`
- isolated role worktrees at `.ragnar/worktrees/<run>/<role>`
- bounded local QA command execution
- file-scope and command policy checks through `ragnar-workspace`
- safe unified-diff application through `ragnar-edit`
- local memory writeback ledger at `.ragnar/memory_writebacks.jsonl`
- local PR drafts at `.ragnar/pr_drafts/`
- role runtime envelopes bound to the Letta manifest when present
- best-effort memory retrieval that degrades cleanly if pgvector or Graphiti are offline

## Provider Config

`ragnar.yaml` defines provider/model selection without storing any keys:

```sh
ragnar-config
ragnar-config --role backend_engineer
```

Before provider integration, every role is set to `provider: local` and `model: provider-not-configured`. When keys arrive, this file becomes the single place to select OpenAI, Anthropic, OpenRouter, or local models per role.

## Agent Contract

Every build role packet now contains:

- `agent_invocation`: exact JSON the provider-backed role call must receive
- `agent_result`: provider-free result stub using the same result shape future agents must return
- `handoffs`: structured role-to-role messages
- `memory_writebacks`: records ready for local review, pgvector promotion, or Graphiti promotion

The role output rule is strict: agents return JSON only, propose patches as unified diffs, and never claim outward actions happened.

## Isolated Worktrees

The CLI prepares one Git worktree per selected build role when it is running inside a Git checkout:

```text
.ragnar/worktrees/<run_id>/<role_id>
```

Example:

```sh
ragnar-orchestrate "build backend API and frontend settings page" --memory-mode off
```

Each role packet includes:

- `workspace`: branch, path, status, and Git availability
- `diff`: changed files, diff stat, and file-scope policy result
- `workspace_policy`: allowed file globs and command families

Disable worktree preparation for a pure planning run:

```sh
ragnar-orchestrate "build backend API" --no-worktrees
```

Inspect the policy directly:

```sh
ragnar-workspace policy
ragnar-workspace check-path backend_engineer src/service/users.py
ragnar-workspace check-command qa_engineer "python -m compileall src"
```

The current role nodes do not yet write code. They prepare and report the isolated execution space so the next step can safely add real edit adapters.

## Edit Adapter

`ragnar-edit` applies proposed unified diffs only after:

- extracting changed files from the diff
- checking every changed file against the role file-scope policy
- running `git apply --check`

Example:

```sh
ragnar-edit backend_engineer .ragnar/worktrees/<run>/backend_engineer proposed.patch
```

Use check-only mode:

```sh
ragnar-edit backend_engineer .ragnar/worktrees/<run>/backend_engineer proposed.patch --check-only
```

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

Multiple commands can be passed by repeating `--qa-command`. Commands that do not match the QA allowlist are not executed and force a `fail` verdict.

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

## PR Drafts

The integrator creates a local PR draft artifact but does not call GitHub:

```sh
ragnar-pr-draft <run_id>
```

The actual `open_pull_request` action remains blocked by owner approval.

## Memory Writebacks

Provider-free runs create memory writeback records inside role packets. When run recording is enabled, those records are also appended locally:

```sh
ragnar-memory-writebacks
ragnar-memory-writebacks --run-id <run_id>
```

These are the records that will later be promoted into private role vector memory, shared project memory, and Graphiti temporal facts.
