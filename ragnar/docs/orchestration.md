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
