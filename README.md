# Matron Foundation

This folder is the first shape of the governed AI delivery crew.

The cloned repositories in `vendor/` are reference/building-block projects. The Matron engine itself lives under `matron/` and should stay role-contract driven rather than agent-name driven.

## Cloned Building Blocks

- `vendor/langgraph`: durable workflow graph, checkpoints, parallel execution, human gates.
- `vendor/agent-framework`: alternate production workflow/runtime reference, especially for durable sessions and A2A/MCP direction.
- `vendor/letta`: durable role agents with identity, memory, tools, and long-lived learning.
- `vendor/graphiti`: shared temporal project/context graph.
- `vendor/openhands`: coding execution worker/runtime reference.
- `vendor/swe-agent`: issue-fixing/code-task execution worker reference.
- `vendor/langfuse`: tracing, evals, cost visibility, and run observability.

## Core Design

Matron should model:

```text
Role Contract -> Runtime Agent Instance -> Memory Scope -> Tool Scope -> Permission Scope -> Handoff Protocol
```

It should not model:

```text
Agent name -> Prompt
```

The display names from the one-pager are labels only. The engine authority comes from role responsibilities, scopes, and permissions.

## Iron Rule

Reads and research run free. Every outward action waits for owner approval.

This must be enforced structurally by the approval broker, not only by prompt instructions.
