# Adapter Boundary

Adapters isolate Ragnar from third-party frameworks.

Planned adapters:

- `langgraph_runtime`: workflow graph, checkpoints, human gates.
- `agent_framework_runtime`: alternate workflow/runtime adapter for Microsoft Agent Framework.
- `letta_agents`: durable role instances and private role memory.
- `graphiti_context`: shared project and portfolio context graph.
- `openhands_execution`: code execution workers.
- `swe_agent_execution`: issue-oriented code execution workers.
- `langfuse_observability`: traces, costs, evals, replay.
- `approval_broker`: structural approval enforcement for outward actions.

No role should call a third-party SDK directly. Roles request capabilities; adapters enforce policy.
