# Matron Vector Memory

Matron uses Letta as the durable agent runtime and memory substrate.

The memory rule is:

- Keep only small identity and operating policy in core memory blocks.
- Store durable role lessons in each agent's archival memory.
- Store reusable project context in shared Letta archives.
- Retrieve memory on demand by semantic search instead of placing raw history in the prompt.

## Layers

1. Core memory: minimal prompt-resident role identity and guardrail reminders.
2. Private archival memory: vector passages owned by one role agent, such as `roles/backend_engineer`.
3. Shared archives: vector passages attached to multiple roles, such as `project_context`, `decisions`, `architecture`, and `incidents`.
4. Context graph: optional Graphiti/Cognee layer for entity and relationship memory when vector chunks are not enough.
5. Observability: Langfuse traces retrieval quality, token use, and failed recalls.

## Bootstrap

Preferred local bootstrap, with local embeddings and the Letta Docker Postgres `pgvector` extension:

```sh
cd matron
. .venv/bin/activate
matron-rag-memory bootstrap
matron-rag-memory search "backend database migration lessons" --role backend_engineer
```

Letta archival-memory bootstrap, once the Letta container has a working embedding provider such as OpenAI or Ollama:

```sh
cd matron
. .venv/bin/activate
matron-bootstrap-memory --base-url http://127.0.0.1:8283
```

`matron-rag-memory` writes to a Matron-owned vector table in the same Postgres server. `matron-bootstrap-memory` writes into Letta's own archival memory and shared archives.

## Recommended Retrieval Flow

For each task:

1. Identify role and project.
2. Search the role's private archival memory.
3. Search only the shared namespaces listed in that role contract.
4. Summarize retrieved passages into a compact working context.
5. Execute or hand off.
6. Write a concise new memory only if the task produced durable learning.

## Repos Worth Wiring

- Letta: primary durable agent runtime and vector archival memory.
- Graphiti: temporal graph memory for evolving facts, decisions, incidents, and relationships.
- Mem0: optional external memory layer if we want a packaged memory service across non-Letta agents.
- Cognee: optional graph plus vector memory platform for documents and project knowledge.
- LangMem: useful if LangGraph becomes the orchestration layer.
- LlamaIndex: useful for document ingestion and RAG pipelines.
- pgvector: storage primitive already present in Letta's Docker Postgres.
