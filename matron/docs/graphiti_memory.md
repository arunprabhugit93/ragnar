# Matron Graphiti Memory

Graphiti adds temporal graph memory beside Matron's pgvector RAG memory.

Use pgvector for chunk retrieval:

- role-private passages such as `roles/backend_engineer`
- shared project passages such as `architecture`, `qa_findings`, and `project_context`
- cheap semantic search before a role acts

Use Graphiti for graph-shaped, time-aware facts:

- role owns memory namespace
- role hands off to another role
- a decision replaced an older decision
- an incident began, changed state, and was resolved
- a project requirement became invalid after a later owner decision

## Start Graphiti

The Matron compose file avoids the local Redis port conflict by exposing FalkorDB on host port `6380`.

```sh
cd vendor/graphiti/mcp_server
docker compose -f docker/docker-compose-matron.yml up -d falkordb graphiti-mcp
```

Endpoints:

- Graphiti MCP: `http://127.0.0.1:8000/mcp/`
- FalkorDB Redis protocol: `127.0.0.1:6380`
- FalkorDB browser: `http://127.0.0.1:3001`

## Provider Key

Graphiti needs an LLM and embedding provider for `add_memory`, `add_triplet`, and graph search. Put the key in:

```sh
vendor/graphiti/mcp_server/.env
```

For OpenAI:

```sh
OPENAI_API_KEY=...
MODEL_NAME=gpt-4.1-mini
EMBEDDER_MODEL=text-embedding-3-small
```

The file already sets non-secret Matron defaults: `GRAPHITI_GROUP_ID=matron`, `FALKORDB_DATABASE=matron`, and `SEMAPHORE_LIMIT=2`.

## Seed Role Graph

```sh
cd matron
. .venv/bin/activate
pip install -e .
matron-graphiti-memory status
matron-graphiti-memory seed-roles
```

This queues one role-contract episode per Matron role and writes direct graph triplets for:

- role -> private memory namespace
- role -> shared memory namespaces
- role -> team
- role -> handoff target roles

## Search

```sh
matron-graphiti-memory search-facts "who can receive backend engineer handoffs"
matron-graphiti-memory search-nodes "qa findings and ship verdicts"
```

## How This Fits

For each durable Letta role agent:

1. Read its Letta core identity and guardrails.
2. Search private pgvector memory.
3. Search allowed shared pgvector namespaces.
4. Search Graphiti for time-aware relationships and current facts.
5. Pass only the relevant snippets into the working prompt.
6. Write new durable lessons to pgvector and durable decisions/incidents/handoffs to Graphiti.
