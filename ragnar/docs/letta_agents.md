# Durable Letta Role Agents

`ragnar_core.letta_provisioner` turns every role in `roles/ragnar_roles.yaml` into a durable Letta agent.

## Provisioning Command

From the `ragnar/` directory:

```bash
pip install -e .
export LETTA_SERVER_URL=http://localhost:8283
# export LETTA_API_KEY=... # only when your Letta server requires it

ragnar-provision-letta
```

If you want to use the cloned Letta repo locally, start it separately from:

```bash
cd ../vendor/letta
docker compose up
```

Then return to `ragnar/` and run `ragnar-provision-letta`.

Dry run:

```bash
PYTHONPATH=src python3 -m ragnar_core.letta_provisioner --dry-run
```

The command writes the created Letta IDs to:

```text
ragnar/.ragnar/letta_agents.json
```

The manifest is used to avoid creating duplicate agents on repeated runs.
Repeated runs also sync Ragnar-managed read-only memory blocks into existing
agents, so role definition changes apply without deleting each agent's
`working_lessons` memory.

## What Each Letta Agent Gets

Each role is created as:

```text
name = ragnar__<role_id>
tags = ragnar, role:<role_id>, team:<team>, memory:<private_namespace>
model = models.roles.<role_id>.model from ragnar.yaml, or RAGNAR_LETTA_MODEL when --no-config is used
embedding = RAGNAR_LETTA_EMBEDDING or letta/letta-free
include_multi_agent_tools = execution.enable_agent_messaging
```

When `ragnar.yaml` is used, each role's model comes from `models.roles`.

Each role receives five memory blocks:

1. `persona`
   - Read-only.
   - Contains the role's operating instructions, responsibility, allowed actions, approval actions, denied actions, and handoff rules.

2. `role_contract`
   - Read-only.
   - JSON copy of the role contract from `ragnar_roles.yaml`.

3. `memory_scope`
   - Read-only.
   - Private and shared memory namespaces this role is allowed to use.

4. `domain_stack_operating_model`
   - Read-only.
   - Defines the durable rule that the agent is domain and stack agnostic
     inside its role boundary.
   - Requires repo evidence or `project_profile` before making stack-specific
     choices about language, framework, package manager, cloud, database,
     folder layout, test command, or deployment surface.
   - Requires stack/domain lessons to be stored with tags such as
     `language:<name>`, `framework:<name>`, `domain:<name>`, and `role:<role_id>`.

5. `working_lessons`
   - Writable.
   - The role's durable self-improvement memory.
   - Stores repo conventions, repeated mistakes, owner preferences, QA findings, and tool-use lessons.

## Communication

Provisioned roles include Letta multi-agent tools only when
`execution.enable_agent_messaging` is true, and are tagged by role and team.
This gives us a base for:

- direct role-to-role handoffs by Letta agent ID
- team broadcasts by tag
- conductor-to-specialist dispatch
- QA feedback back to engineer roles

The Ragnar handoff protocol still needs to wrap this so messages are typed artifacts, not loose chat.

## Self-Training Model

The first self-training loop is memory-based, not base-model fine-tuning:

```text
run task
  -> collect outcome
  -> collect QA result
  -> collect human decision
  -> extract lesson
  -> write lesson to role working_lessons
  -> retrieve lesson on future runs
```

Fine-tuning can come later after enough approved/rejected traces exist.
