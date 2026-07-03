from __future__ import annotations

import json
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ragnar_core.agent_transcripts import AgentTranscriptStore
from ragnar_core.chat import ChatSession, render_state
from ragnar_core.config import ModelConfig, RagnarConfig, load_config
from ragnar_core.contracts import build_invocation_contract
from ragnar_core.edit_adapter import SafePatchAdapter, extract_changed_files
from ragnar_core.letta_provisioner import (
    ROLE_AGENT_DEFINITION_VERSION,
    _memory_blocks,
    _sync_agent_definition_blocks,
)
from ragnar_core.execution import CommandResult
from ragnar_core.orchestrator import MAX_PLAN_REVISIONS, RagnarOrchestrator, run_objective
from ragnar_core.pr_adapter import build_pr_draft
from ragnar_core.project_profiler import build_project_profile, qa_commands_from_profile
from ragnar_core.role_registry import load_role_registry
from ragnar_core.role_runtime import RoleRuntime, RoleRuntimeResult
from ragnar_core.workspace import RoleWorkspaceManager, command_family, default_workspace_policies


class _ScriptedRoleRuntime:
    """Test double: scripted raw_reply sequences for specific (role_id, action) pairs,
    offline-stub behavior (raw_reply=None) for everything else."""

    def __init__(self, scripted: dict[tuple[str, str], list[str]]) -> None:
        self.scripted = scripted
        self._call_counts: dict[tuple[str, str], int] = {}

    def run(self, role: Any, objective: str, action: str, context_hits: list[dict[str, Any]], invocation: Any = None) -> RoleRuntimeResult:
        used_context_hits = sum(len(item.get("hits", [])) for item in context_hits)
        key = (role.role_id, action)
        replies = self.scripted.get(key)
        if replies:
            index = self._call_counts.get(key, 0)
            self._call_counts[key] = index + 1
            reply = replies[min(index, len(replies) - 1)]
            return RoleRuntimeResult(
                role_id=role.role_id,
                runtime="scripted_test_double",
                letta_agent_id="test-agent",
                memory_namespace=role.private_memory_namespace,
                message="scripted reply for test",
                used_context_hits=used_context_hits,
                status="provider_responded",
                raw_reply=reply,
            )
        return RoleRuntimeResult(
            role_id=role.role_id,
            runtime="letta_manifest_stub",
            letta_agent_id=None,
            memory_namespace=role.private_memory_namespace,
            message="offline stub for test",
            used_context_hits=used_context_hits,
            status="no_real_letta_agent_manifest",
            raw_reply=None,
        )


def test_orchestrator_routes_all_build_roles() -> None:
    state = run_objective(
        "build a frontend settings page with backend API and webhook integration",
        checkpoint_db=None,
    )

    assert state["selected_build_roles"] == [
        "backend_engineer",
        "frontend_engineer",
        "workflow_engineer",
    ]
    assert state["blocked"] is True
    assert state["approval_requests"][0]["action"] == "open_pull_request"


def test_orchestrator_uses_backend_default_and_keeps_qa_gate() -> None:
    state = run_objective("fix the login bug", checkpoint_db=None)
    artifact_kinds = [artifact["kind"] for artifact in state["artifacts"]]

    assert state["selected_build_roles"] == ["backend_engineer"]
    assert "backend_work_packet" in artifact_kinds
    assert "qa_verdict" in artifact_kinds
    assert state["phase"] == "awaiting_owner_approval"


def test_orchestrator_runs_configured_qa_command() -> None:
    state = run_objective(
        "fix the database migration",
        checkpoint_db=None,
        qa_commands=[[sys.executable, "-m", "compileall", "-q", "src/ragnar_core"]],
    )
    qa_artifact = next(artifact for artifact in state["artifacts"] if artifact["kind"] == "qa_verdict")

    assert qa_artifact["body"]["verdict"] == "pass"
    assert qa_artifact["body"]["commands"][0]["exit_code"] == 0


def test_orchestrator_does_not_require_memory_provider() -> None:
    state = run_objective(
        "build dashboard UI",
        checkpoint_db=None,
        memory_mode="off",
    )

    assert state["selected_build_roles"] == ["frontend_engineer"]
    assert state.get("memory_context", []) == []


def test_orchestrator_includes_agent_contract_and_writeback_format() -> None:
    state = run_objective(
        "fix the login bug",
        checkpoint_db=None,
        memory_mode="off",
    )
    packet = next(artifact for artifact in state["artifacts"] if artifact["kind"] == "backend_work_packet")

    assert packet["body"]["agent_invocation"]["schema_version"] == "ragnar-contract/v1"
    assert packet["body"]["agent_invocation"]["role_id"] == "backend_engineer"
    assert packet["body"]["agent_result"]["status"] == "no_provider"
    assert packet["body"]["handoffs"][0]["to_role"] == "qa_engineer"
    assert packet["body"]["memory_writebacks"][0]["scope"] == "private"


def test_qa_rejects_disallowed_command() -> None:
    state = run_objective(
        "fix the database migration",
        checkpoint_db=None,
        qa_commands=[[sys.executable, "-c", "print('not allowed')"]],
    )
    qa_artifact = next(artifact for artifact in state["artifacts"] if artifact["kind"] == "qa_verdict")

    assert qa_artifact["body"]["verdict"] == "fail"
    assert qa_artifact["body"]["denied_commands"][0]["policy"]["allowed"] is False


def test_workspace_policy_checks_file_scopes_and_commands() -> None:
    manager = RoleWorkspaceManager(Path.cwd(), enabled=False)

    assert manager.check_path("backend_engineer", "src/service/users.py").allowed is True
    assert manager.check_path("backend_engineer", "components/Button.tsx").allowed is False
    assert manager.check_path("frontend_engineer", "components/Button.tsx").allowed is True
    assert manager.check_command("qa_engineer", [sys.executable, "-m", "compileall", "src"]).allowed is True
    assert manager.check_command("qa_engineer", [sys.executable, "-c", "print(1)"]).allowed is False


def test_workspace_manager_prepares_isolated_git_worktree() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "ragnar@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Ragnar"], cwd=root, check=True)
        (root / "README.md").write_text("test\n", encoding="utf-8")
        subprocess.run(["git", "add", "README.md"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)

        manager = RoleWorkspaceManager(root)
        report = manager.prepare("run-1", "backend_engineer")

        assert report.available is True
        assert report.worktree_path is not None
        assert (Path(report.worktree_path) / ".git").exists()


def test_safe_patch_adapter_applies_allowed_patch_and_rejects_disallowed_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "ragnar@example.com"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Ragnar"], cwd=root, check=True)
        service_path = root / "src" / "service"
        service_path.mkdir(parents=True)
        (service_path / "users.py").write_text("value = 1\n", encoding="utf-8")
        subprocess.run(["git", "add", "src/service/users.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=root, check=True, capture_output=True)

        patch = """diff --git a/src/service/users.py b/src/service/users.py
--- a/src/service/users.py
+++ b/src/service/users.py
@@ -1 +1 @@
-value = 1
+value = 2
"""
        adapter = SafePatchAdapter(RoleWorkspaceManager(root, enabled=False))
        report = adapter.apply("backend_engineer", root, patch)

        assert report.applied is True
        assert (service_path / "users.py").read_text(encoding="utf-8") == "value = 2\n"
        assert extract_changed_files(patch) == ["src/service/users.py"]

        disallowed_patch = """diff --git a/components/Button.tsx b/components/Button.tsx
--- a/components/Button.tsx
+++ b/components/Button.tsx
@@ -1 +1 @@
-old
+new
"""
        validation = adapter.validate("backend_engineer", disallowed_patch)
        assert validation.allowed is False
        assert validation.disallowed_files == ["components/Button.tsx"]


def test_config_and_pr_draft_shapes() -> None:
    config = load_config(Path("ragnar/ragnar.yaml"))
    model = config.role_model("backend_engineer")
    draft = build_pr_draft(
        "run-1",
        "fix auth API",
        [
            {
                "branch": "ragnar/run-1/backend_engineer",
                "changed_files": ["src/service/users.py"],
            }
        ],
    )

    assert model.provider == "openrouter"
    assert model.model.startswith("openrouter/")
    assert draft.status == "draft_only"
    assert draft.source_branches == ["ragnar/run-1/backend_engineer"]
    assert draft.changed_files == ["src/service/users.py"]


def test_chat_session_runs_objective_once() -> None:
    session = ChatSession(
        roles_path=Path("ragnar/roles/ragnar_roles.yaml"),
        config_path=Path("ragnar/ragnar.yaml"),
        memory_mode="off",
        qa_commands=[],
        prepare_worktrees=False,
        record_runs=False,
    )

    output = session.run_objective("build dashboard UI")

    assert "roles: frontend_engineer" in output
    assert "approval required:" in output
    assert session.last_run_id is not None


def test_chat_render_json_mode() -> None:
    rendered = render_state({"run_id": "run-1", "phase": "done", "selected_build_roles": []}, show_json=True)

    assert '"run_id": "run-1"' in rendered


def test_architect_plan_produces_real_agent_artifact_shape_offline() -> None:
    state = run_objective("fix the login bug", checkpoint_db=None)
    plan_artifact = next(a for a in state["artifacts"] if a["kind"] == "architecture_plan")

    assert plan_artifact["owner_role"] == "delivery_architect"
    assert plan_artifact["body"]["agent_invocation"]["role_id"] == "delivery_architect"
    assert plan_artifact["body"]["agent_result"]["status"] == "no_provider"


def test_conductor_review_plan_defaults_to_approve_offline() -> None:
    state = run_objective("fix the login bug", checkpoint_db=None)

    assert state["plan_review_verdict"] == "approve"
    assert state["plan_revision_count"] == 0
    artifact_kinds = [a["kind"] for a in state["artifacts"]]
    assert "conductor_plan_review" in artifact_kinds
    assert "backend_work_packet" in artifact_kinds


def test_conductor_review_qa_defaults_to_approve_offline() -> None:
    state = run_objective("fix the login bug", checkpoint_db=None)

    assert state["qa_review_verdict"] == "approve"
    assert state["qa_rework_roles"] == []
    artifact_kinds = [a["kind"] for a in state["artifacts"]]
    assert "conductor_qa_review" in artifact_kinds
    assert "integration_packet" in artifact_kinds


def test_plan_revision_loop_respects_retry_cap() -> None:
    registry = load_role_registry(Path("ragnar/roles/ragnar_roles.yaml"))
    orchestrator = RagnarOrchestrator(
        registry,
        config_path=Path("ragnar/ragnar.yaml"),
        prepare_worktrees=False,
        record_runs=False,
    )
    orchestrator.role_runtime = _ScriptedRoleRuntime(
        {
            ("conductor", "review_delivery_plan"): [
                '{"verdict":"revise","feedback":"needs more detail","rework_roles":[]}',
            ],
        }
    )

    state = orchestrator.invoke("fix the login bug")

    expected_cap = orchestrator.config.max_plan_revisions(default=MAX_PLAN_REVISIONS)
    assert state["plan_revision_count"] == expected_cap
    assert state["plan_review_verdict"] == "escalated_to_owner"
    plan_artifacts = [a for a in state["artifacts"] if a["kind"] == "architecture_plan"]
    assert len(plan_artifacts) == expected_cap + 1
    assert state.get("final_report")


def test_qa_revision_loop_routes_back_to_named_role() -> None:
    registry = load_role_registry(Path("ragnar/roles/ragnar_roles.yaml"))
    orchestrator = RagnarOrchestrator(
        registry,
        config_path=Path("ragnar/ragnar.yaml"),
        prepare_worktrees=False,
        record_runs=False,
        qa_commands=[[sys.executable, "-m", "pytest", "-q", "/nonexistent-test-file-xyz.py"]],
    )
    orchestrator.role_runtime = _ScriptedRoleRuntime(
        {
            ("conductor", "review_qa_verdict"): [
                '{"verdict":"revise","feedback":"fix the auth check","rework_roles":["backend_engineer"]}',
                '{"verdict":"approve","feedback":"looks good","rework_roles":[]}',
            ],
        }
    )

    state = orchestrator.invoke("fix the login bug")

    backend_packets = [a for a in state["artifacts"] if a["kind"] == "backend_work_packet"]
    assert len(backend_packets) == 2
    rework_feedback = backend_packets[1]["body"]["agent_invocation"]["rework_feedback"]
    assert rework_feedback["conductor_feedback"] == "fix the auth check"
    assert rework_feedback["qa_commands"][0]["exit_code"] != 0
    assert state["qa_review_verdict"] == "approve"


def test_final_report_includes_owner_briefing_offline() -> None:
    state = run_objective("fix the login bug", checkpoint_db=None)

    assert isinstance(state.get("owner_briefing"), str) and state["owner_briefing"]
    parsed = json.loads(state["final_report"])
    assert parsed["run_id"] == state["run_id"]
    assert set(parsed.keys()) == {
        "run_id",
        "objective",
        "phase",
        "selected_build_roles",
        "artifact_count",
        "blocked",
        "approval_requests",
    }


class _FakeAssistantMessage:
    message_type = "assistant_message"

    def __init__(self, content: str) -> None:
        self.id = "msg-assistant-1"
        self.content = content
        self.date = "2024-01-01T00:00:00"
        self.sender_id = "agent-under-test"


class _FakeToolCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCallMessage:
    message_type = "tool_call_message"

    def __init__(self, tool_call: _FakeToolCall) -> None:
        self.id = "msg-tool-call-1"
        self.date = "2024-01-01T00:00:01"
        self.sender_id = "agent-under-test"
        self.tool_call = tool_call


class _FakeToolReturnMessage:
    message_type = "tool_return_message"

    def __init__(self, tool_return: str, is_err: bool = False) -> None:
        self.id = "msg-tool-return-1"
        self.date = "2024-01-01T00:00:02"
        self.sender_id = "agent-under-test"
        self.tool_return = tool_return
        self.is_err = is_err


class _FakeResponse:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages


class _FakeMessagesAPI:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return self.response


class _FakeAgentsAPI:
    def __init__(self, response: _FakeResponse) -> None:
        self.messages = _FakeMessagesAPI(response)


class _FakeLettaClient:
    def __init__(self, response: _FakeResponse) -> None:
        self.agents = _FakeAgentsAPI(response)


def _build_test_invocation(role_id: str):
    registry = load_role_registry(Path("ragnar/roles/ragnar_roles.yaml"))
    role = registry.get(role_id)
    return build_invocation_contract(
        run_id="run-test",
        objective="test objective",
        action="draft_migrations",
        role=role,
        model=ModelConfig(provider="openrouter", model="openrouter/x/y", temperature=0.1, max_tokens=100),
        workspace={"available": False},
        policy=None,
        memory_context=[],
    )


def test_call_agent_sets_max_steps_and_peer_block_when_enabled() -> None:
    reply_json = '{"status":"completed","summary":"ok","proposed_patches":[],"handoffs":[],"memory_writebacks":[],"proposed_actions":[]}'
    fake_client = _FakeLettaClient(
        _FakeResponse(
            [
                _FakeToolCallMessage(_FakeToolCall("send_message_to_agents_matching_tags", '{"match_some": ["role:qa_engineer"]}')),
                _FakeToolReturnMessage('{"agent_id": "agent-qa", "response": ["ack"]}'),
                _FakeAssistantMessage(reply_json),
            ]
        )
    )
    runtime = RoleRuntime(
        Path("ragnar/.ragnar/letta_agents.json"),
        mode="letta",
        enable_agent_messaging=True,
        agent_max_steps=7,
    )
    runtime._client = fake_client

    invocation = _build_test_invocation("backend_engineer")
    raw_reply, transcript = runtime._call_agent("agent-under-test", invocation)

    call_kwargs = fake_client.agents.messages.calls[0]
    assert call_kwargs["max_steps"] == 7
    prompt_content = call_kwargs["messages"][0]["content"]
    assert "send_message_to_agents_matching_tags" in prompt_content
    assert "role:qa_engineer" in prompt_content
    assert raw_reply == reply_json
    assert transcript["messages"][0]["message_type"] == "tool_call_message"
    assert transcript["messages"][0]["tool_name"] == "send_message_to_agents_matching_tags"
    assert transcript["messages"][2]["message_type"] == "assistant_message"


def test_call_agent_omits_peer_block_when_disabled() -> None:
    reply_json = '{"status":"completed","summary":"ok","proposed_patches":[],"handoffs":[],"memory_writebacks":[],"proposed_actions":[]}'
    fake_client = _FakeLettaClient(_FakeResponse([_FakeAssistantMessage(reply_json)]))
    runtime = RoleRuntime(
        Path("ragnar/.ragnar/letta_agents.json"),
        mode="letta",
        enable_agent_messaging=False,
        agent_max_steps=12,
    )
    runtime._client = fake_client

    invocation = _build_test_invocation("backend_engineer")
    runtime._call_agent("agent-under-test", invocation)

    call_kwargs = fake_client.agents.messages.calls[0]
    assert call_kwargs["max_steps"] == 12
    prompt_content = call_kwargs["messages"][0]["content"]
    assert "send_message_to_agents_matching_tags" not in prompt_content


def test_agent_transcript_store_appends_and_filters_by_run_id(tmp_path: Path) -> None:
    store = AgentTranscriptStore(tmp_path / "agent_transcripts.jsonl")
    store.append("run-1", "backend_engineer", "draft_migrations", {"agent_id": "a1", "max_steps": 7, "messages": []})
    store.append("run-2", "qa_engineer", "produce_qa_verdict", {"agent_id": "a2", "max_steps": 7, "messages": []})

    all_records = store.list()
    assert len(all_records) == 2
    run1_records = store.list(run_id="run-1")
    assert len(run1_records) == 1
    assert run1_records[0]["role_id"] == "backend_engineer"


def test_revision_caps_configurable_via_ragnar_config() -> None:
    config = RagnarConfig(version="0.1", raw={"execution": {"max_plan_revisions": 1, "max_qa_revisions": 3}})

    assert config.max_plan_revisions(default=MAX_PLAN_REVISIONS) == 1
    assert config.max_qa_revisions(default=2) == 3

    default_config = RagnarConfig(version="0.1", raw={})
    assert default_config.max_plan_revisions(default=MAX_PLAN_REVISIONS) == MAX_PLAN_REVISIONS
    assert default_config.max_qa_revisions(default=2) == 2


def test_letta_role_memory_defines_domain_stack_agnostic_operating_model() -> None:
    registry = load_role_registry(Path("ragnar/roles/ragnar_roles.yaml"))
    role = registry.get("backend_engineer")

    blocks = {block["label"]: block for block in _memory_blocks(role)}

    operating_model = blocks["domain_stack_operating_model"]
    assert operating_model["read_only"] is True
    assert operating_model["metadata"]["definition_version"] == ROLE_AGENT_DEFINITION_VERSION
    assert "Never assume the repo is Python" in operating_model["value"]
    assert "project_profile" in operating_model["value"]
    assert "language:<name>" in operating_model["value"]

    persona = blocks["persona"]["value"]
    assert "Follow the read-only domain_stack_operating_model memory block" in persona


class _FakeBlockResponse:
    def __init__(self, block_id: str) -> None:
        self.id = block_id


class _FakeAgentBlocksAPI:
    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.attachments: list[dict[str, str]] = []
        self.fail_missing_labels: set[str] = set()

    def update(self, label: str, *, agent_id: str, **kwargs: Any) -> None:
        if label in self.fail_missing_labels:
            raise RuntimeError("404 block not found")
        self.updates.append({"label": label, "agent_id": agent_id, **kwargs})

    def attach(self, block_id: str, *, agent_id: str) -> None:
        self.attachments.append({"block_id": block_id, "agent_id": agent_id})


class _FakeBlocksAPI:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeBlockResponse:
        self.created.append(kwargs)
        return _FakeBlockResponse(f"block-{kwargs['label']}")


class _FakeProvisionAgentsAPI:
    def __init__(self) -> None:
        self.blocks = _FakeAgentBlocksAPI()


class _FakeProvisionClient:
    def __init__(self) -> None:
        self.agents = _FakeProvisionAgentsAPI()
        self.blocks = _FakeBlocksAPI()


def test_existing_letta_agent_definition_blocks_are_synced() -> None:
    registry = load_role_registry(Path("ragnar/roles/ragnar_roles.yaml"))
    role = registry.get("frontend_engineer")
    client = _FakeProvisionClient()
    client.agents.blocks.fail_missing_labels = {"domain_stack_operating_model"}

    _sync_agent_definition_blocks(client, "agent-frontend", role)

    updated_labels = {item["label"] for item in client.agents.blocks.updates}
    assert {"persona", "role_contract", "memory_scope"}.issubset(updated_labels)
    assert client.blocks.created[0]["label"] == "domain_stack_operating_model"
    assert client.blocks.created[0]["read_only"] is True
    assert client.agents.blocks.attachments == [
        {"block_id": "block-domain_stack_operating_model", "agent_id": "agent-frontend"}
    ]


def test_build_project_profile_detects_python(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\ndescription = "A test service"\n', encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("pytest\n", encoding="utf-8")

    profile = build_project_profile(tmp_path).to_dict()

    assert profile["languages"] == ["python"]
    assert profile["package_managers"] == ["pip"]
    assert profile["test_commands"] == ["pytest"]
    assert "A test service" in profile["domain_hints"]
    assert profile["confidence"]["languages"] == 1.0


def test_build_project_profile_detects_node(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        json.dumps({"description": "A frontend app", "scripts": {"test": "jest"}, "dependencies": {"react": "^18.0.0"}}),
        encoding="utf-8",
    )
    (tmp_path / "package-lock.json").write_text("{}", encoding="utf-8")

    profile = build_project_profile(tmp_path).to_dict()

    assert profile["languages"] == ["javascript"]
    assert profile["package_managers"] == ["npm"]
    assert profile["frameworks"] == ["react"]
    assert profile["test_commands"] == ["npm test"]
    assert "A frontend app" in profile["domain_hints"]


def test_build_project_profile_empty_repo_has_no_confident_detection(tmp_path: Path) -> None:
    profile = build_project_profile(tmp_path).to_dict()

    assert profile["languages"] == []
    assert profile["confidence"] == {}


def test_build_project_profile_detects_nested_monorepo_project(tmp_path: Path) -> None:
    nested = tmp_path / "app"
    nested.mkdir()
    (nested / "go.mod").write_text("module example.com/app\n", encoding="utf-8")

    profile = build_project_profile(tmp_path).to_dict()

    assert profile["languages"] == ["go"]
    assert profile["confidence"]["languages"] == 0.7


def test_qa_commands_from_profile_only_maps_allowlisted_strings() -> None:
    commands = qa_commands_from_profile({"test_commands": ["pytest", "rm -rf /", "go test ./..."]})

    assert [sys.executable, "-m", "pytest"] in commands
    assert ["go", "test", "./..."] in commands
    assert not any("rm" in " ".join(command) for command in commands)


def test_all_profile_qa_commands_are_allowed_by_qa_policy() -> None:
    qa_policy = default_workspace_policies()["qa_engineer"]
    commands = qa_commands_from_profile(
        {
            "test_commands": [
                "pytest",
                "python -m unittest",
                "npm test",
                "yarn test",
                "pnpm test",
                "go test ./...",
                "cargo test",
                "mvn test",
                "gradle test",
                "bundle exec rspec",
                "rake test",
                "composer test",
            ]
        }
    )

    denied = [command for command in commands if command_family(command) not in qa_policy.allowed_command_families]

    assert denied == []


def test_project_profiler_runs_before_triage_and_populates_state(tmp_path: Path) -> None:
    # Deliberately no "scripts" key -- a "test" script would make qa_gate's
    # profile-derived fallback later in this same run actually invoke a real
    # `npm test` subprocess, which this test isn't meant to exercise.
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "package.json").write_text(json.dumps({"name": "fixture-app"}), encoding="utf-8")

    registry = load_role_registry(Path("ragnar/roles/ragnar_roles.yaml"))
    orchestrator = RagnarOrchestrator(
        registry,
        config_path=Path("ragnar/ragnar.yaml"),
        prepare_worktrees=False,
        record_runs=False,
    )
    orchestrator.workspaces = RoleWorkspaceManager(tmp_path, enabled=False)

    state = orchestrator.invoke("fix the login bug")

    assert state["project_profile"]["languages"] == ["javascript"]
    artifact_kinds = [artifact["kind"] for artifact in state["artifacts"]]
    assert artifact_kinds.index("project_profile") < artifact_kinds.index("conductor_triage")


class _FakeExecutionAdapter:
    """Records commands instead of ever spawning a real subprocess.

    A real LocalExecutionAdapter running a profile-discovered "pytest" command
    would invoke sys.executable -m pytest -- the same interpreter already
    running this test suite -- which risks the child recursively re-collecting
    and re-running the whole suite. This fake proves the wiring (discovery ->
    policy gate -> execution.run) without ever spawning anything real.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, command: list[str]) -> Any:
        self.calls.append(command)
        return CommandResult(command=command, cwd=".", exit_code=0, stdout="1 passed", stderr="", timed_out=False)


def test_qa_profile_discovery_is_off_by_default() -> None:
    assert RagnarConfig(version="0.1", raw={}).enable_qa_profile_discovery() is False


def test_qa_gate_does_not_run_commands_from_profile_when_discovery_disabled() -> None:
    registry = load_role_registry(Path("ragnar/roles/ragnar_roles.yaml"))
    orchestrator = RagnarOrchestrator(
        registry,
        config_path=Path("ragnar/ragnar.yaml"),
        prepare_worktrees=False,
        record_runs=False,
    )
    fake_execution = _FakeExecutionAdapter()
    orchestrator.execution = fake_execution
    assert orchestrator.config.enable_qa_profile_discovery() is False

    state: Any = {
        "run_id": "run-qa-discovery-off",
        "objective": "fix the login bug",
        "qa_commands": [],
        "project_profile": {"test_commands": ["pytest"]},
        "artifacts": [
            {"kind": "backend_work_packet", "owner_role": "backend_engineer", "created_at": "now", "body": {}}
        ],
    }
    result = orchestrator.qa_gate(state)

    assert fake_execution.calls == []
    qa_artifact = result["artifacts"][0]
    assert qa_artifact["body"]["verdict"] == "pass_with_warnings"


def test_qa_gate_discovers_and_runs_command_from_project_profile() -> None:
    registry = load_role_registry(Path("ragnar/roles/ragnar_roles.yaml"))
    orchestrator = RagnarOrchestrator(
        registry,
        config_path=Path("ragnar/ragnar.yaml"),
        prepare_worktrees=False,
        record_runs=False,
    )
    fake_execution = _FakeExecutionAdapter()
    orchestrator.execution = fake_execution
    # enable_qa_profile_discovery defaults to false in ragnar.yaml precisely
    # because auto-running discovered commands is unsafe by default (see the
    # comment in ragnar.yaml) -- opt in explicitly for this test.
    orchestrator.config = RagnarConfig(
        version=orchestrator.config.version,
        raw={**orchestrator.config.raw, "execution": {**orchestrator.config.raw.get("execution", {}), "enable_qa_profile_discovery": True}},
    )

    state: Any = {
        "run_id": "run-qa-discovery",
        "objective": "fix the login bug",
        "qa_commands": [],
        "project_profile": {"test_commands": ["pytest"]},
        "artifacts": [
            {"kind": "backend_work_packet", "owner_role": "backend_engineer", "created_at": "now", "body": {}}
        ],
    }
    result = orchestrator.qa_gate(state)

    assert fake_execution.calls == [[sys.executable, "-m", "pytest"]]
    qa_artifact = result["artifacts"][0]
    assert qa_artifact["body"]["verdict"] == "pass"
    assert "discovered from the project profile" in " ".join(qa_artifact["body"]["warnings"])
