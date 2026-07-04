from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .context_memory import ContextMemoryProvider
from .role_registry import RoleContract, RoleRegistry
from .run_ledger import RunLedgerStore


@dataclass(frozen=True)
class ContextBudget:
    max_total_hits: int = 5
    max_hits_per_role: int = 2
    max_hit_chars: int = 900
    max_handoffs: int = 4
    max_handoff_chars: int = 700


class ContextBroker:
    """Builds bounded, role-scoped context packets.

    The broker is deliberately deterministic. It prevents every role from
    receiving every memory layer and every previous artifact.
    """

    def __init__(
        self,
        registry: RoleRegistry,
        memory_provider: ContextMemoryProvider,
        run_ledger: RunLedgerStore,
        budget: ContextBudget | None = None,
    ) -> None:
        self.registry = registry
        self.memory_provider = memory_provider
        self.run_ledger = run_ledger
        self.budget = budget or ContextBudget()

    def retrieve_for_roles(self, objective: str, selected_roles: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        queries = self.context_queries(objective, selected_roles)
        lookups = self.memory_provider.retrieve(queries)
        return queries, self._bound_lookups(lookups)

    def context_queries(self, objective: str, selected_roles: list[str]) -> list[dict[str, Any]]:
        queries = [{"scope": "shared", "namespace": "project_context", "query": objective}]
        for role_id in selected_roles:
            role = self.registry.get(role_id)
            queries.append({"scope": "private", "namespace": role.private_memory_namespace, "query": objective})
            for namespace in role.memory.get("shared_namespaces", []):
                queries.append({"scope": "shared", "namespace": namespace, "query": objective})
        seen = set()
        deduped = []
        for query in queries:
            key = (query["scope"], query["namespace"], query["query"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(query)
        return deduped

    def role_context(self, state: dict[str, Any], role: RoleContract) -> list[dict[str, Any]]:
        namespaces = set(role.memory.get("shared_namespaces", []))
        namespaces.add(role.private_memory_namespace)
        role_context = []
        for item in state.get("memory_context", []):
            query = item.get("query", {})
            if query.get("namespace") in namespaces:
                role_context.append(item)
        ledger_hits = self.run_ledger.list(str(state["run_id"]))
        if ledger_hits:
            role_context.append(
                {
                    "scope": "run_ledger",
                    "namespace": "current_run",
                    "query": {"role_id": role.role_id},
                    "hits": self._ledger_hits_for_role(role, ledger_hits),
                    "errors": [],
                }
            )
        return role_context

    def handoff_inputs(self, state: dict[str, Any], target_role_id: str) -> list[dict[str, Any]]:
        inputs: list[dict[str, Any]] = []
        for artifact in state.get("artifacts", []):
            if not str(artifact.get("kind", "")).endswith("_work_packet"):
                continue
            body = artifact.get("body", {})
            handoffs = body.get("handoffs", [])
            sends_to_target = any(item.get("to_role") == target_role_id for item in handoffs)
            is_prior_work = artifact.get("owner_role") != target_role_id
            if not sends_to_target and not is_prior_work:
                continue
            result = body.get("agent_result", {}) or {}
            diff = body.get("diff") or {}
            summary = str(result.get("summary") or "")[: self.budget.max_handoff_chars]
            inputs.append(
                {
                    "from_role": artifact.get("owner_role"),
                    "artifact_kind": artifact.get("kind"),
                    "status": result.get("status"),
                    "summary": summary,
                    "changed_files": diff.get("changed_files", []),
                    "patch_count": len(result.get("proposed_patches", [])),
                }
            )
        return inputs[-self.budget.max_handoffs :]

    def already_done(self, state: dict[str, Any], role_id: str, action: str) -> dict[str, Any] | None:
        for artifact in reversed(state.get("artifacts", [])):
            body = artifact.get("body", {})
            result = body.get("agent_result", {}) or {}
            if artifact.get("owner_role") != role_id or body.get("allowed_action") != action:
                continue
            if result.get("status") in {"proposed", "completed"}:
                return {
                    "artifact_kind": artifact.get("kind"),
                    "summary": result.get("summary"),
                    "changed_files": (body.get("diff") or {}).get("changed_files", []),
                }
        record = self.run_ledger.latest_for(str(state["run_id"]), role_id, action)
        if record and record.get("status") in {"proposed", "completed"}:
            return record
        return None

    def _bound_lookups(self, lookups: list[dict[str, Any]]) -> list[dict[str, Any]]:
        remaining = self.budget.max_total_hits
        bounded = []
        for lookup in lookups:
            hits = []
            for hit in lookup.get("hits", [])[: self.budget.max_hits_per_role]:
                if remaining <= 0:
                    break
                trimmed = dict(hit)
                trimmed["text"] = str(trimmed.get("text", ""))[: self.budget.max_hit_chars]
                hits.append(trimmed)
                remaining -= 1
            bounded.append({**lookup, "hits": hits})
            if remaining <= 0:
                break
        return bounded

    def _ledger_hits_for_role(self, role: RoleContract, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        relevant = []
        receives_from = set(role.handoffs.get("receives_from", []))
        for record in records:
            if record.get("role_id") == role.role_id or record.get("role_id") in receives_from:
                relevant.append(record)
        # Bounded by count already (max_handoffs); also bound text length the same
        # way _bound_lookups does for memory-provider hits -- a run_ledger summary
        # can be up to 1000 chars (record_from_artifact's own cap) and was riding
        # along uncapped here. Fingerprint is a 64-char hash with no value to a
        # role's reasoning, so it's dropped rather than trimmed.
        return [
            {
                key: (str(value)[: self.budget.max_hit_chars] if key == "summary" else value)
                for key, value in record.items()
                if key != "fingerprint"
            }
            for record in relevant[-self.budget.max_handoffs :]
        ]
