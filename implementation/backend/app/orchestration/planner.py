from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any, Iterable, Mapping, Sequence

from ..agents.contracts import list_contracts
from ..schemas.agents import AgentCapability
from ..core.config import PlanningSettings


@dataclass(slots=True)
class PlannedStep:
    step_id: str
    title: str
    agent: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    eta_iso: str | None = None
    deadline_iso: str | None = None
    priority: int = 0
    retry_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskPlan:
    task_id: str
    summary: str
    steps: list[PlannedStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskPlanner:
    async def build_plan(
        self,
        *,
        task: dict[str, Any],
        outputs: Iterable[dict[str, Any]],
        negotiation: dict[str, Any] | None,
    ) -> TaskPlan | None:
        raise NotImplementedError


class SimpleTaskPlanner(TaskPlanner):
    def __init__(self, *, default_step_prefix: str = "step") -> None:
        self._default_step_prefix = default_step_prefix

    async def build_plan(
        self,
        *,
        task: dict[str, Any],
        outputs: Iterable[dict[str, Any]],
        negotiation: dict[str, Any] | None,
    ) -> TaskPlan | None:
        task_id = str(task.get("id") or task.get("task_id") or "")
        if not task_id:
            return None

        summary = "Orchestrated task plan"
        if negotiation and negotiation.get("outcome"):
            summary = str(negotiation["outcome"])

        steps: list[PlannedStep] = []
        for index, raw in enumerate(outputs):
            if not isinstance(raw, dict):
                continue
            agent_name = str(raw.get("agent", "unknown"))
            description = str(raw.get("summary") or raw.get("content") or "").strip()
            if not description:
                continue
            step_id = f"{task_id}-{self._default_step_prefix}-{index + 1}"
            depends_on = [steps[-1].step_id] if steps else []
            raw_metadata = raw.get("metadata")
            metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
            steps.append(
                PlannedStep(
                    step_id=step_id,
                    title=f"{agent_name} deliverable",
                    agent=agent_name,
                    description=description,
                    depends_on=list(depends_on),
                    metadata=metadata,
                )
            )

        if not steps:
            return None

        plan_metadata: dict[str, Any] = {
            "strategy": "simple",
            "negotiation": negotiation or {},
        }
        return TaskPlan(task_id=task_id, summary=summary, steps=steps, metadata=plan_metadata)


def clone_plan_with_schedule(plan: TaskPlan, *, timestamps: list[str]) -> TaskPlan:
    if len(timestamps) != len(plan.steps):
        raise ValueError("Timestamp list must align with plan steps")
    scheduled_steps = [replace(step, eta_iso=ts) for step, ts in zip(plan.steps, timestamps)]
    return TaskPlan(task_id=plan.task_id, summary=plan.summary, steps=scheduled_steps, metadata=dict(plan.metadata))


def _build_capability_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for contract in list_contracts():
        index[contract.capability.value] = contract.name
    return index


class DependencyTaskPlanner(TaskPlanner):
    def __init__(
        self,
        *,
        settings: PlanningSettings,
        agent_capability_index: Mapping[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._agent_capability_index = dict(agent_capability_index or _build_capability_index())

    async def build_plan(
        self,
        *,
        task: dict[str, Any],
        outputs: Iterable[dict[str, Any]],
        negotiation: dict[str, Any] | None,
    ) -> TaskPlan | None:
        task_id = str(task.get("id") or task.get("task_id") or "")
        if not task_id:
            return None

        raw_actions = self._collect_actions(task=task, outputs=outputs, negotiation=negotiation)
        if not raw_actions:
            return None

        limited_actions = raw_actions[: self._settings.max_subtasks]
        steps: list[PlannedStep] = []
        dependency_graph: dict[str, set[str]] = {}
        for index, action in enumerate(limited_actions, start=1):
            step_id = action.get("id") or f"{task_id}-step-{index}"
            title = action.get("title") or action.get("summary") or f"Step {index}"

            description = action.get("description") or action.get("detail") or action.get("summary") or ""
            if not description:
                continue

            capability = self._normalize_capability(action.get("capability") or action.get("agent_capability"))
            agent = self._resolve_agent(capability, fallback=action.get("agent"))
            depends = self._normalize_dependencies(action.get("depends_on"), steps)
            dependency_graph[step_id] = set(depends)

            priority = int(action.get("priority") or 0)
            retry_policy = self._build_retry_policy(action)
            deadline_minutes = self._resolve_deadline_minutes(action)
            metadata = {
                "source": action.get("source") or "orchestrator",
                "capability": capability,
                "original": action,
            }

            steps.append(
                PlannedStep(
                    step_id=step_id,
                    title=title,
                    agent=agent,
                    description=description,
                    depends_on=depends,
                    metadata=metadata,
                    priority=priority,
                    retry_policy=retry_policy,
                    deadline_iso=self._compute_deadline_iso(deadline_minutes),
                )
            )

        if not steps:
            return None

        ordered = self._topological_sort(steps, dependency_graph)
        metadata = {
            "strategy": "dependency",
            "max_subtasks": self._settings.max_subtasks,
            "dependency_levels": self._dependency_levels(dependency_graph),
        }
        summary = negotiation.get("outcome") if negotiation else task.get("prompt", "Task plan")
        return TaskPlan(task_id=task_id, summary=str(summary or "Task plan"), steps=ordered, metadata=metadata)

    def _collect_actions(
        self,
        *,
        task: dict[str, Any],
        outputs: Iterable[dict[str, Any]],
        negotiation: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        metadata = task.get("metadata")
        if isinstance(metadata, dict):
            actions.extend(self._extract_actions(metadata.get("subtasks")))
            actions.extend(self._extract_actions(metadata.get("actions")))

        if negotiation:
            negotiation_meta = negotiation.get("metadata")
            if isinstance(negotiation_meta, dict):
                actions.extend(self._extract_actions(negotiation_meta.get("recommended_actions")))
                actions.extend(self._extract_actions(negotiation_meta.get("actions")))

        for output in outputs:
            if not isinstance(output, dict):
                continue
            meta = output.get("metadata")
            if isinstance(meta, dict):
                actions.extend(self._extract_actions(meta.get("plan")))
                actions.extend(self._extract_actions(meta.get("actions")))
                actions.extend(self._extract_actions(meta.get("subtasks")))
            summary = output.get("summary")
            if summary and not meta:
                actions.append({
                    "title": f"Review {output.get('agent', 'agent')} output",
                    "description": summary,
                    "agent": output.get("agent"),
                    "source": "agent_summary",
                })

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for action in actions:
            key = action.get("title") or action.get("summary") or action.get("description")
            if not key:
                continue
            normalized = key.strip().lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(action)
        return deduped

    def _extract_actions(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _normalize_capability(self, value: Any) -> str | None:
        if isinstance(value, AgentCapability):
            return value.value
        if isinstance(value, str):
            lower = value.strip().lower()
            for capability in AgentCapability:
                if capability.value == lower:
                    return capability.value
        return None

    def _resolve_agent(self, capability: str | None, fallback: Any) -> str:
        if capability and capability in self._agent_capability_index:
            return self._agent_capability_index[capability]
        if isinstance(fallback, str) and fallback:
            return fallback
        # round robin fallback if strategy demands
        if self._settings.assignment_strategy == "round_robin" and self._agent_capability_index:
            keys = sorted(self._agent_capability_index.values())
            return keys[0]
        return next(iter(self._agent_capability_index.values()), "orchestrator")

    def _normalize_dependencies(self, value: Any, existing: Sequence[PlannedStep]) -> list[str]:
        if isinstance(value, list):
            deps = [str(item) for item in value if isinstance(item, (str, int))]
            depth_limit = self._settings.max_dependency_depth
            if len(deps) > depth_limit:
                return deps[:depth_limit]
            return deps
        if existing:
            # default to sequential dependency
            return [existing[-1].step_id]
        return []

    def _build_retry_policy(self, action: dict[str, Any]) -> dict[str, Any]:
        policy = action.get("retry_policy")
        if isinstance(policy, dict):
            return policy
        attempts = action.get("max_attempts")
        backoff = action.get("backoff_seconds")
        multiplier = action.get("backoff_multiplier")
        result: dict[str, Any] = {}
        if attempts is not None:
            result["max_attempts"] = int(attempts)
        if backoff is not None:
            result["backoff_seconds"] = float(backoff)
        if multiplier is not None:
            result["backoff_multiplier"] = float(multiplier)
        return result

    def _resolve_deadline_minutes(self, action: dict[str, Any]) -> int:
        value = action.get("deadline_minutes") or action.get("duration_minutes")
        if isinstance(value, (int, float)):
            return max(1, int(value))
        return self._settings.default_step_duration_minutes

    def _compute_deadline_iso(self, minutes: int) -> str | None:
        if minutes <= 0:
            return None
        deadline = timedelta(minutes=minutes)
        return f"+{int(deadline.total_seconds())}s"

    def _topological_sort(
        self,
        steps: list[PlannedStep],
        dependency_graph: Mapping[str, set[str]],
    ) -> list[PlannedStep]:
        step_map = {step.step_id: step for step in steps}
        indegree: dict[str, int] = {step.step_id: 0 for step in steps}
        adjacency: dict[str, set[str]] = {step.step_id: set() for step in steps}

        for node, deps in dependency_graph.items():
            for dep in deps:
                if dep in step_map:
                    indegree[node] = indegree.get(node, 0) + 1
                    adjacency.setdefault(dep, set()).add(node)

        zero_indegree = [step_id for step_id, deg in indegree.items() if deg == 0]
        ordered_ids: list[str] = []
        while zero_indegree:
            current = zero_indegree.pop(0)
            ordered_ids.append(current)
            for neighbor in adjacency.get(current, set()):
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    zero_indegree.append(neighbor)

        # Fallback for cycles: append remaining nodes in original order
        if len(ordered_ids) < len(steps):
            remaining = [step.step_id for step in steps if step.step_id not in ordered_ids]
            ordered_ids.extend(remaining)

        return [step_map[step_id] for step_id in ordered_ids if step_id in step_map]

    def _dependency_levels(self, dependency_graph: Mapping[str, set[str]]) -> dict[str, int]:
        levels: dict[str, int] = {}

        def compute(node: str) -> int:
            if node in levels:
                return levels[node]
            dependencies = dependency_graph.get(node, set())
            if not dependencies:
                levels[node] = 0
                return 0
            depth = 1 + max(compute(dep) for dep in dependencies)
            levels[node] = min(depth, self._settings.max_dependency_depth)
            return levels[node]

        for node in dependency_graph.keys():
            compute(node)
        return levels
