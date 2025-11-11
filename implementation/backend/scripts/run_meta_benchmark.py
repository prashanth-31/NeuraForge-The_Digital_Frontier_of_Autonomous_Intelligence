"""CLI for running meta-agent benchmarking scenarios using the simulation harness."""

from __future__ import annotations

import argparse
import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from app.agents.base import AgentContext
from app.agents.creative import CreativeAgent
from app.agents.enterprise import EnterpriseAgent
from app.agents.finance import FinanceAgent
from app.agents.general import GeneralistAgent
from app.agents.research import ResearchAgent
from app.api.routes import _build_orchestration_pipeline
from app.core.config import Settings, get_settings
from app.dependencies import get_review_manager_singleton
from app.orchestration.simulation import SimulationHarness, SimulationScenario, summarize_reports
from app.services.llm import LLMService
from app.services.memory import HybridMemoryService
from app.services.scoring import ConfidenceScorer


def _load_scenarios(path: Path | None, *, repetitions: int, concurrency: int) -> list[SimulationScenario]:
    if path is None:
        return [
            SimulationScenario(
                name="meta-escalation-demo",
                base_task={
                    "prompt": "Prepare launch readiness synopsis",
                    "metadata": {"priority": "high"},
                },
                variations=[
                    {"metadata": {"region": "na"}},
                    {"metadata": {"region": "eu"}},
                ],
                repetitions=repetitions,
                concurrency=concurrency,
                notes="Default benchmark scenario exercising negotiation and meta-agent synthesis.",
            )
        ]

    payload = json.loads(path.read_text(encoding="utf-8"))
    scenarios: list[SimulationScenario] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("Scenario configuration must be a list of objects")
        scenarios.append(
            SimulationScenario(
                name=item.get("name", "scenario"),
                base_task=item.get("base_task", {}),
                variations=item.get("variations", []),
                repetitions=item.get("repetitions", repetitions),
                concurrency=item.get("concurrency", concurrency),
                notes=item.get("notes"),
            )
        )
    return scenarios


async def _run_benchmark(settings: Settings, scenarios: list[SimulationScenario]) -> dict[str, Any]:
    review_manager = get_review_manager_singleton(settings)
    reports = []
    async with AsyncExitStack() as exit_stack:
        memory_service = HybridMemoryService.from_settings(settings)
        memory = await exit_stack.enter_async_context(memory_service.lifecycle())
        llm_service = LLMService.from_settings(settings)
        agents = [GeneralistAgent(), ResearchAgent(), FinanceAgent(), CreativeAgent(), EnterpriseAgent()]
        orchestrator, embedding_service, context_assembler = await _build_orchestration_pipeline(
            agents=agents,
            settings=settings,
            memory=memory,
            exit_stack=exit_stack,
            state_store=None,
            llm_service=llm_service,
            review_manager=review_manager,
        )
        if embedding_service is not None:
            exit_stack.push_async_callback(embedding_service.aclose)

        def context_factory(task: dict[str, Any]) -> AgentContext:
            return AgentContext(
                memory=memory,
                llm=llm_service,
                context=context_assembler,
                scorer=ConfidenceScorer(settings.scoring),
            )

        harness = SimulationHarness(orchestrator, context_factory=context_factory)
        for scenario in scenarios:
            reports.append(await harness.run(scenario))

    aggregated = summarize_reports(reports)
    aggregated["scenario_count"] = len(reports)
    aggregated["scenarios"] = [
        {
            "name": report.scenario.name,
            "success_rate": report.success_rate,
            "average_latency": report.average_latency,
            "escalation_rate": report.escalation_rate,
        }
        for report in reports
    ]
    return aggregated


def main() -> None:
    parser = argparse.ArgumentParser(description="Run meta-agent benchmarking scenarios")
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=None,
        help="Path to a JSON file containing simulation scenario definitions.",
    )
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions per scenario variation.")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent orchestrator runs per scenario.")
    args = parser.parse_args()

    settings = get_settings()
    scenarios = _load_scenarios(args.scenarios, repetitions=args.repetitions, concurrency=args.concurrency)
    results = asyncio.run(_run_benchmark(settings, scenarios))

    print("Meta-Agent Benchmark Results")
    scenario_details = results.pop("scenarios", [])
    for key, value in results.items():
        print(f"- {key.replace('_', ' ').title()}: {value}")
    if scenario_details:
        print("\nScenario Breakdown")
        for detail in scenario_details:
            print(
                f"  • {detail['name']}: success_rate={detail['success_rate']:.2f}, "
                f"avg_latency={detail['average_latency']:.2f}s, escalations={detail['escalation_rate']:.2f}"
            )


if __name__ == "__main__":
    main()