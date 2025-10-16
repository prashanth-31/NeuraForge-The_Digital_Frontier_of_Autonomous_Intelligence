#!/usr/bin/env python3
"""Utility to validate that the NeuraForge stack is running end-to-end.

This script performs the operational checks called out in the phase 7
verification checklist. It exercises the FastAPI surface, reviewer APIs,
streaming endpoint, and observability targets so operators can confirm a
healthy deployment from a single command.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import httpx

DEFAULT_PROMPT = "Stack verification ping"
DEFAULT_TIMEOUT = 30.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str

    def format(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


def _normalize_base(url: str) -> str:
    url = url.strip()
    if url.endswith("/"):
        return url[:-1]
    return url


async def check_health(client: httpx.AsyncClient) -> CheckResult:
    try:
        response = await client.get("/health", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status") if isinstance(payload, dict) else None
        if status == "ok":
            return CheckResult("Backend health", True, "GET /health returned status=ok")
        return CheckResult("Backend health", False, f"Unexpected payload: {payload!r}")
    except Exception as exc:  # pragma: no cover - network dependent
        return CheckResult("Backend health", False, f"{exc}")


async def check_reviews(client: httpx.AsyncClient, token: Optional[str]) -> CheckResult:
    if not token:
        return CheckResult(
            "Reviewer API",
            False,
            "Missing reviewer token. Provide --review-token or set REVIEW_API_TOKEN",
        )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    try:
        response = await client.get("/api/v1/reviews", headers=headers, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return CheckResult("Reviewer API", True, f"Fetched {len(payload)} ticket(s)")
        return CheckResult("Reviewer API", False, f"Unexpected payload: {payload!r}")
    except httpx.HTTPStatusError as exc:
        return CheckResult("Reviewer API", False, f"{exc.response.status_code} {exc.response.text}")
    except Exception as exc:  # pragma: no cover - network dependent
        return CheckResult("Reviewer API", False, str(exc))


async def _stream_task(
    client: httpx.AsyncClient,
    prompt: str,
    metadata: Optional[dict[str, Any]] = None,
) -> tuple[CheckResult, Optional[str]]:
    payload = {"prompt": prompt, "metadata": metadata or {"source": "stack_verifier"}}
    try:
        async with client.stream(
            "POST",
            "/api/v1/submit_task/stream",
            json=payload,
            timeout=None,
        ) as stream:
            if stream.status_code != 200:
                text = await stream.aread()
                return (
                    CheckResult(
                        "Task stream",
                        False,
                        f"HTTP {stream.status_code}: {text.decode('utf-8', errors='replace')}",
                    ),
                    None,
                )

            buffer = ""
            event_name: Optional[str] = None
            event_data: Optional[dict[str, Any]] = None
            captured_events: list[str] = []
            task_id: Optional[str] = None

            async for chunk in stream.aiter_bytes():  # pragma: no branch - depends on server
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    if not block.strip():
                        event_name = None
                        event_data = None
                        continue
                    for line in block.split("\n"):
                        if line.startswith("event:"):
                            event_name = line[len("event:") :].strip()
                        elif line.startswith("data:"):
                            data_raw = line[len("data:") :].strip()
                            try:
                                event_data = json.loads(data_raw)
                            except json.JSONDecodeError:
                                event_data = None
                    if event_name and event_data:
                        captured_events.append(event_name)
                        if event_name == "task_started" and isinstance(event_data, dict):
                            task_id = str(event_data.get("task_id")) if event_data.get("task_id") else task_id
                        if event_name in {"task_completed", "task_failed"}:
                            # We received a terminal event; capture detail and finish.
                            detail = event_data.get("status") or event_data.get("error") if isinstance(event_data, dict) else None
                            return (
                                CheckResult(
                                    "Task stream",
                                    True,
                                    f"Received events: {', '.join(captured_events)} (detail={detail})",
                                ),
                                task_id,
                            )
                        if len(captured_events) >= 5:
                            return (
                                CheckResult(
                                    "Task stream",
                                    True,
                                    f"Received events: {', '.join(captured_events)}",
                                ),
                                task_id,
                            )
            return (
                CheckResult("Task stream", False, "Stream ended without receiving telemetry"), task_id)
    except Exception as exc:  # pragma: no cover - network dependent
        return CheckResult("Task stream", False, str(exc)), None


async def check_task_status(
    client: httpx.AsyncClient,
    task_id: Optional[str],
) -> CheckResult:
    if not task_id:
        return CheckResult("Task status", False, "Task id unavailable from stream")
    try:
        response = await client.get(f"/api/v1/tasks/{task_id}", timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        status = payload.get("status") if isinstance(payload, dict) else None
        if status:
            metrics = payload.get("metrics") if isinstance(payload, dict) else {}
            guardrails = payload.get("guardrails") if isinstance(payload, dict) else {}
            detail = textwrap.shorten(
                f"status={status} metrics={metrics} guardrails={bool(guardrails)}",
                width=120,
                placeholder="…",
            )
            return CheckResult("Task status", True, detail)
        return CheckResult("Task status", False, f"Unexpected payload: {payload!r}")
    except httpx.HTTPStatusError as exc:
        return CheckResult(
            "Task status",
            False,
            f"{exc.response.status_code} while fetching /api/v1/tasks/{task_id}",
        )
    except Exception as exc:  # pragma: no cover - network dependent
        return CheckResult("Task status", False, str(exc))


async def check_prometheus(prometheus_url: Optional[str]) -> CheckResult:
    if not prometheus_url:
        return CheckResult("Prometheus", False, "Prometheus URL not provided")
    try:
        base = _normalize_base(prometheus_url)
        async with httpx.AsyncClient(base_url=base) as client:
            response = await client.get("/-/ready", timeout=DEFAULT_TIMEOUT)
            if response.status_code == 200:
                return CheckResult("Prometheus", True, "Prometheus ready endpoint responded OK")
            return CheckResult("Prometheus", False, f"Unexpected status code {response.status_code}")
    except Exception as exc:  # pragma: no cover - network dependent
        return CheckResult("Prometheus", False, str(exc))


async def check_grafana(grafana_url: Optional[str]) -> CheckResult:
    if not grafana_url:
        return CheckResult("Grafana", False, "Grafana URL not provided")
    try:
        base = _normalize_base(grafana_url)
        async with httpx.AsyncClient(base_url=base) as client:
            response = await client.get("/api/health", timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("database") == "ok":
                return CheckResult("Grafana", True, "Grafana health OK")
            return CheckResult("Grafana", False, f"Unexpected payload: {payload!r}")
    except httpx.HTTPStatusError as exc:
        return CheckResult("Grafana", False, f"HTTP {exc.response.status_code}: {exc.response.text}")
    except Exception as exc:  # pragma: no cover - network dependent
        return CheckResult("Grafana", False, str(exc))


async def run_checks(args: argparse.Namespace) -> list[CheckResult]:
    base_url = _normalize_base(args.backend_url)
    async with httpx.AsyncClient(base_url=base_url) as client:
        results: list[CheckResult] = []
        results.append(await check_health(client))
        results.append(await check_reviews(client, args.review_token))
        stream_result, task_id = await _stream_task(client, args.prompt, args.metadata)
        results.append(stream_result)
        status_result = await check_task_status(client, task_id)
        results.append(status_result)
    # Prometheus and Grafana run in their own clients to avoid base_url coupling.
    results.append(await check_prometheus(args.prometheus_url))
    results.append(await check_grafana(args.grafana_url))
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that the NeuraForge stack is running and healthy.",
    )
    parser.add_argument(
        "--backend-url",
        default="http://localhost:8000",
        help="Base URL for the FastAPI backend (default: %(default)s)",
    )
    parser.add_argument(
        "--prometheus-url",
        default="http://localhost:9090",
        help="Prometheus base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--grafana-url",
        default="http://localhost:3000",
        help="Grafana base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--review-token",
        default=None,
        help="Bearer token for reviewer endpoints. Falls back to REVIEW_API_TOKEN env var.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt to submit when exercising the streaming endpoint.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        type=json.loads,
        help="Optional JSON metadata to include with the task submission.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit as soon as any check fails.",
    )
    return parser


def _resolve_token(cli_token: Optional[str]) -> Optional[str]:
    if cli_token:
        return cli_token
    return os.environ.get("REVIEW_API_TOKEN")


def _print_summary(results: Iterable[CheckResult]) -> None:
    print("\nVerification summary:\n" + "-" * 80)
    for result in results:
        print(result.format())


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.review_token = _resolve_token(args.review_token)

    if args.metadata is not None and not isinstance(args.metadata, dict):
        print("--metadata must be a JSON object", file=sys.stderr)
        return 2

    try:
        results = asyncio.run(run_checks(args))
    except KeyboardInterrupt:  # pragma: no cover - operator convenience
        print("Verification aborted by user", file=sys.stderr)
        return 130

    _print_summary(results)

    failures = [result for result in results if not result.ok]
    if failures:
        if args.fail_fast:
            first = failures[0]
            print(f"\nFirst failure: {first.name} -> {first.detail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
