from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import urllib.request
import zipfile

REPO_OWNER = "prashanth-31"
REPO_NAME = "NeuraForge-The_Digital_Frontier_of_Autonomous_Intelligence"
WORKFLOW_FILE = "phase5-observability.yml"
API_VERSION = "2022-11-28"
USER_AGENT = "staging-sync-artifacts/1.0"

REQUIRED_ARTIFACTS = {
    "phase5-orchestrator-dashboard",
    "review-operations-dashboard",
    "task-guardrail-dashboard",
    "review-queue-dashboard",
    "k6-submit-task",
}

ROOT = Path(__file__).resolve().parents[2]


def _github_request(url: str, token: str) -> dict[str, object]:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
        "X-GitHub-Api-Version": API_VERSION,
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:  # noqa: S310 - GitHub API endpoint
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _download_zip(url: str, token: str) -> bytes:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": USER_AGENT,
    }
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:  # noqa: S310 - GitHub API endpoint
        return response.read()


def _extract_zip(data: bytes, repo_root: Path) -> list[Path]:
    extracted: list[Path] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "artifact.zip"
        tmp_path.write_bytes(data)
        with zipfile.ZipFile(tmp_path) as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                with zf.open(member) as src:
                    target = repo_root / member.filename
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("wb") as dst:
                        dst.write(src.read())
                    extracted.append(target)
    return extracted


def _sync_grafana_dashboards(repo_root: Path) -> None:
    dashboard_src = repo_root / "implementation" / "observability" / "dashboards"
    grafana_dir = repo_root / "implementation" / "observability" / "grafana" / "dashboards"
    grafana_dir.mkdir(parents=True, exist_ok=True)
    for json_path in dashboard_src.glob("*.json"):
        target = grafana_dir / json_path.name
        target.write_bytes(json_path.read_bytes())


def fetch_artifacts(branch: str, log: bool) -> None:
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN environment variable is required to download artifacts.")

    base = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}"  # noqa: E231
    runs_url = (
        f"{base}/actions/runs?workflow_id={WORKFLOW_FILE}&branch={branch}"  # noqa: E231
        "&status=success&per_page=1"
    )
    runs = _github_request(runs_url, token)
    workflow_runs = runs.get("workflow_runs", [])
    if not workflow_runs:
        raise SystemExit("No successful workflow runs found for the specified branch.")

    run = workflow_runs[0]
    artifacts_url = run.get("artifacts_url")
    if not isinstance(artifacts_url, str):
        raise SystemExit("Workflow run does not expose an artifacts_url")

    artifacts_payload = _github_request(artifacts_url, token)
    artifacts = artifacts_payload.get("artifacts", [])
    available = {artifact.get("name") for artifact in artifacts}
    missing = REQUIRED_ARTIFACTS - {name for name in available if isinstance(name, str)}
    if missing and log:
        print(f"Warning: missing artifacts {sorted(missing)}")

    matched = [artifact for artifact in artifacts if artifact.get("name") in REQUIRED_ARTIFACTS]
    if not matched:
        raise SystemExit("No matching artifacts found to download.")

    for artifact in matched:
        name = artifact.get("name")
        download_url = artifact.get("archive_download_url")
        if not isinstance(name, str) or not isinstance(download_url, str):
            continue
        if log:
            print(f"Downloading artifact '{name}'")
        zip_bytes = _download_zip(download_url, token)
        extracted = _extract_zip(zip_bytes, ROOT)
        if log:
            for path in extracted:
                print(f"  → {path.relative_to(ROOT)}")

    _sync_grafana_dashboards(ROOT)
    if log:
        print("Grafana dashboards synced under observability/grafana/dashboards")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync CI artifacts into staging assets")
    parser.add_argument("--branch", default="main", help="Branch to pull successful workflow artifacts from")
    parser.add_argument("--quiet", action="store_true", help="Silence progress output")
    args = parser.parse_args()

    try:
        fetch_artifacts(branch=args.branch, log=not args.quiet)
    except Exception as exc:  # pragma: no cover - CLI level error reporting
        if not args.quiet:
            print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no mutate
    main()
