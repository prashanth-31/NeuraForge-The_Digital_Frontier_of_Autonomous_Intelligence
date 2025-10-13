# MCP Category Onboarding Plan

This guide translates roadmap item **6.3 Category Onboarding Plan** into executable checklists. Use it to coordinate tool enablement across sprints and environments.

## Workflow Overview

1. **Catalog Registration** – ensure the MCP router exposes the tool with metadata, schemas, and health probes.
2. **Credential Provisioning** – inject secrets via environment (`TOOLS__MCP__…`) or secret manager references before enabling in production.
3. **Smoke Tests** – extend `tests/smoke/test_tools_catalog.py` (and tool-specific tests) to assert catalog visibility and basic invocation.
4. **Diagnostics Hookup** – confirm `/api/v1/diagnostics/mcp` reports the tool, including aliases and circuit state.
5. **Agent Enablement** – update agent playbooks and fallbacks once tooling is stable.

Track progress with the checklists below.

> Tip: run `scripts/register_research_tools.py` with staging/prod credentials to seed the catalog for research tools.

## Sprint 4A – Research Tooling

| Tool | Status | Checklist |
| --- | --- | --- |
| DuckDuckGo Search (`search/duckduckgo`) | ☑ | Alias wired in backend defaults → Register in MCP catalog → Configure OSS MCP adapter → Add smoke test coverage. |
| ArXiv Fetch (`research/arxiv`) | ☐ | Declare input schema (query, max_results) → Configure rate limits → Validate output schema. |
| Wikipedia Summaries (`research/wikipedia`) | ☐ | Attach anonymized usage headers → Ensure response localization handling → Update agent playbook with fallback. |
| Document Loader (`research/doc_loader`) | ☐ | Verify file size limits → Assert storage bucket permissions → Add deterministic fixture in tests. |
| Qdrant Retriever (`research/qdrant`) | ☐ | Configure Qdrant API key via secret → Confirm vector filters supported → Link to memory consolidation flow. |
| Summarizer (`research/summarizer`) | ☐ | Ensure timeout bounds → Emit latency histogram labels → Add regression test for max token handling. |

## Sprint 4B – Finance Tooling

| Tool | Status | Checklist |
| --- | --- | --- |
| Yahoo Finance (`finance/yfinance`) | ☐ | Supply API credentials → Validate caching behavior → Update smoke test assertions. |
| Pandas Analytics (`finance/pandas`) | ☐ | Lock dependency versions → Limit dataframe size → Document accepted operations. |
| Plotting (`finance/plot`) | ☐ | Define output MIME types (PNG/HTML) → Store artifacts in object storage → Add SSE telemetry for plot links. |
| CoinGecko News (`finance/coingecko_news`) | ☐ | Configure OSS adapter (free API) → Map categories to agent intents → Provide fallback to cached news. |
| CSV Analyzer (`finance/csv`) | ☐ | Enforce file validation → Sanitize output before returning → Record example inputs in docs. |
| FinBERT Sentiment (`finance/finbert`) | ☐ | Bundle model weights or endpoint URL → Calibrate confidence scaling → Add negative/positive sentiment fixtures. |

## Post-Sprint 4B – Creative & Enterprise

Maintain the same flow once Research/Finance stabilize. Suggested initial targets:

- **Creative**: Prompt styler, tone checker, whisper transcription, image generator.
- **Enterprise**: Notion connector, calendar sync, policy checker, CRM adapter.

Record each tool’s onboarding log (date, owner, PR link) in `docs/mcp/onboarding-log.md`.

## Secret Management

- **Local/dev**: Populate `.env` entries (`TOOLS__MCP__API_KEY`, `TOOLS__MCP__SIGNING_SECRET`, etc.).
- **Staging/Prod**: Point environment variables to secret manager references (e.g., Azure Key Vault, AWS Secrets Manager). Example:
  - `TOOLS__MCP__API_KEY=@Microsoft.KeyVault(SecretUri=https://vault.vault.azure.net/secrets/mcp-api-key/)`
  - `TOOLS__MCP__SIGNING_SECRET=aws-secrets://neuraforge/mcp-signing`
- Document handoffs and expiry dates in `docs/mcp/onboarding-log.md` to keep credential ownership clear.

## Runbook Snippets

- **Validate catalog availability**:
  ```bash
  curl -H "Authorization: Bearer $TOOLS__MCP__API_KEY" "$TOOLS__MCP__ENDPOINT$TOOLS__MCP__CATALOG_PATH"
  ```
- **Check diagnostics endpoint**:
  ```bash
  curl http://localhost:8000/api/v1/diagnostics/mcp | jq
  ```
- **Re-run smoke tests**:
  ```powershell
  E:/NeuraForge-The_Digital_Frontier_of_Autonomous_Intelligence/implementation/backend/.venv/Scripts/python.exe -m pytest tests/smoke/test_tools_catalog.py
  ```
- **Seed research tools**:
  ```powershell
  $env:TOOLS__MCP__ENDPOINT="https://mcp-staging.neuraforge.ai"
  $env:TOOLS__MCP__API_KEY="<catalog-admin-token>"
  E:/NeuraForge-The_Digital_Frontier_of_Autonomous_Intelligence/implementation/backend/.venv/Scripts/python.exe scripts/register_research_tools.py
  ```

## Reporting

Log weekly status in Dashboard/Notion including:

- Tools enabled this week and corresponding PRs.
- Outstanding secrets or rate-limit agreements.
- Blockers (schema disagreements, catalog downtime, credential rotations).

Use this document as the single source for onboarding readiness across teams.
