# MCP Configuration Guide

This guide outlines how to configure Model Context Protocol (MCP) tools within NeuraForge. The configuration lives under the `tools.mcp` namespace of `app.core.config.Settings` and can be overridden via environment variables.

## Environment Variables

| Variable | Description |
| --- | --- |
| `TOOLS__MCP__ENDPOINT` | Base URL for the MCP router or gateway. |
| `TOOLS__MCP__API_KEY` | Optional bearer token used when authenticating with the MCP router. |
| `TOOLS__MCP__API_KEY_HEADER` | Override the header name used when attaching the API token (`Authorization` by default). |
| `TOOLS__MCP__AUTH_SCHEME` | Prefix applied to the API token (e.g., `Bearer`, `Basic`, or empty). |
| `TOOLS__MCP__CLIENT_ID` | Optional client identifier used for Basic auth or request signing. |
| `TOOLS__MCP__CLIENT_SECRET` | Optional client secret used for Basic auth or request signing. |
| `TOOLS__MCP__TIMEOUT_SECONDS` | Request timeout in seconds for outbound MCP calls. |
| `TOOLS__MCP__ENABLED` | Enable (`true`) or disable (`false`) tooling integration. |
| `TOOLS__MCP__CACHE_TTL_SECONDS` | TTL applied to MCP response cache in seconds. |
| `TOOLS__MCP__RATE_LIMIT__MAX_CALLS` | Maximum tool invocations within the rolling rate window. |
| `TOOLS__MCP__RATE_LIMIT__PERIOD_SECONDS` | Length of the rolling window for rate limiting. |
| `TOOLS__MCP__HEALTHCHECK_PATH` | Relative path used to verify MCP availability. |
| `TOOLS__MCP__CATALOG_PATH` | Endpoint used to refresh the MCP tool catalog. |
| `TOOLS__MCP__INVOKE_PATH_TEMPLATE` | Template path used when invoking a tool (supports `{tool}` placeholder). |
| `TOOLS__MCP__CATALOG_REFRESH_SECONDS` | TTL before the tool catalog is refreshed. |
| `TOOLS__MCP__VERIFY_SSL` | Validate HTTPS certificates (`true` by default). |
| `TOOLS__MCP__EXTRA_HEADERS__<HEADER>` | Additional headers to forward to the MCP router. |
| `TOOLS__MCP__ALIASES__<ALIAS>` | Map logical alias to catalog identifier, e.g. `finance.snapshot=finance/alpha_vantage`. |
| `TOOLS__MCP__MAX_RETRIES` | Maximum retry attempts for a single MCP request. |
| `TOOLS__MCP__RETRY_BACKOFF_SECONDS` | Base backoff window applied between retries. |
| `TOOLS__MCP__RETRY_JITTER_SECONDS` | Maximum jitter added to each retry backoff. |
| `TOOLS__MCP__CIRCUIT_BREAKER_THRESHOLD` | Number of consecutive failures before opening the circuit. |
| `TOOLS__MCP__CIRCUIT_BREAKER_RESET_SECONDS` | Duration to keep the circuit open before new attempts are allowed. |
| `TOOLS__MCP__SIGNING_SECRET` | Optional HMAC secret used to sign MCP requests. |
| `TOOLS__MCP__SIGNING_HEADER` | Header name that carries the generated request signature. |
| `TOOLS__MCP__SIGNING_ALGORITHM` | Signing algorithm identifier (currently `hmac-sha256`). |

## Payload Schemas

The canonical MCP request and response envelope schemas are published under `docs/mcp/schemas/`. Refer to [`schemas/README.md`](./schemas/README.md) for version history, validation workflow, and sample payloads.

- [`request.schema.json`](./schemas/request.schema.json)
- [`response.schema.json`](./schemas/response.schema.json)

Each tool MUST validate incoming payloads against the tool-specific input schema exposed by the catalog, **in addition to** the global envelope schema documented above.

## Implementation Checklist

1. **Register the tool** in the MCP catalog with accurate metadata (description, input/output schemas, labels).
2. **Configure secrets** and credentials using environment variables or the platform secret store.
3. **Validate health checks** via the configured `healthcheck_path` to ensure the tool is available before activation.
4. **Run smoke tests** using `scripts/tools_smoketest.py` (to be added) or equivalent harness prior to enabling the tool in production.
5. **Monitor telemetry**: Prometheus metrics prefixed with `neuraforge_mcp_` capture request rates, latency, and circuit breaker activity.
6. **Document onboarding** steps for operators in `docs/agents/README.md` once the tool is exposed to agent playbooks.
7. **Track rollout** using [`docs/mcp/category_onboarding.md`](./category_onboarding.md) and update the log in [`docs/mcp/onboarding-log.md`](./onboarding-log.md).

## Environment Examples

Sample environment files for managed deployments:

- Staging: `backend/environments/staging.env.example`
- Production: `backend/environments/production.env.example`

Each file references secret manager entries (Key Vault / AWS Secrets Manager) for MCP API keys, client secrets, and signing material. Copy the example, replace placeholders with real secret URIs, and load it via your deployment pipeline.
