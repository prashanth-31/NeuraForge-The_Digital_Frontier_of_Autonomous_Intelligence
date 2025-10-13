# MCP Schema Reference

This directory contains the canonical envelope schemas for Model Context Protocol (MCP) tool invocations. The schemas complement each tool's specific input and output contracts exposed through the MCP catalog.

## Versioning

- Envelope schemas follow [Semantic Versioning](https://semver.org/).
- The current envelope version is **1.0.0** (`version` field in both request and response).
- Minor revisions (e.g. `1.1.x`) may add optional fields. Major revisions (`2.x.y`) signal breaking changes and will coexist alongside prior versions during migration.

Version history is tracked in [`CHANGELOG.md`](./CHANGELOG.md). Downstream services should validate the `version` field so they can reject or adapt to unsupported revisions.

## Schemas

| Artifact | Description |
| --- | --- |
| [`request.schema.json`](./request.schema.json) | Envelope for MCP tool invocations. Defines metadata (`id`, `version`, `timestamp`), routing (`tool`), optional `auth`, `context`, `metadata`, and the tool-specific `payload`. |
| [`response.schema.json`](./response.schema.json) | Envelope returned by MCP tools. Includes correlation (`id`, `version`), `status` object with standard codes (`success`, `accepted`, `retry`, `error`), optional `errors`, `output`, `metadata`, and execution timings. |

Both schemas comply with JSON Schema Draft 7 and can be included in OpenAPI documents via `$ref`.

## Sample Payloads

```jsonc
// Request
{
  "id": "3fd7d2b1-79d1-4b8e-92c8-105c1ce9b6f7",
  "version": "1.0.0",
  "timestamp": "2025-10-13T15:10:00Z",
  "tool": "research.search",
  "trace_id": "trace-9ab213ef",
  "context": {
    "task_id": "task-123",
    "agent": "ResearchAgent"
  },
  "auth": {
    "type": "bearer",
    "token": "mcp-dev-token",
    "scopes": ["search:read"]
  },
  "metadata": {
    "priority": "high"
  },
  "payload": {
    "query": "latest transformer interpretability papers",
    "max_results": 5
  }
}
```

```jsonc
// Response
{
  "id": "3fd7d2b1-79d1-4b8e-92c8-105c1ce9b6f7",
  "version": "1.0.0",
  "status": {
    "code": "success",
    "http": 200,
    "message": "3 results returned"
  },
  "timestamp": "2025-10-13T15:10:01Z",
  "duration_ms": 845,
  "output": [
    { "title": "Interpreting Attention", "url": "https://arxiv.org/abs/2401.01234" }
  ],
  "metadata": {
    "cached": false,
    "tool": "search/tavily"
  }
}
```

## Validation Workflow

1. Fetch the latest schemas from this directory (or via the published `$id` URLs).
2. Validate envelope JSON using a Draft 7 compliant validator before invoking downstream adapters.
3. Validate the `payload` against the tool-specific schema provided by the MCP catalog.
4. Log and surface any schema mismatches via MCP diagnostics to aid operators.

## OpenAPI Integration

The envelope schemas can be referenced inside FastAPI's OpenAPI document:

```python
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI()

schema_components = {
    "MCPRequest": {
        "$ref": "./docs/mcp/schemas/request.schema.json"
    },
    "MCPResponse": {
        "$ref": "./docs/mcp/schemas/response.schema.json"
    },
}

app.openapi_schema = None

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="NeuraForge MCP",
        version="1.0.0",
        routes=app.routes,
    )
    openapi_schema.setdefault("components", {}).update({"schemas": schema_components})
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

Keep the OpenAPI metadata in sync with the schema version to avoid confusing integrators.
