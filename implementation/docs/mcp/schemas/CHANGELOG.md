# MCP Envelope Schema Changelog

All notable changes to the MCP request/response envelope schemas are documented here.

## [1.0.0] - 2025-10-13
- Published canonical request and response envelope schemas (Draft 7).
- Added explicit `version` field to both envelopes to support semver negotiation.
- Documented optional `trace_id`, `metadata`, and standardized status codes.
- Captured execution timing (`duration_ms`) and optional headers in response envelope.
