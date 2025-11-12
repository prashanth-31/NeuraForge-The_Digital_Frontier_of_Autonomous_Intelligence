# Agent Capability Contracts

This guide enumerates the Phase 4 agent interfaces. Contracts are enforced via Pydantic models in `app/schemas/agents.py` and registered through `app/agents/contracts.py`.

## Shared Models
- **AgentInput**: Carries task-level information (prompt, metadata, retrieved context, prior exchanges).
- **AgentOutput**: Standardized agent reply with capability identifier, confidence score (0-1), rationale, and optional evidence items.
- **AgentContractMetadata**: Describes tooling, timeouts, and streaming behaviour for each agent capability.

## Capability Matrix

| Capability | Agent Name         | Description                                                            | Tooling (MCP)                                                                                 | Timeout |
|------------|--------------------|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|---------|
| Research   | `research_agent`   | Synthesizes research-backed insights with citations and gap analysis.  | `search/tavily`, `research/summarizer`, `research/qdrant`, `research/doc_loader`             | 45s     |
| Finance    | `finance_agent`    | Produces financial assessments with market data and compliance checks. | `finance/alpha_vantage`, `finance/pandas`, `finance/plot`, `finance/finbert`                  | 60s     |
| Creative   | `creative_agent`   | Crafts stylized content aligned to brand and tone guidelines.          | `creative/stylizer`, `creative/tone_checker`, `creative/whisper_transcription`               | 75s*    |
| Enterprise | `enterprise_agent` | Delivers strategic recommendations referencing policy and knowledge.   | `enterprise/playbook` (composite), `enterprise/notion`, `enterprise/policy_checker`, `enterprise/crm` | 90s     |

\*Creative agent supports streaming responses for long-form generation.

## Validation Utilities
- `validate_agent_request(capability, payload)` returns a canonical `AgentInput` instance.
- `validate_agent_response(capability, payload)` returns `AgentOutput` and ensures the capability matches.
- `list_contracts()` exposes metadata for documentation or UI surfacing.

Include these validators in orchestrator flows before dispatching to ensure task payloads conform to the contract, and run outputs back through the response validator to maintain schema consistency. The enterprise playbook wrapper orchestrates a Notion knowledge search with policy-checker fallback so the agent always returns actionable steps even when the primary catalog entry is unavailable; downstream consumers should rely on the `actions` list emitted by the tool response rather than hitting MCP endpoints directly.
