# Meta-Agent Observability Runbook

## Alert Catalog

| Alert | Trigger | Owner | Playbook |
| --- | --- | --- | --- |
| `MetaAgentDisputeSpike` | `rate(neuraforge_meta_agent_disputes_total[10m]) > 3` for 10 minutes | Orchestrator On-Call | Investigate surge in dissenting agent confidence scores. |
| `MetaResolutionLatencySLO` | `P95` meta-agent latency above 5s for 15 minutes | Orchestrator On-Call | Validate LLM health and dispute detector responsiveness. |

## Triage Checklist

1. **Confirm Alert Context**
   - Open Grafana dashboard *Meta-Agent Overview* (import `observability/grafana/dashboards/meta-agent.json` once published).
   - Review panels for dispute count, resolution latency, and escalation rate.

2. **Inspect Recent Tasks**
   - Query `/api/v1/reports/{task_id}/dossier.json` for impacted tasks to review agent perspectives and the synthesized summary.
   - Cross-check guardrail decisions and escalation sources in the dossier payload.

3. **Validate Meta-Agent Health**
   - Ensure Ollama model latency is within normal bounds (`neuraforge_meta_agent_resolution_latency_seconds`).
   - Spot-check the dispute detector thresholds in `core/config.py` (`meta_agent.consensus_delta_threshold`, `meta_agent.stddev_threshold`).

4. **Mitigation Options**
   - If disputes are legitimate, escalate to human reviewer via the operations Slack channel and attach the Markdown dossier.
   - For noisy disputes, temporarily raise the consensus delta threshold and file a follow-up issue to refine scoring logic.
   - When latency is elevated, restart the LLM service or switch to a lighter meta-summarization model in `Settings.meta_agent`.

5. **Postmortem Notes**
   - Record findings in the incident tracker, including task IDs and guardrail outcomes.
   - Update this runbook with any newly discovered remediation steps.

## Quick Reference API Calls

- Download JSON dossier: `GET /api/v1/reports/{task_id}/dossier.json`
- Download Markdown dossier: `GET /api/v1/reports/{task_id}/dossier.md`
- Fetch latest task history: `GET /api/v1/history/{task_id}`

## Escalation Path

1. Notify on-call orchestrator engineer (PagerDuty rotation `orchestrator@neuraforge.local`).
2. If sustained disputes correlate with guardrail escalations, loop in the risk and compliance delegate.
3. For repeated latency breaches, escalate to the infrastructure SRE to inspect GPU/CPU saturation on the Ollama host.
