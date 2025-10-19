# Phase 7 Follow-Up Execution Plan

_Date: 2025-10-19_

## 1. Objectives
- Land the remaining integration touchpoints after Phase 7 delivery.
- Prove stability of the upgraded CI pipelines under real workloads.
- Capture knowledge and handoffs needed for cross-team rollout.
- Prepare the backlog that will seed Phase 8 roadmap discussions.

---

## 2. Immediate Priorities (Day 0-2)
1. **Frontend Streaming Alignment**
   - Sync with frontend owners to review the `neuraforge.task-event.v1` envelope schema.
   - Update Socket.io client listeners to parse `payload` fragments instead of legacy flat events.
   - Exercise the reviewer console against a local backend build; confirm task timelines render and regression screenshots are captured.
   - File tracking issue if any component still expects pre-Phase7 envelope fields.
    - Follow `docs/runbooks/frontend_streaming_validation.md` for the smoke test and evidence capture.
2. **Reviewer Dashboard Velocity Widgets**
   - Extend the frontend charts to consume `velocity.daily_average` and `velocity.per_reviewer` fields.
   - Validate Prometheus panels display the new metrics; snapshot Grafana dashboard once visuals settle.
   - Share dashboard links with the reviewer operations team for sign-off.
3. **CI Runtime Monitoring**
   - Record two full runs of `.github/workflows/phase5-observability.yml` to baseline duration after the metrics acceptance split.
   - Open an issue if runtime exceeds 14 minutes or queuing becomes problematic.

---

## 3. Near-Term Workstream (Week 1)
- **Operational Runbooks**
  - Fold the new reviewer velocity and SSE behavior into `docs/runbooks/reviewer_operations.md` and `docs/runbooks/staging_k6_smoke.md` checklists.
  - Add troubleshooting steps for mismatched event schemas or missing metrics.
- **Quality Gates**
  - Implement lightweight contract tests on the frontend (Playwright or Vitest) to verify the streaming payload; wire them into PR checks.
  - Expand backend tests around the `/tasks/{task_id}` endpoint to cover empty event stores and truncated histories.
- **Pydantic v3 Preparation**
  - Inventory remaining `@model_validator(mode="before")` usages and schedule upgrades to `field_validator`/`model_validator` patterns supported in Pydantic 3.
  - Draft migration notes to keep adapters compliant once Pydantic 3 lands.

---

## 4. Cross-Team Coordination (Week 2)
- **Release Management**
  - Publish a Phase 7 change summary to `docs/daily/2025-10-24.md`, highlighting operator-impacting updates.
  - Confirm staging environment picks up the new Docker images; watch Prometheus / Grafana after deploy.
- **Security Review**
  - Re-run JWT / rate limiting penetration checklist with security stakeholders.
  - Ensure audit log ingestion (Elastic/Loki) receives the enriched envelope when Phase 7 binaries roll out.
- **Data & Memory Sync**
  - Validate that hybrid memory consolidation jobs continue to run after SSE/metrics changes.
  - Rebuild Qdrant indexes if review velocity metadata needs new vector dimensions (open an RFC if required).

---

## 5. Phase 8 Backlog Seeding
- Collect lessons learned from Phase 7 in a retro (capture in `docs/PHASE7_RETRO.md`).
- Draft initial Phase 8 themes:
  1. Guided workflows for enterprise agents (role-based templates, compliance logging).
  2. Automated benchmarking harness integration into CI (nightly load tests).
  3. Progressive delivery for frontend + backend (feature flags, canary toggles).
- Socialize themes with stakeholders before locking roadmap.

---

## 6. Checklist
- [ ] Frontend SSE consumers updated and verified.
- [ ] Reviewer dashboard visualizations refreshed and approved.
- [ ] CI runtime metrics captured and compared to targets.
- [ ] Runbooks updated with Phase 7 behaviors.
- [ ] Frontend contract tests added.
- [ ] Pydantic validator audit completed.
- [ ] Staging smoke test executed post-deploy.
- [ ] Phase 7 retrospective drafted.
- [ ] Phase 8 proposal circulated.
- [ ] Streaming alignment sync scheduled (see `docs/meetings/phase7_streaming_alignment_sync.md`).

---

## 7. Owner Matrix
| Workstream | Primary | Support |
| --- | --- | --- |
| Frontend Streaming Alignment | Frontend Tech Lead | Backend API lead |
| Reviewer Dashboard Velocity | Data Viz Engineer | Observability lead |
| CI Runtime Monitoring | DevOps | Backend QA |
| Runbook Updates | Ops Enablement | Technical Writer |
| Pydantic Migration Prep | Backend Platform | MCP Adapter owners |
| Phase 8 Roadmap | Product Manager | Engineering Leadership |

---

## 8. Reporting Cadence
- Daily stand-up notes in `docs/daily/` (append to dated entry).
- Weekly sync summary posted in `docs/daily-updates/weeklies.md`.
- Escalations logged through Pager rotation and noted in runbooks.

---

## 9. Risks & Mitigations
- **Frontend drift**: Mitigate via contract tests and shared schema docs (`docs/api.md`).
- **CI instability**: Set alert if workflow variance exceeds ±20%; be ready to split suites.
- **Operator adoption**: Schedule enablement session to walk through new metrics and runbook flows.
- **Tech debt creep**: Track validator migration and hybrid memory verifications in Jira; review weekly.

---

## 10. Success Criteria
- All checklist items marked complete within two weeks.
- No unresolved S1/S2 issues reported by operators post Phase 7 rollout.
- Phase 8 roadmap approved with clear scope and resource plan.
