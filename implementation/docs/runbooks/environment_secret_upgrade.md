# Environment & Secret Upgrade Guidance

_Last updated: 2025-10-17_

This guide documents the approved process for rolling out configuration and secret changes across NeuraForge environments. Follow it whenever upgrading infrastructure endpoints, rotating credentials, or promoting new feature flags from staging to production.

## Scope
- FastAPI backend services (`implementation/backend`)
- Supporting data stores (Postgres, Redis, Qdrant)
- Observability tooling (Prometheus, Grafana, Alertmanager)
- Tooling integrations accessed through MCP

## Environment Variable Matrix
The baseline `.env` templates live in `environments/staging.env.example` and `environments/production.env.example`. Use the matrix below to determine responsibility and storage location for each setting.

| Category | Variable prefix | Staging source | Production source | Notes |
| --- | --- | --- | --- | --- |
| Core service | `ENVIRONMENT`, `API_V1_PREFIX`, `BACKEND_BASE_URL` | Checked-in template | Checked-in template | Update when endpoint paths change.
| Data stores | `POSTGRES__*`, `REDIS__*`, `QDRANT__*` | Azure Key Vault references | Azure Key Vault references | Validate connectivity and rotation windows.
| Auth | `AUTH__*` | Azure Key Vault secret | Azure Key Vault secret | Keep algorithm and expiry in version control; secrets rotate via vault.
| Observability | `OBSERVABILITY__*` | Template overrides | Template overrides | Toggle features (e.g., `PROMETHEUS_ENABLED`) per environment.
| MCP tooling | `TOOLS__MCP__*` | Mixed (Key Vault + AWS Secrets Manager) | Mixed (Key Vault + AWS Secrets Manager) | Signing secret stored in AWS Secrets; other credentials in Key Vault.
| LLM provider | `OLLAMA__*` | Template for host/port; Key Vault for tokens if needed | Template for host/port; Key Vault for tokens if needed | Ensure model upgrades are coordinated with scaling plan.

## Upgrade Checklist
1. **Plan the change**
   - Draft a change request with target variables, reason for update, and rollback plan.
   - Confirm downstream dependencies (frontend URLs, k6 scripts, alert rules) remain valid.
2. **Stage environment update**
   - Modify `staging.env.example` in a feature branch if the change is structural (new variable, renamed key).
   - For secret rotations, update the value directly in Azure Key Vault or AWS Secrets Manager and note the version ID.
   - Run `scripts/staging_sync_artifacts.py` to pull the latest workflow outputs when testing dashboard or alert updates.
3. **Validate in staging**
   - Deploy the new values to staging by reloading the `backend` service (Docker Compose or Kubernetes rollout).
   - Run targeted tests (`pytest tests/test_review_metrics.py tests/test_review_metrics_endpoint.py`) and hit `/metrics` to ensure no missing configs.
   - Observe Grafana dashboards and Alertmanager to verify environment labeling and routing still align with expectations.
4. **Production promotion**
   - Mirror template changes into `production.env.example` once staging validation passes.
   - Apply secret rotations in production vaults during a planned window; document old/new version IDs.
   - Redeploy the backend and confirm health checks, reviewer console auth, and MCP integration succeed.
5. **Post-change tasks**
   - Update `docs/daily/<date>.md` with the change summary and verification evidence.
   - Close the roadmap item by ticking the checkbox in `docs/PHASE7_ROADMAP.md` once both environments are aligned.
   - Schedule follow-up rotation reminders if the new secret has a defined expiry.

## Secret Management Notes
- **Azure Key Vault** references follow the `@Microsoft.KeyVault(SecretUri=...)` format; confirm the service principal used by the deployment pipeline has access to the new version before rollout.
- **AWS Secrets Manager** entries (e.g., `aws-secrets://neuraforge/prod/mcp-signing`) require the IAM role assumed by the backend to include `secretsmanager:GetSecretValue` permissions for the ARN. Validate access with a dry-run `aws secretsmanager get-secret-value` using the pipeline role.
- **Local development** should not point to production vaults. Use `.env.development` files with placeholder values or local secrets tooling like `direnv`.
- **Rotation cadence**: security-critical secrets (JWT signing, MCP client secrets) rotate quarterly; datastore passwords rotate semi-annually unless mandated sooner by incident response.

## Rollback Procedure
1. Restore the previous secret version in the vault (set the old version to `enabled` and disable the new version).
2. Reapply the prior `.env` template via infrastructure automation or manual edit.
3. Redeploy services and confirm health checks.
4. Capture the incident in the daily log and create a follow-up ticket to diagnose the failed upgrade.

## References
- `implementation/backend/environments/staging.env.example`
- `implementation/backend/environments/production.env.example`
- `implementation/docs/security.md`
- `implementation/scripts/staging_sync_artifacts.py`
