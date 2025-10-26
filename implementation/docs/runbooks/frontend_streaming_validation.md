# Frontend Streaming Validation Runbook

_Date: 2025-10-19_

## Purpose
Confirm that the reviewer console renders the Phase 7 structured SSE envelopes (`neuraforge.task-event.v1`) correctly and capture the required before/after evidence.

## Prerequisites
- Backend running locally with Phase 7 code (`uvicorn app.main:app --reload`).
- Frontend dev server started from `implementation/frontend` via `npm run dev`.
- Test user with reviewer permissions and valid JWT for the console (see `docs/security.md`).
- Screenshot tooling (Snip & Sketch, Flameshot, or built-in OS shortcut).

## Steps
1. **Launch Console**
   - Open `http://localhost:5173/reviews` in a Chromium-based browser.
   - Authenticate using reviewer credentials; ensure the JWT loads successfully.
2. **Baseline Screenshot**
   - Navigate to an existing completed task timeline.
   - Capture the current timeline and lifecycle chips (label screenshot `pre-envelope-update.png`).
3. **Trigger New Task**
   - In a separate tab, open the main console and submit a test prompt (e.g., "Summarise the latest LLM paper on retrieval augmentation").
   - Wait for streaming updates; confirm timeline chips pulse as agents start/complete.
4. **Verify SSE Payload Rendering**
   - Ensure tool invocation rows show latency and payload key chips.
   - Confirm guardrail badges appear when `guardrail_triggered` events fire.
   - If elements are missing, grab a console log export (`F12 → Console → Save as...`).
5. **Post-Update Screenshot**
   - Once the task completes, capture the updated timeline (`post-envelope-update.png`).
   - Take an additional close-up of the lifecycle list highlighting velocity or guardrail metadata (`post-lifecycle-detail.png`).
6. **Attach Evidence**
   - Store screenshots in `docs/daily-updates/assets/2025-10-19/`.
   - Update `docs/daily/2025-10-19.md` with a short note referencing the new assets and findings.

## Troubleshooting
- **No Events Displayed**: Check browser devtools network tab; confirm `/api/v1/submit_task/stream` responses include `schema: "neuraforge.task-event.v1"`. If missing, restart backend.
- **JSON Parse Errors**: Inspect console for `sse_parse_error`; capture payload sample and open an issue tagged `frontend` + `backend`.
- **UI Drift**: If timeline still expects legacy fields, note the component path (likely `src/components/reviews/Timeline.tsx`) and open a task to remap props.

## Deliverables
- Three screenshots (`pre-envelope-update.png`, `post-envelope-update.png`, `post-lifecycle-detail.png`).
- Console log export if anomalies detected.
- `docs/daily/2025-10-19.md` entry summarizing results and linking to assets.
