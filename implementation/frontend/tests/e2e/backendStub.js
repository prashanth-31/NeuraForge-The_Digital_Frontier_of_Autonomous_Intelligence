/* eslint-disable no-console */
import http from "node:http";

const port = Number(process.env.E2E_STUB_PORT || 8000);
const host = process.env.E2E_STUB_HOST || "127.0.0.1";
const reviewToken = process.env.E2E_REVIEW_TOKEN || "test-review-token";

let lastTask = null;

const reviews = [
  {
    ticket_id: "stub-ticket-001",
    task_id: "stub-task-001",
    status: "open",
    summary: "Guardrail escalation requires validation",
    created_at: new Date(Date.now() - 45 * 60000).toISOString(),
    updated_at: new Date(Date.now() - 15 * 60000).toISOString(),
    assigned_to: null,
    sources: ["guardrail"],
    notes: [],
    escalation_payload: {
      prompt: "Ensure quarterly report references the latest revenue figures.",
    },
  },
  {
    ticket_id: "stub-ticket-002",
    task_id: "stub-task-002",
    status: "in_review",
    summary: "Finance agent awaiting approval",
    created_at: new Date(Date.now() - 120 * 60000).toISOString(),
    updated_at: new Date(Date.now() - 10 * 60000).toISOString(),
    assigned_to: "reviewer-a",
    sources: ["agent"],
    notes: [
      {
        note_id: "note-1",
        author: "reviewer-a",
        content: "Verifying calculations",
        created_at: new Date(Date.now() - 20 * 60000).toISOString(),
      },
    ],
    escalation_payload: {
      prompt: "Compile financial variance analysis for Q3.",
    },
  },
];

const reviewMetrics = {
  generated_at: new Date().toISOString(),
  totals: {
    open: 1,
    in_review: 1,
    resolved: 0,
    dismissed: 0,
  },
  assignment: {
    by_reviewer: {
      "reviewer-a": 1,
    },
    unassigned_open: 1,
  },
  aging: {
    open_average_minutes: 42,
    open_oldest_minutes: 75,
    in_review_average_minutes: 110,
  },
  resolution: {
    average_minutes: 68,
    median_minutes: 50,
    completed_last_24h: 3,
  },
};

const sendJson = (res, status, payload) => {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Allow-Methods": "GET,POST,PATCH,OPTIONS",
  });
  res.end(body);
};

const handleOptions = (res) => {
  res.writeHead(204, {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Allow-Methods": "GET,POST,PATCH,OPTIONS",
  });
  res.end();
};

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  if (req.method === "OPTIONS") {
    handleOptions(res);
    return;
  }

  const ensureAuth = () => {
    const header = req.headers.authorization;
    if (!header || header !== `Bearer ${reviewToken}`) {
      sendJson(res, 401, { detail: "Unauthorized" });
      return false;
    }
    return true;
  };

  if (req.method === "GET" && url.pathname === "/healthz") {
    sendJson(res, 200, { status: "ok" });
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/v1/reviews") {
    if (!ensureAuth()) return;
    sendJson(res, 200, reviews);
    return;
  }

  if (req.method === "GET" && url.pathname === "/api/v1/reviews/metrics") {
    if (!ensureAuth()) return;
    sendJson(res, 200, reviewMetrics);
    return;
  }

  if (req.method === "GET" && url.pathname.startsWith("/api/v1/history/")) {
    const taskId = decodeURIComponent(url.pathname.split("/").pop());
    if (!lastTask || taskId !== lastTask.id) {
      sendJson(res, 404, { detail: "Task not found" });
      return;
    }
    sendJson(res, 200, lastTask.history);
    return;
  }

  if (req.method === "GET" && url.pathname.startsWith("/api/v1/tasks/")) {
    const taskId = decodeURIComponent(url.pathname.split("/").pop());
    if (!lastTask || taskId !== lastTask.id) {
      sendJson(res, 404, { detail: "Task not found" });
      return;
    }
    sendJson(res, 200, lastTask.status);
    return;
  }

  if (req.method === "POST" && url.pathname === "/api/v1/submit_task/stream") {
    let buffer = "";
    req.setEncoding("utf8");
    req.on("data", (chunk) => {
      buffer += chunk;
    });
    req.on("end", () => {
      let payload;
      try {
        payload = buffer ? JSON.parse(buffer) : {};
      } catch (error) {
        sendJson(res, 400, { detail: "Invalid JSON" });
        return;
      }

      const taskId = `stub-task-${Date.now()}`;
      lastTask = {
        id: taskId,
        prompt: payload?.prompt ?? "",
        history: [
          {
            agent: "insights-agent",
            content: { text: "Stub agent response" },
            confidence: 0.92,
            timestamp: new Date().toISOString(),
          },
        ],
        status: {
          task_id: taskId,
          status: "completed",
          metrics: {
            agents_completed: 1,
            agents_failed: 0,
            guardrail_events: 0,
            negotiation_rounds: 1,
          },
          guardrails: { decisions: [] },
          events: [
            {
              event: "agent_completed",
              agent: "insights-agent",
              timestamp: new Date().toISOString(),
            },
            {
              event: "task_completed",
              timestamp: new Date().toISOString(),
            },
          ],
        },
      };

      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "Access-Control-Allow-Origin": "*",
      });

      const sendEvent = (event, data) => {
        res.write(`event: ${event}\n`);
        res.write(`data: ${JSON.stringify(data)}\n\n`);
      };

      sendEvent("task_started", {
        task_id: taskId,
        timestamp: new Date().toISOString(),
        prompt: lastTask.prompt,
      });

      setTimeout(() => {
        sendEvent("agent_completed", {
          agent: "insights-agent",
          task_id: taskId,
          output: {
            summary: "Stub agent response",
            confidence: 0.92,
            metadata: {
              confidence_breakdown: {
                base: 0.6,
                evidence: 0.3,
                self_assessment: 0.1,
              },
            },
          },
          timestamp: new Date().toISOString(),
        });
      }, 50);

      setTimeout(() => {
        sendEvent("task_completed", {
          task_id: taskId,
          status: "completed",
          timestamp: new Date().toISOString(),
        });
        res.end();
      }, 120);
    });
    return;
  }

  sendJson(res, 404, { detail: "Not Found" });
});

server.listen(port, host, () => {
  console.log(`Stub backend listening on http://${host}:${port}`);
});

const shutdown = () => {
  server.close(() => {
    process.exit(0);
  });
};

process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
