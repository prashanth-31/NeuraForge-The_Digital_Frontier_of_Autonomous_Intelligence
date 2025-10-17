import http from "k6/http";
import { Trend, Counter } from "k6/metrics";
import { check, sleep } from "k6";

const taskLatency = new Trend("submit_task_latency", true);
const taskFailures = new Counter("submit_task_failures");

export const options = {
  vus: Number(__ENV.NEURAFORGE_VUS || 10),
  duration: __ENV.NEURAFORGE_DURATION || "1m",
  thresholds: {
    submit_task_latency: ["p(95)<1500", "avg<750"],
    submit_task_failures: ["count<5"],
  },
};

const baseUrl = `${__ENV.NEURAFORGE_BASE_URL || "http://localhost:8000"}`;

function buildPayload(iteration) {
  return {
    prompt: `Load-test prompt #${iteration}`,
    metadata: {
      source: "k6",
      iteration,
      priority: iteration % 3 === 0 ? "high" : "standard",
    },
  };
}

export default function () {
  const iteration = __ITER + 1;
  const payload = JSON.stringify(buildPayload(iteration));
  const headers = { "Content-Type": "application/json" };

  const start = Date.now();
  const response = http.post(`${baseUrl}/api/v1/submit_task`, payload, {
    headers,
    tags: {
      endpoint: "submit_task",
    },
  });
  const latency = Date.now() - start;
  taskLatency.add(latency);

  const ok = check(response, {
    "status is 200": (res) => res.status === 200,
    "task id present": (res) => Boolean(res.json("task_id")),
  });

  if (!ok) {
    taskFailures.add(1);
  }

  sleep(Number(__ENV.NEURAFORGE_SLEEP || 1));
}
