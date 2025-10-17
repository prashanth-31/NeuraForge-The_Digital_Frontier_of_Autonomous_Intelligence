import { defineConfig, devices } from "@playwright/test";

const previewHost = "127.0.0.1";
const previewPort = Number(process.env.E2E_PREVIEW_PORT || 4173);
const stubHost = "127.0.0.1";
const stubPort = Number(process.env.E2E_STUB_PORT || 8000);
const reviewToken = process.env.E2E_REVIEW_TOKEN || "test-review-token";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 60_000,
  expect: {
    timeout: 15_000,
  },
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: `http://${previewHost}:${previewPort}`,
    trace: process.env.CI ? "retain-on-failure" : "on-first-retry",
    video: "off",
    screenshot: "only-on-failure",
  },
  reporter: [
    ["list"],
    ["html", { outputFolder: "playwright-report", open: "never" }],
  ],
  webServer: [
    {
      command: `node tests/e2e/backendStub.js`,
      port: stubPort,
      reuseExistingServer: !process.env.CI,
      env: {
        E2E_STUB_PORT: String(stubPort),
        E2E_STUB_HOST: stubHost,
        E2E_REVIEW_TOKEN: reviewToken,
      },
      cwd: "./",
    },
    {
      command: `npm run preview -- --host ${previewHost} --port ${previewPort}`,
      port: previewPort,
      reuseExistingServer: !process.env.CI,
      env: {
        VITE_API_BASE_URL: `http://${stubHost}:${stubPort}`,
        VITE_REVIEW_API_TOKEN: reviewToken,
      },
      cwd: "./",
    },
  ],
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
