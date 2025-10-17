import { test, expect, type Page } from "@playwright/test";

const REVIEW_NAV_SELECTOR = 'a:has-text("Reviews")';
const REVIEWER_CONSOLE_HEADING = "Reviewer Console";
const WORKSPACE_PLACEHOLDER = "Ask NeuraForge anything...";
const STREAM_RESPONSE_TEXT = "Stub agent response";

async function signIn(page: Page) {
  await page.goto("/auth");
  await expect(page).toHaveTitle(/NeuraForge/);

  await page.fill('input[type="email"]', "demo@neuraforge.ai");
  await page.fill('input[type="password"]', "NeuraForge123!");
  await page.click('button:has-text("Sign In")');
  await page.waitForURL("/**", { waitUntil: "networkidle" });
}

test("workspace streaming and reviewer console smoke", async ({ page }) => {
  await signIn(page);

  // Ensure workspace shows initial state
  await expect(page.locator("text=Collaborative Mode")).toBeVisible();

  // Trigger a streaming task and observe the assistant response
  const textarea = page.locator(`textarea[placeholder="${WORKSPACE_PLACEHOLDER}"]`);
  await textarea.fill("Smoke test prompt");
  await textarea.press("Enter");

  await expect(page.locator("text=Agents responding...")).toBeVisible({ timeout: 10_000 });
  await expect(page.locator(`text=${STREAM_RESPONSE_TEXT}`)).toBeVisible({ timeout: 10_000 });

  // Navigate to reviewer console and verify metrics render
  await page.click(REVIEW_NAV_SELECTOR);
  await expect(page.locator(`text=${REVIEWER_CONSOLE_HEADING}`)).toBeVisible();
  const queueMetricsHeading = page.getByRole("heading", { name: "Queue metrics" });
  await expect(queueMetricsHeading).toBeVisible();
  await expect(page.getByText("Unassigned", { exact: true })).toBeVisible({ timeout: 5000 });
});
