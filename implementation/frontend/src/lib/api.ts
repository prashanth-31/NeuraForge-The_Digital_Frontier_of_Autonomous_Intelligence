const DEFAULT_FALLBACK_URL = "http://localhost:8000";

const sanitizeBaseUrl = (value: string) => value.replace(/\/$/, "");

const resolveRuntimeBaseUrl = (): string | undefined => {
  if (typeof window === "undefined") return undefined;

  const globalValue = typeof window.__NEURAFORGE_API_BASE__ === "string" ? window.__NEURAFORGE_API_BASE__.trim() : "";
  if (globalValue) return globalValue;

  const meta = typeof document !== "undefined" ? (document.querySelector('meta[name="api-base-url"]') as HTMLMetaElement | null) : null;
  if (meta?.content?.trim()) return meta.content.trim();

  if (window.location?.origin) {
    return window.location.origin;
  }

  return undefined;
};

const envBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const runtimeBaseUrl = envBaseUrl ?? resolveRuntimeBaseUrl() ?? DEFAULT_FALLBACK_URL;

if (!envBaseUrl && typeof window !== "undefined") {
  console.warn(
    "VITE_API_BASE_URL is not defined. Falling back to runtime origin for API requests (", 
    runtimeBaseUrl,
    ").",
  );
}

export const API_BASE_URL = sanitizeBaseUrl(runtimeBaseUrl);

export const REVIEW_API_TOKEN = import.meta.env.VITE_REVIEW_API_TOKEN ?? "";

export const buildReviewHeaders = (token?: string) => {
  const bearer = token ?? REVIEW_API_TOKEN;
  return bearer
    ? {
        Authorization: `Bearer ${bearer}`,
        "Content-Type": "application/json",
      }
    : {
        "Content-Type": "application/json",
      };
};
