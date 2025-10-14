export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

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
