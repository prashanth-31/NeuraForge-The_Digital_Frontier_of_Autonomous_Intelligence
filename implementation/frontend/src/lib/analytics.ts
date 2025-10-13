export interface ConfidenceAnalyticsEvent {
  agent: string;
  breakdown: Record<string, number>;
  confidence?: number;
  taskId?: string | null;
  timestamp: string;
  source: "stream" | "history";
}

export const emitConfidenceAnalytics = (event: ConfidenceAnalyticsEvent) => {
  if (typeof window === "undefined") {
    return;
  }

  const analyticsEvent = new CustomEvent("neuraforge:confidence-breakdown", { detail: event });
  window.dispatchEvent(analyticsEvent);

  if (import.meta.env.DEV) {
    console.debug("[analytics] confidence-breakdown", event);
  }
};

export interface WorkspaceStateEvent {
  messageCount: number;
  isStreaming: boolean;
  timestamp: string;
  lastMessage?: {
    id: string;
    role: string;
    agentName?: string;
    confidence?: number;
  };
}

export const emitWorkspaceState = (event: WorkspaceStateEvent) => {
  if (typeof window === "undefined") {
    return;
  }

  const analyticsEvent = new CustomEvent("neuraforge:workspace-state", { detail: event });
  window.dispatchEvent(analyticsEvent);

  if (import.meta.env.DEV) {
    console.debug("[analytics] workspace-state", event);
  }
};
