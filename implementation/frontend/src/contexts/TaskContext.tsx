import { API_BASE_URL } from "@/lib/api";
import { emitConfidenceAnalytics } from "@/lib/analytics";
import {
  PropsWithChildren,
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
} from "react";

type ChatRole = "user" | "assistant" | "system";

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  timestamp: string;
  agentName?: string;
  confidence?: number;
  confidenceBreakdown?: Record<string, number>;
  toolMetadata?: ToolMetadata;
  metadata?: Record<string, unknown>;
}

export interface HistoryEntry {
  agent: string;
  content: string;
  confidence: number;
  timestamp: string;
  confidenceBreakdown?: Record<string, number>;
  toolMetadata?: ToolMetadata;
  metadata?: Record<string, unknown>;
}

interface ToolMetadata {
  name?: string;
  resolved?: string;
  cached?: boolean;
  latency?: number;
}

export interface ToolEvent {
  id: string;
  tool: string;
  resolvedTool?: string;
  status: "success" | "error";
  cached?: boolean;
  latencyMs?: number;
  timestamp: string;
  composite?: boolean;
  error?: string;
  payloadKeys?: string[];
}

export type MCPDiagnostics = Record<string, unknown>;

interface TaskContextValue {
  messages: ChatMessage[];
  history: HistoryEntry[];
  currentTaskId: string | null;
  isStreaming: boolean;
  submitTask: (prompt: string, metadata?: Record<string, unknown>) => Promise<void>;
  refreshHistory: (taskId: string) => Promise<void>;
  resetSession: () => void;
  toolEvents: ToolEvent[];
  mcpDiagnostics: MCPDiagnostics | null;
}

const TaskContext = createContext<TaskContextValue | undefined>(undefined);

const formatTimestamp = (iso?: string) => {
  if (!iso) {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
};

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

const toDisplayString = (payload: unknown): string => {
  if (payload === null || payload === undefined) {
    return "";
  }
  if (typeof payload === "string") {
    return payload;
  }
  if (Array.isArray(payload)) {
    return payload.map((item) => toDisplayString(item)).join("\n");
  }
  if (isRecord(payload)) {
    if (typeof payload.text === "string") {
      return payload.text;
    }
    if (typeof payload.content === "string") {
      return payload.content;
    }
    return JSON.stringify(payload, null, 2);
  }
  return String(payload);
};

const parseContentPayload = (
  payload: unknown,
): { text: string; metadata?: Record<string, unknown> } => {
  if (isRecord(payload)) {
    const metadata = isRecord(payload.metadata) ? (payload.metadata as Record<string, unknown>) : undefined;
    const textSource = payload.text ?? payload.content ?? payload.summary ?? payload;
    return { text: toDisplayString(textSource), metadata };
  }
  return { text: toDisplayString(payload) };
};

const mergeMetadata = (
  sources: Array<Record<string, unknown> | undefined>,
): Record<string, unknown> | undefined => {
  const merged: Record<string, unknown> = {};
  for (const source of sources) {
    if (!source) continue;
    for (const [key, value] of Object.entries(source)) {
      merged[key] = value;
    }
  }
  return Object.keys(merged).length > 0 ? merged : undefined;
};

const extractConfidenceBreakdown = (metadata?: Record<string, unknown>) => {
  if (!metadata) return undefined;
  const breakdown = metadata.confidence_breakdown;
  if (!isRecord(breakdown)) {
    return undefined;
  }
  const result: Record<string, number> = {};
  for (const [key, value] of Object.entries(breakdown)) {
    if (typeof value === "number") {
      result[key] = value;
    }
  }
  return Object.keys(result).length > 0 ? result : undefined;
};

const extractToolMetadata = (metadata?: Record<string, unknown>): ToolMetadata | undefined => {
  if (!metadata) return undefined;
  const raw = metadata.tool;
  if (!isRecord(raw)) {
    return undefined;
  }
  const info: ToolMetadata = {};
  if (typeof raw.name === "string") info.name = raw.name;
  if (typeof raw.resolved === "string") info.resolved = raw.resolved;
  if (typeof raw.cached === "boolean") info.cached = raw.cached;
  if (typeof raw.latency === "number") info.latency = raw.latency;
  return Object.keys(info).length > 0 ? info : undefined;
};

const randomId = () => (typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2));

export const TaskProvider = ({ children }: PropsWithChildren) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
  const [mcpDiagnostics, setMcpDiagnostics] = useState<MCPDiagnostics | null>(null);

  const resetSession = useCallback(() => {
    setMessages([]);
    setHistory([]);
    setCurrentTaskId(null);
    setIsStreaming(false);
    setToolEvents([]);
    setMcpDiagnostics(null);
  }, []);

  const refreshHistory = useCallback(async (taskId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/history/${taskId}`);
      if (!response.ok) {
        if (response.status === 404) {
          setHistory([]);
          return;
        }
        throw new Error(`Unable to fetch history (${response.status})`);
      }
      const data: Array<{
        agent: string;
        content: unknown;
        confidence: number;
        timestamp: string;
      }> = await response.json();
      const mapped = data.map((item) => {
        const parsed = parseContentPayload(item.content);
        const metadata = parsed.metadata;
        const confidenceBreakdown = extractConfidenceBreakdown(metadata);
        const toolMetadata = extractToolMetadata(metadata);
        if (confidenceBreakdown) {
          emitConfidenceAnalytics({
            agent: item.agent,
            breakdown: confidenceBreakdown,
            confidence: item.confidence,
            taskId,
            timestamp: item.timestamp,
            source: "history",
          });
        }
        return {
          agent: item.agent,
          content: parsed.text,
          confidence: item.confidence,
          timestamp: formatTimestamp(item.timestamp),
          confidenceBreakdown,
          toolMetadata,
          metadata,
        } satisfies HistoryEntry;
      });
      setHistory(mapped);
    } catch (error) {
      console.error("history_fetch_failed", error);
    }
  }, []);

  const submitTask = useCallback(
    async (prompt: string, metadata: Record<string, unknown> = {}) => {
      const trimmed = prompt.trim();
      if (!trimmed || isStreaming) {
        return;
      }

      const timestamp = formatTimestamp();
      const userMessage: ChatMessage = {
        id: randomId(),
        role: "user",
        content: trimmed,
        timestamp,
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsStreaming(true);
      setHistory([]);
      setToolEvents([]);
      setMcpDiagnostics(null);

      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/submit_task/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ prompt: trimmed, metadata }),
        });

        if (!response.ok || !response.body) {
          throw new Error(`Streaming request failed (${response.status})`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let activeTaskId: string | null = null;

        const processEvent = (eventName: string, payload: Record<string, unknown>) => {
          if (eventName === "task_started") {
            if (typeof payload.task_id === "string") {
              activeTaskId = payload.task_id;
              setCurrentTaskId(activeTaskId);
            }
            return;
          }

          if (eventName === "tool_invocation") {
            const toolName = typeof payload.tool === "string" ? payload.tool : "unknown.tool";
            const resolvedTool = typeof payload.resolved_tool === "string" ? payload.resolved_tool : undefined;
            const status = payload.status === "error" ? "error" : "success";
            const cached = typeof payload.cached === "boolean" ? payload.cached : undefined;
            const latencyValue = typeof payload.latency === "number" ? payload.latency : undefined;
            const composite = typeof payload.composite === "boolean" ? payload.composite : undefined;
            const errorMessage = typeof payload.error === "string" ? payload.error : undefined;
            const payloadKeys = Array.isArray(payload.payload_keys)
              ? payload.payload_keys.filter((item) => typeof item === "string")
              : undefined;
            const timestampIso = typeof payload.timestamp === "string" ? payload.timestamp : undefined;

            const event: ToolEvent = {
              id: randomId(),
              tool: toolName,
              resolvedTool,
              status,
              cached,
              latencyMs: typeof latencyValue === "number" ? latencyValue * 1000 : undefined,
              composite,
              error: errorMessage,
              payloadKeys,
              timestamp: formatTimestamp(timestampIso),
            };

            setToolEvents((previous) => [event, ...previous].slice(0, 50));
            return;
          }

          if (eventName === "mcp_status") {
            const statusPayload = payload.status;
            if (isRecord(statusPayload)) {
              setMcpDiagnostics(statusPayload as MCPDiagnostics);
            }
            return;
          }

          if (eventName === "agent_completed") {
            const outputRaw = payload.output as Record<string, unknown> | string | undefined;
            const outputRecord = isRecord(outputRaw) ? (outputRaw as Record<string, unknown>) : undefined;
            const outputMetadata = isRecord(outputRecord?.metadata)
              ? (outputRecord!.metadata as Record<string, unknown>)
              : undefined;
            const payloadMetadata = isRecord(payload.metadata)
              ? (payload.metadata as Record<string, unknown>)
              : undefined;
            const mergedMetadata = mergeMetadata([outputMetadata, payloadMetadata]);

            const confidenceValue =
              typeof outputRecord?.confidence === "number"
                ? (outputRecord.confidence as number)
                : typeof payload.confidence === "number"
                  ? (payload.confidence as number)
                  : undefined;

            const contentSource = outputRecord?.content ?? outputRecord?.summary ?? outputRaw ?? payload.output;
            const agentName = typeof payload.agent === "string" ? payload.agent : undefined;
            const confidenceBreakdown = extractConfidenceBreakdown(mergedMetadata);
            const toolMetadata = extractToolMetadata(mergedMetadata);

            const message: ChatMessage = {
              id: randomId(),
              role: "assistant",
              agentName,
              confidence: confidenceValue,
              confidenceBreakdown,
              toolMetadata,
              metadata: mergedMetadata,
              content: toDisplayString(contentSource),
              timestamp: formatTimestamp(typeof payload.timestamp === "string" ? payload.timestamp : undefined),
            };

            if (message.confidenceBreakdown) {
              emitConfidenceAnalytics({
                agent: agentName ?? "unknown_agent",
                breakdown: message.confidenceBreakdown,
                confidence: message.confidence,
                taskId: activeTaskId,
                timestamp: new Date().toISOString(),
                source: "stream",
              });
            }

            setMessages((prev) => [...prev, message]);
            return;
          }

          if (eventName === "task_failed") {
            const errorMessage = typeof payload.error === "string" ? payload.error : "Task failed";
            const systemMessage: ChatMessage = {
              id: randomId(),
              role: "system",
              content: errorMessage,
              timestamp: formatTimestamp(),
            };
            setMessages((prev) => [...prev, systemMessage]);
            return;
          }

          if (eventName === "task_completed" && typeof payload.task_id === "string") {
            activeTaskId = payload.task_id;
          }
        };

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }
          buffer += decoder.decode(value, { stream: true });

          let boundary = buffer.indexOf("\n\n");
          while (boundary !== -1) {
            const chunk = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            boundary = buffer.indexOf("\n\n");

            if (!chunk.trim()) {
              continue;
            }

            let eventName = "";
            let dataPayload: Record<string, unknown> | null = null;
            const lines = chunk.split("\n");
            for (const line of lines) {
              if (line.startsWith("event:")) {
                eventName = line.slice(6).trim();
              } else if (line.startsWith("data:")) {
                const raw = line.slice(5).trim();
                try {
                  dataPayload = JSON.parse(raw);
                } catch (parseError) {
                  console.error("sse_parse_error", parseError);
                }
              }
            }

            if (eventName && dataPayload) {
              processEvent(eventName, dataPayload);
            }
          }
        }

        if (activeTaskId) {
          await refreshHistory(activeTaskId);
        }
      } catch (error) {
        console.error("task_stream_failed", error);
        setMessages((prev) => [
          ...prev,
          {
            id: randomId(),
            role: "system",
            content: "Sorry, something went wrong while processing your task.",
            timestamp: formatTimestamp(),
          },
        ]);
      } finally {
        setIsStreaming(false);
      }
    },
    [isStreaming, refreshHistory],
  );

  const value = useMemo(
    () => ({
      messages,
      history,
      currentTaskId,
      isStreaming,
      submitTask,
      refreshHistory,
      resetSession,
      toolEvents,
      mcpDiagnostics,
    }),
    [
      messages,
      history,
      currentTaskId,
      isStreaming,
      submitTask,
      refreshHistory,
      resetSession,
      toolEvents,
      mcpDiagnostics,
    ],
  );

  return <TaskContext.Provider value={value}>{children}</TaskContext.Provider>;
};

export const useTaskContext = () => {
  const context = useContext(TaskContext);
  if (!context) {
    throw new Error("useTaskContext must be used within a TaskProvider");
  }
  return context;
};
