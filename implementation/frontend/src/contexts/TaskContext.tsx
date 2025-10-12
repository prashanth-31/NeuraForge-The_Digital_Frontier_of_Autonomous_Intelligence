import { API_BASE_URL } from "@/lib/api";
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
}

export interface HistoryEntry {
  agent: string;
  content: string;
  confidence: number;
  timestamp: string;
}

interface TaskContextValue {
  messages: ChatMessage[];
  history: HistoryEntry[];
  currentTaskId: string | null;
  isStreaming: boolean;
  submitTask: (prompt: string, metadata?: Record<string, unknown>) => Promise<void>;
  refreshHistory: (taskId: string) => Promise<void>;
  resetSession: () => void;
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

const parseHistoryContent = (payload: unknown): string => {
  if (payload === null || payload === undefined) {
    return "";
  }
  if (typeof payload === "string") {
    return payload;
  }
  if (typeof payload === "object") {
    const record = payload as Record<string, unknown>;
    if (typeof record.text === "string") {
      return record.text;
    }
    return JSON.stringify(record, null, 2);
  }
  return String(payload);
};

const randomId = () => (typeof crypto !== "undefined" && crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2));

export const TaskProvider = ({ children }: PropsWithChildren) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const resetSession = useCallback(() => {
    setMessages([]);
    setHistory([]);
    setCurrentTaskId(null);
    setIsStreaming(false);
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
      const mapped = data.map((item) => ({
        agent: item.agent,
        content: parseHistoryContent(item.content),
        confidence: item.confidence,
        timestamp: formatTimestamp(item.timestamp),
      }));
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

          if (eventName === "agent_completed") {
            const output = payload.output as Record<string, unknown> | string | undefined;
            const contentPayload = typeof output === "string" ? output : output?.content ?? payload.output;
            const confidenceValue =
              typeof (output as Record<string, unknown> | undefined)?.confidence === "number"
                ? ((output as Record<string, unknown>).confidence as number)
                : typeof payload.confidence === "number"
                  ? (payload.confidence as number)
                  : undefined;
            const message: ChatMessage = {
              id: randomId(),
              role: "assistant",
              agentName: typeof payload.agent === "string" ? payload.agent : undefined,
              confidence: confidenceValue,
              content: parseHistoryContent(contentPayload),
              timestamp: formatTimestamp(typeof payload.timestamp === "string" ? payload.timestamp : undefined),
            };
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
    () => ({ messages, history, currentTaskId, isStreaming, submitTask, refreshHistory, resetSession }),
    [messages, history, currentTaskId, isStreaming, submitTask, refreshHistory, resetSession],
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
