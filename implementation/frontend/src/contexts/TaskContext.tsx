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

type LifecycleEventType = "agent_started" | "agent_completed" | "agent_failed" | "guardrail_triggered";

export interface GuardrailDecisionEvent {
  decision?: string;
  reason?: string;
  riskScore?: number;
  policyId?: string;
  metadata?: Record<string, unknown>;
}

export interface LifecycleEvent {
  id: string;
  type: LifecycleEventType;
  agent?: string;
  stepId?: string;
  timestamp: string;
  latencyMs?: number;
  error?: string;
  guardrail?: GuardrailDecisionEvent;
}

export interface TaskStatusMetricsState {
  agentsCompleted: number;
  agentsFailed: number;
  guardrailEvents: number;
  negotiationRounds: number | null;
}

export interface TaskStatusEventState {
  sequence: number;
  eventType: string;
  agent?: string | null;
  payload: Record<string, unknown>;
  createdAt: string;
}

export interface TaskStatusState {
  taskId: string;
  status: string;
  runId: string | null;
  prompt: string | null;
  metadata: Record<string, unknown>;
  outputs: Array<Record<string, unknown>>;
  plan: Record<string, unknown> | null;
  negotiation: Record<string, unknown> | null;
  guardrailDecisions: GuardrailDecisionEvent[];
  metrics: TaskStatusMetricsState;
  lastError: string | null;
  createdAt: string | null;
  updatedAt: string | null;
  events: TaskStatusEventState[];
}

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
  lifecycleEvents: LifecycleEvent[];
  taskStatus: TaskStatusState | null;
  refreshTaskStatus: (taskId: string) => Promise<void>;
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

const TASK_EVENT_SCHEMA = "neuraforge.task-event.v1";

interface TaskStreamEventEnvelope {
  version: number;
  schema?: string;
  event: string;
  type?: string;
  sequence: number;
  task_id: string;
  run_id?: string;
  timestamp?: string;
  payload?: unknown;
}

const isTaskStreamEnvelope = (value: unknown): value is TaskStreamEventEnvelope => {
  if (!isRecord(value)) return false;
  if (typeof value.event !== "string") return false;
  if (typeof value.sequence !== "number") return false;
  if (typeof value.task_id !== "string") return false;
  if ("schema" in value && typeof value.schema !== "string") return false;
  const payload = value.payload;
  if (payload === undefined || payload === null) {
    return true;
  }
  return isRecord(payload);
};

export const TaskProvider = ({ children }: PropsWithChildren) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([]);
  const [mcpDiagnostics, setMcpDiagnostics] = useState<MCPDiagnostics | null>(null);
  const [lifecycleEvents, setLifecycleEvents] = useState<LifecycleEvent[]>([]);
  const [taskStatus, setTaskStatus] = useState<TaskStatusState | null>(null);

  const resetSession = useCallback(() => {
    setMessages([]);
    setHistory([]);
    setCurrentTaskId(null);
    setIsStreaming(false);
    setToolEvents([]);
    setMcpDiagnostics(null);
    setLifecycleEvents([]);
    setTaskStatus(null);
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

  const refreshTaskStatus = useCallback(async (taskId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/tasks/${taskId}`);
      if (!response.ok) {
        if (response.status === 404) {
          setTaskStatus(null);
          return;
        }
        throw new Error(`Unable to fetch task status (${response.status})`);
      }
      const raw = (await response.json()) as Record<string, unknown>;
      if (!isRecord(raw)) {
        setTaskStatus(null);
        return;
      }

      const metricsRaw = isRecord(raw.metrics) ? (raw.metrics as Record<string, unknown>) : {};
      const guardrailsRaw = isRecord(raw.guardrails) ? (raw.guardrails as Record<string, unknown>) : undefined;
      const guardrailDecisionsRaw = Array.isArray(guardrailsRaw?.decisions)
        ? guardrailsRaw?.decisions
        : [];

      const guardrailDecisions: GuardrailDecisionEvent[] = guardrailDecisionsRaw
        .filter((item): item is Record<string, unknown> => isRecord(item))
        .map((item) => ({
          decision: typeof item.decision === "string" ? item.decision : undefined,
          reason: typeof item.reason === "string" ? item.reason : undefined,
          riskScore: typeof item.risk_score === "number" ? item.risk_score : undefined,
          policyId: typeof item.policy_id === "string" ? item.policy_id : undefined,
          metadata: isRecord(item.metadata) ? (item.metadata as Record<string, unknown>) : undefined,
        }));

      const eventsRaw = Array.isArray(raw.events) ? raw.events : [];
      const events: TaskStatusEventState[] = eventsRaw
        .filter((item): item is Record<string, unknown> => isRecord(item))
        .map((item) => ({
          sequence: typeof item.sequence === "number" ? item.sequence : 0,
          eventType: typeof item.event_type === "string" ? item.event_type : "unknown",
          agent: typeof item.agent === "string" ? item.agent : null,
          payload: isRecord(item.payload) ? (item.payload as Record<string, unknown>) : {},
          createdAt: typeof item.created_at === "string" ? item.created_at : new Date().toISOString(),
        }))
        .sort((a, b) => a.sequence - b.sequence);

      const mapped: TaskStatusState = {
        taskId: typeof raw.task_id === "string" ? raw.task_id : taskId,
        status: typeof raw.status === "string" ? raw.status : "unknown",
        runId: typeof raw.run_id === "string" ? raw.run_id : null,
        prompt: typeof raw.prompt === "string" ? raw.prompt : null,
        metadata: isRecord(raw.metadata) ? (raw.metadata as Record<string, unknown>) : {},
        outputs: Array.isArray(raw.outputs)
          ? raw.outputs.filter((item): item is Record<string, unknown> => isRecord(item)).map((item) => ({ ...item }))
          : [],
        plan: isRecord(raw.plan) ? (raw.plan as Record<string, unknown>) : null,
        negotiation: isRecord(raw.negotiation) ? (raw.negotiation as Record<string, unknown>) : null,
        guardrailDecisions,
        metrics: {
          agentsCompleted: typeof metricsRaw.agents_completed === "number" ? metricsRaw.agents_completed : 0,
          agentsFailed: typeof metricsRaw.agents_failed === "number" ? metricsRaw.agents_failed : 0,
          guardrailEvents: typeof metricsRaw.guardrail_events === "number" ? metricsRaw.guardrail_events : 0,
          negotiationRounds:
            typeof metricsRaw.negotiation_rounds === "number"
              ? metricsRaw.negotiation_rounds
              : metricsRaw.negotiation_rounds === null
                ? null
                : null,
        },
        lastError: typeof raw.last_error === "string" ? raw.last_error : null,
        createdAt: typeof raw.created_at === "string" ? raw.created_at : null,
        updatedAt: typeof raw.updated_at === "string" ? raw.updated_at : null,
        events,
      };

      setTaskStatus(mapped);
    } catch (error) {
      console.error("task_status_fetch_failed", error);
    }
  }, []);

  const submitTask = useCallback(
    async (prompt: string, metadata: Record<string, unknown> = {}) => {
      const trimmed = prompt.trim();
      if (!trimmed || isStreaming) {
        return;
      }

      const continuationTaskId = currentTaskId && typeof currentTaskId === "string" ? currentTaskId : null;

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
      setLifecycleEvents([]);
      setTaskStatus(null);

      try {
        const payload: Record<string, unknown> = {
          prompt: trimmed,
          metadata,
        };

        if (continuationTaskId) {
          payload.continuation_task_id = continuationTaskId;
        }

        const response = await fetch(`${API_BASE_URL}/api/v1/submit_task/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(payload),
        });

        if (!response.ok || !response.body) {
          throw new Error(`Streaming request failed (${response.status})`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let activeTaskId: string | null = null;

        const appendLifecycleEvent = (event: LifecycleEvent) => {
          setLifecycleEvents((previous) => [event, ...previous].slice(0, 100));
        };

        const processEvent = (
          eventName: string,
          payload: Record<string, unknown>,
          envelope?: TaskStreamEventEnvelope | null,
        ) => {
          const taskIdentifier =
            typeof payload.task_id === "string"
              ? (payload.task_id as string)
              : typeof envelope?.task_id === "string"
                ? envelope.task_id
                : undefined;
          const eventTimestamp =
            typeof payload.timestamp === "string"
              ? (payload.timestamp as string)
              : typeof envelope?.timestamp === "string"
                ? envelope.timestamp
                : undefined;
          if (envelope) {
            if (!("task_id" in payload) && taskIdentifier) {
              payload.task_id = taskIdentifier;
            }
            if (!("run_id" in payload) && typeof envelope.run_id === "string") {
              payload.run_id = envelope.run_id;
            }
            if (!("timestamp" in payload) && eventTimestamp) {
              payload.timestamp = eventTimestamp;
            }
            if (!("sequence" in payload)) {
              payload.sequence = envelope.sequence;
            }
          }

          if (eventName === "task_started") {
            if (taskIdentifier) {
              activeTaskId = taskIdentifier;
              setCurrentTaskId(activeTaskId);
            }
            return;
          }

          if (eventName === "tool_invocation" || eventName === "tool_invoked") {
            const toolName = typeof payload.tool === "string" ? payload.tool : "unknown.tool";
            const resolvedTool = typeof payload.resolved_tool === "string" ? payload.resolved_tool : undefined;
            const status = payload.status === "error" ? "error" : "success";
            const cached = typeof payload.cached === "boolean" ? payload.cached : undefined;
            const latencySeconds = typeof payload.latency === "number" ? payload.latency : undefined;
            const latencyMsExplicit = typeof payload.latency_ms === "number" ? payload.latency_ms : undefined;
            const latencyValue = typeof latencyMsExplicit === "number"
              ? latencyMsExplicit / 1000
              : latencySeconds;
            const composite = typeof payload.composite === "boolean" ? payload.composite : undefined;
            const errorMessage = typeof payload.error === "string" ? payload.error : undefined;
            const payloadKeys = Array.isArray(payload.payload_keys)
              ? payload.payload_keys.filter((item) => typeof item === "string")
              : undefined;
            const timestampIso = eventTimestamp;

            const event: ToolEvent = {
              id: randomId(),
              tool: toolName,
              resolvedTool,
              status,
              cached,
              latencyMs: typeof latencyMsExplicit === "number"
                ? latencyMsExplicit
                : typeof latencyValue === "number"
                  ? latencyValue * 1000
                  : undefined,
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

          if (eventName === "agent_started") {
            const agentName = typeof payload.agent === "string" ? payload.agent : undefined;
            const stepId = typeof payload.step_id === "string" ? payload.step_id : undefined;
            const timestampIso = eventTimestamp;
            appendLifecycleEvent({
              id: randomId(),
              type: "agent_started",
              agent: agentName,
              stepId,
              timestamp: formatTimestamp(timestampIso),
            });
            return;
          }

          if (eventName === "agent_failed") {
            const agentName = typeof payload.agent === "string" ? payload.agent : "unknown_agent";
            const timestampIso = eventTimestamp;
            const latencySeconds = typeof payload.latency === "number" ? payload.latency : undefined;
            const latencyMsExplicit = typeof payload.latency_ms === "number" ? payload.latency_ms : undefined;
            const latencyMs = typeof latencyMsExplicit === "number"
              ? latencyMsExplicit
              : typeof latencySeconds === "number"
                ? latencySeconds * 1000
                : undefined;
            const errorMessage = typeof payload.error === "string" ? payload.error : undefined;

            appendLifecycleEvent({
              id: randomId(),
              type: "agent_failed",
              agent: agentName,
              timestamp: formatTimestamp(timestampIso),
              latencyMs,
              error: errorMessage,
              stepId: typeof payload.step_id === "string" ? payload.step_id : undefined,
            });

            if (errorMessage) {
              setMessages((prev) => [
                ...prev,
                {
                  id: randomId(),
                  role: "system",
                  content: `Agent ${agentName} reported an error: ${errorMessage}`,
                  timestamp: formatTimestamp(timestampIso),
                },
              ]);
            }
            return;
          }

          if (eventName === "guardrail_triggered") {
            const timestampIso = eventTimestamp;
            appendLifecycleEvent({
              id: randomId(),
              type: "guardrail_triggered",
              agent: typeof payload.agent === "string" ? payload.agent : undefined,
              stepId: typeof payload.step_id === "string" ? payload.step_id : undefined,
              timestamp: formatTimestamp(timestampIso),
              guardrail: {
                decision: typeof payload.decision === "string" ? payload.decision : undefined,
                reason: typeof payload.reason === "string" ? payload.reason : undefined,
                riskScore: typeof payload.risk_score === "number" ? payload.risk_score : undefined,
                policyId: typeof payload.policy_id === "string" ? payload.policy_id : undefined,
                metadata: isRecord(payload.metadata) ? (payload.metadata as Record<string, unknown>) : undefined,
              },
            });
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
              timestamp: formatTimestamp(eventTimestamp),
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
            appendLifecycleEvent({
              id: randomId(),
              type: "agent_completed",
              agent: agentName,
              timestamp: formatTimestamp(eventTimestamp),
              latencyMs:
                typeof payload.latency_ms === "number"
                  ? payload.latency_ms
                  : typeof payload.latency === "number"
                    ? payload.latency * 1000
                    : undefined,
              stepId: typeof payload.step_id === "string" ? payload.step_id : undefined,
            });
            return;
          }

          if (eventName === "planner_failed") {
            const errorMessage =
              typeof payload.error === "string"
                ? (payload.error as string)
                : "Automatic planner failed to route this task.";
            setMessages((prev) => [
              ...prev,
              {
                id: randomId(),
                role: "system",
                content: errorMessage,
                timestamp: formatTimestamp(eventTimestamp),
              },
            ]);
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

          if (eventName === "task_completed" && taskIdentifier) {
            activeTaskId = taskIdentifier;
            setCurrentTaskId(taskIdentifier);
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
            let dataPayload: unknown = null;
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

            if (eventName && dataPayload !== null) {
              let envelope: TaskStreamEventEnvelope | null = null;
              let payloadRecord: Record<string, unknown> = {};

              if (isTaskStreamEnvelope(dataPayload)) {
                const envelopePayload = isRecord(dataPayload.payload) ? dataPayload.payload : {};
                envelope = { ...dataPayload, payload: envelopePayload };
                payloadRecord = { ...envelopePayload };
              } else if (isRecord(dataPayload)) {
                payloadRecord = { ...dataPayload };
              }

              if (envelope) {
                if (!("task_id" in payloadRecord)) {
                  payloadRecord.task_id = envelope.task_id;
                }
                if (!("timestamp" in payloadRecord) && typeof envelope.timestamp === "string") {
                  payloadRecord.timestamp = envelope.timestamp;
                }
                if (!("run_id" in payloadRecord) && typeof envelope.run_id === "string") {
                  payloadRecord.run_id = envelope.run_id;
                }
                payloadRecord.sequence = envelope.sequence;
                if (envelope.schema) {
                  payloadRecord.__schema = envelope.schema;
                } else if (TASK_EVENT_SCHEMA && !("schema" in payloadRecord)) {
                  payloadRecord.__schema = TASK_EVENT_SCHEMA;
                }
                payloadRecord.__version = envelope.version;
              }

              const effectiveEventName =
                envelope && typeof envelope.type === "string" && envelope.type
                  ? envelope.type
                  : envelope && typeof envelope.event === "string" && envelope.event
                    ? envelope.event
                    : eventName;

              processEvent(effectiveEventName, payloadRecord, envelope);
            }
          }
        }

        if (activeTaskId) {
          await refreshHistory(activeTaskId);
          await refreshTaskStatus(activeTaskId);
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
    [currentTaskId, isStreaming, refreshHistory, refreshTaskStatus],
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
      lifecycleEvents,
      taskStatus,
      refreshTaskStatus,
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
      lifecycleEvents,
      taskStatus,
      refreshTaskStatus,
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
