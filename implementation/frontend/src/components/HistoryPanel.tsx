import { type ReactNode, useMemo, useState } from "react";

import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  History,
  Loader,
  PlayCircle,
  ShieldAlert,
  TimerReset,
  Workflow,
} from "lucide-react";

import { useTaskContext } from "@/contexts/TaskContext";
import { cn } from "@/lib/utils";
import { ScrollArea } from "./ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Badge } from "./ui/badge";

interface HistoryPanelProps {
  className?: string;
  embedded?: boolean;
}

const HistoryPanel = ({ className, embedded = false }: HistoryPanelProps) => {
  const {
    history,
    currentTaskId,
    isStreaming,
    toolEvents,
    mcpDiagnostics,
    lifecycleEvents,
    taskStatus,
  } = useTaskContext();
  const hasHistory = history.length > 0;
  const hasToolEvents = toolEvents.length > 0;
  const hasTimelineEvents = lifecycleEvents.length > 0;
  const componentOrder = ["base", "evidence", "tool_reliability", "self_assessment"];
  const [tabValue, setTabValue] = useState("history");

  const guardrailDecisions = useMemo(() => taskStatus?.guardrailDecisions ?? [], [taskStatus]);
  const hasGuardrailDecisions = guardrailDecisions.length > 0;

  const formattedDiagnostics = useMemo(() => {
    if (!mcpDiagnostics) {
      return null;
    }

    const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null && !Array.isArray(value);

    const toStringSafe = (value: unknown) => {
      if (typeof value === "string") return value;
      if (typeof value === "number" || typeof value === "boolean") return String(value);
      if (value === null || value === undefined) return "—";
      try {
        return JSON.stringify(value, null, 2);
      } catch {
        return String(value);
      }
    };

    const health = isRecord((mcpDiagnostics as Record<string, unknown>).last_health)
      ? (mcpDiagnostics as Record<string, unknown>).last_health as Record<string, unknown>
      : null;
    const lastInvocation = isRecord((mcpDiagnostics as Record<string, unknown>).last_invocation)
      ? (mcpDiagnostics as Record<string, unknown>).last_invocation as Record<string, unknown>
      : null;
    const circuit = isRecord((mcpDiagnostics as Record<string, unknown>).circuit)
      ? (mcpDiagnostics as Record<string, unknown>).circuit as Record<string, unknown>
      : null;

    return {
      enabled: toStringSafe((mcpDiagnostics as Record<string, unknown>).enabled),
      endpoint: toStringSafe((mcpDiagnostics as Record<string, unknown>).endpoint),
      catalogSize: toStringSafe((mcpDiagnostics as Record<string, unknown>).catalog_size),
      lastHealthStatus: toStringSafe(health?.status),
      lastHealthTimestamp: toStringSafe(health?.timestamp),
      lastHealthError: health?.error ? toStringSafe(health.error) : undefined,
      lastInvocation: lastInvocation
        ? {
            tool: toStringSafe(lastInvocation.tool),
            resolved: toStringSafe(lastInvocation.resolved),
            cached: toStringSafe(lastInvocation.cached),
            latency: toStringSafe(lastInvocation.latency),
            timestamp: toStringSafe(lastInvocation.timestamp),
          }
        : null,
      lastError: (mcpDiagnostics as Record<string, unknown>).last_error
        ? toStringSafe((mcpDiagnostics as Record<string, unknown>).last_error)
        : undefined,
      circuit: circuit
        ? {
            state: toStringSafe(circuit.state),
            failureCount: toStringSafe(circuit.failure_count),
            openedAt: toStringSafe(circuit.opened_at),
          }
        : null,
      aliasesCount: isRecord((mcpDiagnostics as Record<string, unknown>).aliases)
        ? Object.keys((mcpDiagnostics as Record<string, unknown>).aliases as Record<string, unknown>).length
        : undefined,
    };
  }, [mcpDiagnostics]);

  const renderNoDataState = (icon: ReactNode, message: string) => (
    <div className="h-full flex flex-col items-center justify-center text-center text-xs text-muted-foreground gap-3 px-4">
      <div className="w-10 h-10 rounded-full bg-muted flex items-center justify-center">{icon}</div>
      <p>{message}</p>
    </div>
  );

  const formatLatency = (value?: number) => {
    if (!value || Number.isNaN(value)) {
      return null;
    }
    if (value >= 1000) {
      return `${(value / 1000).toFixed(1)} s`;
    }
    return `${value.toFixed(0)} ms`;
  };

  const formatTimestamp = (iso: string | null | undefined) => {
    if (!iso) {
      return "—";
    }
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) {
      return "—";
    }
    return `${date.toLocaleDateString()} ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
  };

  const renderTimelineIcon = (type: string) => {
    switch (type) {
      case "agent_started":
        return <PlayCircle className="h-4 w-4 text-primary" />;
      case "agent_completed":
        return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
      case "agent_failed":
        return <AlertTriangle className="h-4 w-4 text-destructive" />;
      case "guardrail_triggered":
        return <ShieldAlert className="h-4 w-4 text-amber-500" />;
      default:
        return <TimerReset className="h-4 w-4 text-muted-foreground" />;
    }
  };

  return (
    <aside
      className={cn(
        embedded
          ? "flex flex-col h-full overflow-hidden"
          : "hidden lg:flex w-80 xl:w-96 flex-col border-l border-slate-200/60 bg-white/95 backdrop-blur-sm shadow-soft sticky top-16 h-[calc(100vh-4rem)]",
        className,
      )}
    >
      {!embedded && (
        <div className="p-4 border-b border-slate-200/60 bg-gradient-to-r from-slate-50 to-white">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-100 to-primary-200 flex items-center justify-center">
              <History className="h-4 w-4 text-primary-600" />
            </div>
            <div>
              <h2 className="font-semibold text-sm text-foreground">History</h2>
              <p className="text-xs text-muted-foreground">
                {currentTaskId ? `Task ${currentTaskId.slice(0, 8)}…` : "Past conversations"}
              </p>
            </div>
          </div>
        </div>
      )}

      <Tabs value={tabValue} onValueChange={setTabValue} className="flex-1 flex flex-col overflow-hidden">
        <div className="px-3 pt-3">
          <TabsList className="grid grid-cols-4 w-full">
            <TabsTrigger value="history">History</TabsTrigger>
            <TabsTrigger value="timeline">Timeline</TabsTrigger>
            <TabsTrigger value="tools">Tools</TabsTrigger>
            <TabsTrigger value="diagnostics">Diagnostics</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent
          value="history"
          className="flex-1 overflow-hidden data-[state=inactive]:hidden px-3 pb-3"
        >
          <ScrollArea className="h-full pr-2">
            {hasHistory ? (
              <div className="space-y-3">
                {history.map((entry, index) => (
                  <div key={`${entry.agent}-${index}`} className="p-4 rounded-xl border border-slate-200/60 bg-gradient-to-br from-white to-slate-50/50 shadow-xs hover:shadow-soft transition-shadow duration-200">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold text-foreground">{entry.agent}</span>
                      {typeof entry.confidence === "number" && (
                        <span className="text-[11px] font-semibold text-primary-600 bg-primary-50 px-2 py-0.5 rounded-md">
                          {(entry.confidence * 100).toFixed(0)}%
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap">
                      {entry.content}
                    </p>
                    {entry.confidenceBreakdown && (
                      <div className="flex flex-wrap gap-1.5 mt-3">
                        {Object.entries(entry.confidenceBreakdown)
                          .sort((a, b) => {
                            const left = componentOrder.indexOf(a[0]);
                            const right = componentOrder.indexOf(b[0]);
                            const fallback = componentOrder.length;
                            return (left === -1 ? fallback : left) - (right === -1 ? fallback : right);
                          })
                          .map(([key, value]) => (
                            <span
                              key={key}
                              className="text-[10px] uppercase tracking-wider font-medium bg-slate-100 border border-slate-200/60 px-2 py-0.5 rounded-md text-slate-600"
                            >
                              {`${key.replace(/_/g, " ")}: ${(value * 100).toFixed(0)}%`}
                            </span>
                          ))}
                      </div>
                    )}
                    {entry.toolMetadata && (
                      <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground mt-2">
                        {entry.toolMetadata.name && <span className="bg-slate-50 px-1.5 py-0.5 rounded">Tool: {entry.toolMetadata.name}</span>}
                        {entry.toolMetadata.resolved && entry.toolMetadata.resolved !== entry.toolMetadata.name && (
                          <span className="bg-slate-50 px-1.5 py-0.5 rounded">Target: {entry.toolMetadata.resolved}</span>
                        )}
                        {typeof entry.toolMetadata.latency === "number" && (
                          <span className="bg-slate-50 px-1.5 py-0.5 rounded">Latency: {(entry.toolMetadata.latency * 1000).toFixed(0)} ms</span>
                        )}
                        {typeof entry.toolMetadata.cached === "boolean" && (
                          <span className="bg-slate-50 px-1.5 py-0.5 rounded">{entry.toolMetadata.cached ? "Cached result" : "Live result"}</span>
                        )}
                      </div>
                    )}
                    <div className="flex items-center gap-1.5 text-[11px] text-slate-400 mt-3">
                      <Clock className="h-3 w-3" />
                      <span>{entry.timestamp}</span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              renderNoDataState(
                isStreaming ? <Loader className="h-5 w-5 animate-spin" /> : <History className="h-5 w-5" />, 
                isStreaming && currentTaskId
                  ? "Agents are working. Results will appear here shortly."
                  : "Submit a task to build a timeline of agent insights.",
              )
            )}
          </ScrollArea>
        </TabsContent>

        <TabsContent
          value="timeline"
          className="flex-1 overflow-hidden data-[state=inactive]:hidden px-3 pb-3"
        >
          <ScrollArea className="h-full pr-2">
            {hasTimelineEvents ? (
              <div className="space-y-3">
                {lifecycleEvents.map((event) => {
                  const latency = formatLatency(event.latencyMs);
                  const decisionLabel = event.guardrail?.decision
                    ? event.guardrail.decision.replace(/_/g, " ")
                    : undefined;
                  return (
                    <div key={event.id} className="p-4 rounded-xl border border-slate-200/60 bg-gradient-to-br from-white to-slate-50/50 shadow-xs hover:shadow-soft transition-shadow duration-200">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-start gap-2.5">
                          <div className="mt-0.5 w-7 h-7 rounded-lg bg-slate-100 flex items-center justify-center">{renderTimelineIcon(event.type)}</div>
                          <div>
                            <p className="text-sm font-semibold text-foreground capitalize">
                              {event.type.replace(/_/g, " ")}
                            </p>
                            <p className="text-[11px] text-muted-foreground">
                              {event.agent ? `Agent ${event.agent}` : "System"}
                              {event.stepId ? ` • ${event.stepId}` : ""}
                            </p>
                          </div>
                        </div>
                        <span className="text-[10px] text-slate-400 bg-slate-50 px-1.5 py-0.5 rounded">{event.timestamp}</span>
                      </div>
                      <div className="mt-3 text-[11px] text-muted-foreground space-y-1.5">
                        {latency && <div className="bg-slate-50 inline-block px-1.5 py-0.5 rounded">Latency: {latency}</div>}
                        {event.error && (
                          <div className="text-rose-600 bg-rose-50 border border-rose-200/60 rounded-lg px-3 py-2">
                            {event.error}
                          </div>
                        )}
                        {event.guardrail && (
                          <div className="space-y-1.5">
                            {decisionLabel && (
                              <Badge variant="destructive" className="text-[10px] uppercase tracking-wider">
                                {decisionLabel}
                              </Badge>
                            )}
                            {event.guardrail.reason && <div>Reason: {event.guardrail.reason}</div>}
                            {typeof event.guardrail.riskScore === "number" && (
                              <div>Risk score: {event.guardrail.riskScore.toFixed(2)}</div>
                            )}
                            {event.guardrail.policyId && <div>Policy: {event.guardrail.policyId}</div>}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              renderNoDataState(
                isStreaming ? <Loader className="h-5 w-5 animate-spin" /> : <Clock className="h-5 w-5" />,
                isStreaming
                  ? "Tracking agent lifecycle..."
                  : "Run a task to view agent timelines and guardrail activity.",
              )
            )}
          </ScrollArea>
        </TabsContent>

        <TabsContent
          value="tools"
          className="flex-1 overflow-hidden data-[state=inactive]:hidden px-3 pb-3"
        >
          <ScrollArea className="h-full pr-2">
            {hasToolEvents ? (
              <div className="space-y-3">
                {toolEvents.map((event) => (
                  <div key={event.id} className="p-4 rounded-xl border border-slate-200/60 bg-gradient-to-br from-white to-slate-50/50 shadow-xs hover:shadow-soft transition-shadow duration-200">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2.5">
                        <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ backgroundColor: event.status === "success" ? "rgb(236 253 245)" : "rgb(255 241 242)" }}>
                          {event.status === "success" ? (
                            <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                          ) : (
                            <AlertTriangle className="h-4 w-4 text-rose-500" />
                          )}
                        </div>
                        <div>
                          <p className="text-sm font-semibold text-foreground break-all">{event.tool}</p>
                          {event.resolvedTool && event.resolvedTool !== event.tool && (
                            <p className="text-[11px] text-muted-foreground break-all">→ {event.resolvedTool}</p>
                          )}
                        </div>
                      </div>
                      <Badge variant={event.status === "success" ? "success" : "destructive"} className="text-[10px]">
                        {event.status === "success" ? "Success" : "Error"}
                      </Badge>
                    </div>
                    <div className="mt-3 flex flex-wrap gap-1.5 text-[11px] text-muted-foreground">
                      <span className="bg-slate-50 px-1.5 py-0.5 rounded">{event.timestamp}</span>
                      {typeof event.latencyMs === "number" && (
                        <span className="bg-slate-50 px-1.5 py-0.5 rounded">Latency: {event.latencyMs.toFixed(0)} ms</span>
                      )}
                      {typeof event.cached === "boolean" && <span className="bg-slate-50 px-1.5 py-0.5 rounded">{event.cached ? "Cached" : "Live"}</span>}
                      {event.composite && <span className="bg-primary-50 text-primary-600 px-1.5 py-0.5 rounded">Composite flow</span>}
                      {event.payloadKeys && event.payloadKeys.length > 0 && (
                        <span className="bg-slate-50 px-1.5 py-0.5 rounded">Payload: {event.payloadKeys.join(", ")}</span>
                      )}
                    </div>
                    {event.error && (
                      <p className="mt-2 text-[11px] text-rose-600 bg-rose-50 border border-rose-200/60 rounded-lg px-3 py-2">
                        {event.error}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              renderNoDataState(<Workflow className="h-5 w-5" />, "Tool telemetry will appear once a task runs.")
            )}
          </ScrollArea>
        </TabsContent>

        <TabsContent
          value="diagnostics"
          className="flex-1 overflow-hidden data-[state=inactive]:hidden px-3 pb-3"
        >
          <ScrollArea className="h-full pr-2">
            {taskStatus || formattedDiagnostics ? (
              <div className="space-y-3 text-xs text-muted-foreground">
                {taskStatus && (
                  <div className="space-y-3">
                    <div className="grid grid-cols-1 gap-2">
                      <div className="flex items-center justify-between">
                        <span>Status</span>
                        <span className="font-medium text-foreground capitalize">{taskStatus.status.replace(/_/g, " ")}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Run Id</span>
                        <span className="font-medium text-foreground break-all">
                          {taskStatus.runId ?? "—"}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Updated</span>
                        <span className="font-medium text-foreground">
                          {formatTimestamp(taskStatus.updatedAt)}
                        </span>
                      </div>
                    </div>

                    <div className="border border-border/50 rounded-lg p-3 bg-muted/20">
                      <p className="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">Metrics</p>
                      <div className="grid grid-cols-2 gap-2">
                        <div className="flex items-center justify-between">
                          <span>Agents completed</span>
                          <span className="font-medium text-foreground">{taskStatus.metrics.agentsCompleted}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span>Agents failed</span>
                          <span className="font-medium text-foreground">{taskStatus.metrics.agentsFailed}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span>Guardrail events</span>
                          <span className="font-medium text-foreground">{taskStatus.metrics.guardrailEvents}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span>Negotiation rounds</span>
                          <span className="font-medium text-foreground">
                            {typeof taskStatus.metrics.negotiationRounds === "number"
                              ? taskStatus.metrics.negotiationRounds
                              : "—"}
                          </span>
                        </div>
                      </div>
                      {taskStatus.lastError && (
                        <div className="mt-2 text-destructive bg-destructive/10 border border-destructive/30 rounded-md px-2 py-1">
                          {taskStatus.lastError}
                        </div>
                      )}
                    </div>

                    {hasGuardrailDecisions && (
                      <div className="border border-border/50 rounded-lg p-3 bg-muted/20">
                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">Guardrail decisions</p>
                        <div className="space-y-2">
                          {guardrailDecisions.map((decision, index) => (
                            <div key={`${decision.policyId ?? "decision"}-${index}`} className="space-y-1 border border-border/40 rounded-md px-2 py-2 bg-background/60">
                              <div className="flex items-center justify-between gap-2">
                                <span className="text-[11px] text-muted-foreground uppercase tracking-wide">
                                  {decision.decision ? decision.decision.replace(/_/g, " ") : "Decision"}
                                </span>
                                {decision.policyId && <Badge variant="secondary">{decision.policyId}</Badge>}
                              </div>
                              {decision.reason && <div className="text-xs text-muted-foreground">{decision.reason}</div>}
                              {typeof decision.riskScore === "number" && (
                                <div className="text-[11px] text-muted-foreground">Risk score {decision.riskScore.toFixed(2)}</div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {formattedDiagnostics && (
                  <>
                    <div className="grid grid-cols-1 gap-2">
                      <div className="flex items-center justify-between">
                        <span>Enabled</span>
                        <span className="font-medium text-foreground">{formattedDiagnostics.enabled}</span>
                      </div>
                      <div className="flex items-center justify-between gap-4">
                        <span>Endpoint</span>
                        <span className="font-medium text-foreground break-all text-right">
                          {formattedDiagnostics.endpoint}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Catalog Size</span>
                        <span className="font-medium text-foreground">{formattedDiagnostics.catalogSize}</span>
                      </div>
                      {typeof formattedDiagnostics.aliasesCount === "number" && (
                        <div className="flex items-center justify-between">
                          <span>Aliases Registered</span>
                          <span className="font-medium text-foreground">{formattedDiagnostics.aliasesCount}</span>
                        </div>
                      )}
                    </div>

                    <div className="border border-border/50 rounded-lg p-3 bg-muted/20">
                      <p className="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">Last Health Check</p>
                      <div className="space-y-1">
                        <div className="flex items-center justify-between">
                          <span>Status</span>
                          <span className="font-medium text-foreground">{formattedDiagnostics.lastHealthStatus}</span>
                        </div>
                        <div className="flex items-center justify-between gap-4">
                          <span>Timestamp</span>
                          <span className="font-medium text-foreground break-all text-right">
                            {formattedDiagnostics.lastHealthTimestamp}
                          </span>
                        </div>
                        {formattedDiagnostics.lastHealthError && (
                          <div className="text-destructive text-[11px] bg-destructive/10 border border-destructive/30 rounded-md px-2 py-1">
                            {formattedDiagnostics.lastHealthError}
                          </div>
                        )}
                      </div>
                    </div>

                    {formattedDiagnostics.lastInvocation && (
                      <div className="border border-border/50 rounded-lg p-3 bg-muted/20">
                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">Last Invocation</p>
                        <div className="grid grid-cols-1 gap-1">
                          <div className="flex items-center justify-between gap-4">
                            <span>Tool</span>
                            <span className="font-medium text-foreground break-all text-right">
                              {formattedDiagnostics.lastInvocation.tool}
                            </span>
                          </div>
                          {formattedDiagnostics.lastInvocation.resolved && (
                            <div className="flex items-center justify-between gap-4">
                              <span>Resolved</span>
                              <span className="font-medium text-foreground break-all text-right">
                                {formattedDiagnostics.lastInvocation.resolved}
                              </span>
                            </div>
                          )}
                          <div className="flex items-center justify-between">
                            <span>Cached</span>
                            <span className="font-medium text-foreground">
                              {formattedDiagnostics.lastInvocation.cached}
                            </span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span>Latency</span>
                            <span className="font-medium text-foreground">
                              {formattedDiagnostics.lastInvocation.latency}
                            </span>
                          </div>
                          <div className="flex items-center justify-between gap-4">
                            <span>Timestamp</span>
                            <span className="font-medium text-foreground break-all text-right">
                              {formattedDiagnostics.lastInvocation.timestamp}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}

                    {formattedDiagnostics.circuit && (
                      <div className="border border-border/50 rounded-lg p-3 bg-muted/20">
                        <p className="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">Circuit Breaker</p>
                        <div className="grid grid-cols-1 gap-1">
                          <div className="flex items-center justify-between">
                            <span>State</span>
                            <span className="font-medium text-foreground">{formattedDiagnostics.circuit.state}</span>
                          </div>
                          <div className="flex items-center justify-between">
                            <span>Failures</span>
                            <span className="font-medium text-foreground">{formattedDiagnostics.circuit.failureCount}</span>
                          </div>
                          <div className="flex items-center justify-between gap-4">
                            <span>Opened</span>
                            <span className="font-medium text-foreground break-all text-right">
                              {formattedDiagnostics.circuit.openedAt}
                            </span>
                          </div>
                        </div>
                      </div>
                    )}

                    {formattedDiagnostics.lastError && (
                      <div className="text-destructive text-[11px] bg-destructive/10 border border-destructive/30 rounded-md px-2 py-1">
                        {formattedDiagnostics.lastError}
                      </div>
                    )}
                  </>
                )}
              </div>
            ) : (
              renderNoDataState(<Workflow className="h-5 w-5" />, "Diagnostics will appear after tooling initializes.")
            )}
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </aside>
  );
};

export default HistoryPanel;
