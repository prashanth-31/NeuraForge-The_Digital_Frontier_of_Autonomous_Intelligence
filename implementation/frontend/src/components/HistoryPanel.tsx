import { type ReactNode, useMemo, useState } from "react";

import { AlertTriangle, CheckCircle2, Clock, History, Loader, Workflow } from "lucide-react";

import { useTaskContext } from "@/contexts/TaskContext";
import { cn } from "@/lib/utils";
import { ScrollArea } from "./ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { Badge } from "./ui/badge";

interface HistoryPanelProps {
  className?: string;
}

const HistoryPanel = ({ className }: HistoryPanelProps) => {
  const { history, currentTaskId, isStreaming, toolEvents, mcpDiagnostics } = useTaskContext();
  const hasHistory = history.length > 0;
  const hasToolEvents = toolEvents.length > 0;
  const componentOrder = ["base", "evidence", "tool_reliability", "self_assessment"];
  const [tabValue, setTabValue] = useState("history");

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

  return (
    <aside
      className={cn(
        "hidden lg:flex w-80 xl:w-96 flex-col border-l border-border bg-card shadow-soft sticky top-16 h-[calc(100vh-4rem)]",
        className,
      )}
    >
      <div className="p-4 border-b border-border">
        <h2 className="font-semibold text-sm text-foreground">History</h2>
        <p className="text-xs text-muted-foreground mt-1">
          {currentTaskId ? `Task ${currentTaskId.slice(0, 8)}…` : "Past conversations"}
        </p>
      </div>

      <Tabs value={tabValue} onValueChange={setTabValue} className="flex-1 flex flex-col overflow-hidden">
        <div className="px-3 pt-3">
          <TabsList className="grid grid-cols-3 w-full">
            <TabsTrigger value="history">History</TabsTrigger>
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
                  <div key={`${entry.agent}-${index}`} className="p-3 rounded-lg border border-border/40 bg-muted/30">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-foreground">{entry.agent}</span>
                      {typeof entry.confidence === "number" && (
                        <span className="text-[11px] font-semibold text-primary">
                          {(entry.confidence * 100).toFixed(0)}% confidence
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed whitespace-pre-wrap">
                      {entry.content}
                    </p>
                    {entry.confidenceBreakdown && (
                      <div className="flex flex-wrap gap-2 mt-2">
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
                              className="text-[10px] uppercase tracking-wide bg-background/70 border border-border px-2 py-0.5 rounded-full text-muted-foreground"
                            >
                              {`${key.replace(/_/g, " ")}: ${(value * 100).toFixed(0)}%`}
                            </span>
                          ))}
                      </div>
                    )}
                    {entry.toolMetadata && (
                      <div className="flex flex-wrap gap-2 text-[11px] text-muted-foreground mt-2">
                        {entry.toolMetadata.name && <span>Tool: {entry.toolMetadata.name}</span>}
                        {entry.toolMetadata.resolved && entry.toolMetadata.resolved !== entry.toolMetadata.name && (
                          <span>Target: {entry.toolMetadata.resolved}</span>
                        )}
                        {typeof entry.toolMetadata.latency === "number" && (
                          <span>Latency: {(entry.toolMetadata.latency * 1000).toFixed(0)} ms</span>
                        )}
                        {typeof entry.toolMetadata.cached === "boolean" && (
                          <span>{entry.toolMetadata.cached ? "Cached result" : "Live result"}</span>
                        )}
                      </div>
                    )}
                    <div className="flex items-center gap-1 text-[11px] text-muted-foreground mt-2">
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
          value="tools"
          className="flex-1 overflow-hidden data-[state=inactive]:hidden px-3 pb-3"
        >
          <ScrollArea className="h-full pr-2">
            {hasToolEvents ? (
              <div className="space-y-3">
                {toolEvents.map((event) => (
                  <div key={event.id} className="p-3 rounded-lg border border-border/40 bg-muted/20">
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        {event.status === "success" ? (
                          <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 text-destructive" />
                        )}
                        <div>
                          <p className="text-sm font-medium text-foreground break-all">{event.tool}</p>
                          {event.resolvedTool && event.resolvedTool !== event.tool && (
                            <p className="text-[11px] text-muted-foreground break-all">→ {event.resolvedTool}</p>
                          )}
                        </div>
                      </div>
                      <Badge variant={event.status === "success" ? "secondary" : "destructive"}>
                        {event.status === "success" ? "Success" : "Error"}
                      </Badge>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
                      <span>{event.timestamp}</span>
                      {typeof event.latencyMs === "number" && (
                        <span>Latency: {event.latencyMs.toFixed(0)} ms</span>
                      )}
                      {typeof event.cached === "boolean" && <span>{event.cached ? "Cached" : "Live"}</span>}
                      {event.composite && <span>Composite flow</span>}
                      {event.payloadKeys && event.payloadKeys.length > 0 && (
                        <span>Payload: {event.payloadKeys.join(", ")}</span>
                      )}
                    </div>
                    {event.error && (
                      <p className="mt-2 text-[11px] text-destructive bg-destructive/10 border border-destructive/30 rounded-md px-2 py-1">
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
            {formattedDiagnostics ? (
              <div className="space-y-3 text-xs text-muted-foreground">
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
