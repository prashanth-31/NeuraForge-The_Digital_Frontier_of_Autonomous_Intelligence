import { Brain, User } from "lucide-react";
import { cn } from "@/lib/utils";

interface MessageCardProps {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  agentName?: string;
  confidence?: number;
  confidenceBreakdown?: Record<string, number>;
  toolMetadata?: {
    name?: string;
    resolved?: string;
    cached?: boolean;
    latency?: number;
  };
}

const COMPONENT_ORDER = ["base", "evidence", "tool_reliability", "self_assessment"];

const MessageCard = ({
  role,
  content,
  timestamp,
  agentName,
  confidence,
  confidenceBreakdown,
  toolMetadata,
}: MessageCardProps) => {
  const isAssistant = role !== "user";
  const isSystem = role === "system";
  const breakdownEntries = confidenceBreakdown
    ? Object.entries(confidenceBreakdown).sort((a, b) => {
        const left = COMPONENT_ORDER.indexOf(a[0]);
        const right = COMPONENT_ORDER.indexOf(b[0]);
        const fallback = COMPONENT_ORDER.length;
        return (left === -1 ? fallback : left) - (right === -1 ? fallback : right);
      })
    : undefined;

  return (
    <div
      className={cn(
        "flex gap-4 p-6 rounded-xl transition-smooth animate-fade-in",
        isSystem
          ? "bg-muted"
          : isAssistant
            ? "bg-[hsl(var(--message-ai))]"
            : "bg-[hsl(var(--message-user))]"
    )}>
      <div className={cn(
        "w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0",
        isSystem ? "bg-muted-foreground/20" : isAssistant ? "bg-primary/10" : "bg-secondary/10"
      )}>
        {isAssistant ? (
          <Brain className="h-5 w-5 text-primary" />
        ) : (
          <User className="h-5 w-5 text-secondary" />
        )}
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-sm font-medium text-foreground flex items-center gap-2">
            {isSystem
              ? "NeuraForge"
              : isAssistant
                ? agentName ?? "NeuraForge Agent"
                : "You"}
            {typeof confidence === "number" && (
              <span className="text-[10px] uppercase tracking-wide text-muted-foreground bg-background/60 border border-border px-2 py-0.5 rounded-full">
                {(confidence * 100).toFixed(0)}% confidence
              </span>
            )}
          </span>
          <span className="text-xs text-muted-foreground">{timestamp}</span>
        </div>
        <div className="text-sm text-foreground/90 leading-relaxed whitespace-pre-wrap">
          {content}
        </div>
        {breakdownEntries && (
          <div className="mt-4 flex flex-wrap gap-2">
            {breakdownEntries.map(([key, value]) => (
              <span
                key={key}
                className="text-[10px] uppercase tracking-wide bg-background/70 border border-border px-2 py-0.5 rounded-full text-muted-foreground"
              >
                {`${key.replace(/_/g, " ")}: ${(value * 100).toFixed(0)}%`}
              </span>
            ))}
          </div>
        )}
        {toolMetadata && (
          <div className="mt-3 text-[11px] text-muted-foreground flex flex-wrap gap-3">
            {toolMetadata.name && <span>Tool: {toolMetadata.name}</span>}
            {toolMetadata.resolved && toolMetadata.resolved !== toolMetadata.name && (
              <span>Target: {toolMetadata.resolved}</span>
            )}
            {typeof toolMetadata.latency === "number" && (
              <span>Latency: {(toolMetadata.latency * 1000).toFixed(0)} ms</span>
            )}
            {typeof toolMetadata.cached === "boolean" && (
              <span>{toolMetadata.cached ? "Cached result" : "Live result"}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageCard;
