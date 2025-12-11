import { useState } from "react";
import {
  Brain,
  Lightbulb,
  AlertTriangle,
  Wrench,
  ChevronDown,
  ChevronRight,
  Activity,
  Sparkles,
  Target,
  Search,
  GitBranch,
  Zap,
  RefreshCw,
  ArrowRightLeft,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "./ui/badge";

// ────────────────────────────────────────────────────────────────────────────────
// Types for Agent Thinking Events
// ────────────────────────────────────────────────────────────────────────────────

export type ThinkingEventType =
  | "agent_thinking"
  | "agent_planning"
  | "agent_tool_deciding"
  | "agent_tool_progress"
  | "agent_evaluating"
  | "agent_finding"
  | "agent_uncertainty"
  | "agent_collaboration"
  | "agent_handoff"
  | "parallel_start"
  | "parallel_complete"
  | "replan_triggered";

export interface ThinkingEvent {
  id: string;
  eventType: ThinkingEventType;
  agent: string;
  thought: string;
  stepIndex?: number;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

interface AgentActivityPanelProps {
  thinkingEvents: ThinkingEvent[];
  isStreaming?: boolean;
  className?: string;
}

// ────────────────────────────────────────────────────────────────────────────────
// Event Type Configuration
// ────────────────────────────────────────────────────────────────────────────────

const EVENT_CONFIG: Record<
  ThinkingEventType,
  {
    icon: typeof Brain;
    label: string;
    color: string;
    bgColor: string;
    borderColor: string;
  }
> = {
  agent_thinking: {
    icon: Brain,
    label: "Thinking",
    color: "text-primary-600",
    bgColor: "bg-primary-50",
    borderColor: "border-primary-200",
  },
  agent_planning: {
    icon: Target,
    label: "Planning",
    color: "text-indigo-600",
    bgColor: "bg-indigo-50",
    borderColor: "border-indigo-200",
  },
  agent_tool_deciding: {
    icon: Wrench,
    label: "Tool Selection",
    color: "text-amber-600",
    bgColor: "bg-amber-50",
    borderColor: "border-amber-200",
  },
  agent_tool_progress: {
    icon: Activity,
    label: "Tool Progress",
    color: "text-blue-600",
    bgColor: "bg-blue-50",
    borderColor: "border-blue-200",
  },
  agent_evaluating: {
    icon: Search,
    label: "Evaluating",
    color: "text-purple-600",
    bgColor: "bg-purple-50",
    borderColor: "border-purple-200",
  },
  agent_finding: {
    icon: Lightbulb,
    label: "Finding",
    color: "text-emerald-600",
    bgColor: "bg-emerald-50",
    borderColor: "border-emerald-200",
  },
  agent_uncertainty: {
    icon: AlertTriangle,
    label: "Uncertainty",
    color: "text-orange-600",
    bgColor: "bg-orange-50",
    borderColor: "border-orange-200",
  },
  agent_collaboration: {
    icon: Sparkles,
    label: "Collaboration",
    color: "text-pink-600",
    bgColor: "bg-pink-50",
    borderColor: "border-pink-200",
  },
  agent_handoff: {
    icon: ArrowRightLeft,
    label: "Handoff",
    color: "text-cyan-600",
    bgColor: "bg-cyan-50",
    borderColor: "border-cyan-200",
  },
  parallel_start: {
    icon: GitBranch,
    label: "Parallel Start",
    color: "text-violet-600",
    bgColor: "bg-violet-50",
    borderColor: "border-violet-200",
  },
  parallel_complete: {
    icon: Zap,
    label: "Parallel Complete",
    color: "text-green-600",
    bgColor: "bg-green-50",
    borderColor: "border-green-200",
  },
  replan_triggered: {
    icon: RefreshCw,
    label: "Re-planning",
    color: "text-rose-600",
    bgColor: "bg-rose-50",
    borderColor: "border-rose-200",
  },
};

// ────────────────────────────────────────────────────────────────────────────────
// Single Thinking Event Card
// ────────────────────────────────────────────────────────────────────────────────

interface ThinkingEventCardProps {
  event: ThinkingEvent;
  isLatest?: boolean;
}

const ThinkingEventCard = ({ event, isLatest }: ThinkingEventCardProps) => {
  const config = EVENT_CONFIG[event.eventType] || EVENT_CONFIG.agent_thinking;
  const Icon = config.icon;

  return (
    <div
      className={cn(
        "relative flex gap-3 p-3 rounded-lg border transition-all duration-300",
        config.bgColor,
        config.borderColor,
        isLatest && "ring-2 ring-primary-300 ring-opacity-50 animate-pulse-subtle"
      )}
    >
      {/* Icon */}
      <div
        className={cn(
          "w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0",
          "bg-white/60 shadow-sm"
        )}
      >
        <Icon className={cn("h-4 w-4", config.color)} />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <Badge
            variant="secondary"
            className={cn("text-[10px] px-1.5 py-0 font-medium", config.color)}
          >
            {config.label}
          </Badge>
          <span className="text-[10px] text-muted-foreground">{event.agent}</span>
          {event.stepIndex && (
            <span className="text-[10px] text-muted-foreground">
              Step {event.stepIndex}
            </span>
          )}
          <span className="text-[10px] text-muted-foreground ml-auto">
            {event.timestamp}
          </span>
        </div>
        <p className="text-sm text-foreground/80 leading-relaxed">{event.thought}</p>

        {/* Metadata badges */}
        {event.metadata && Object.keys(event.metadata).length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {event.metadata.step_type && (
              <span className="text-[9px] uppercase tracking-wider bg-white/50 px-1.5 py-0.5 rounded text-muted-foreground">
                {String(event.metadata.step_type)}
              </span>
            )}
            {event.metadata.tool && (
              <span className="text-[9px] uppercase tracking-wider bg-white/50 px-1.5 py-0.5 rounded text-muted-foreground">
                🔧 {String(event.metadata.tool)}
              </span>
            )}
            {typeof event.metadata.confidence === "number" && (
              <span className="text-[9px] uppercase tracking-wider bg-white/50 px-1.5 py-0.5 rounded text-muted-foreground">
                {(event.metadata.confidence as number * 100).toFixed(0)}% confident
              </span>
            )}
            {event.metadata.selected !== undefined && (
              <span
                className={cn(
                  "text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded",
                  event.metadata.selected
                    ? "bg-emerald-100 text-emerald-700"
                    : "bg-red-100 text-red-700"
                )}
              >
                {event.metadata.selected ? "✓ Selected" : "✗ Rejected"}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// ────────────────────────────────────────────────────────────────────────────────
// Agent Grouping
// ────────────────────────────────────────────────────────────────────────────────

interface AgentGroupProps {
  agentName: string;
  events: ThinkingEvent[];
  isExpanded: boolean;
  onToggle: () => void;
  isActiveAgent?: boolean;
}

const AgentGroup = ({
  agentName,
  events,
  isExpanded,
  onToggle,
  isActiveAgent,
}: AgentGroupProps) => {
  const latestEventId = events[0]?.id;

  return (
    <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-soft">
      {/* Agent Header */}
      <button
        onClick={onToggle}
        className={cn(
          "w-full flex items-center gap-3 p-3 text-left transition-colors",
          "hover:bg-slate-50",
          isActiveAgent && "bg-primary-50/50"
        )}
      >
        <div
          className={cn(
            "w-8 h-8 rounded-lg flex items-center justify-center",
            "bg-gradient-to-br from-primary-100 to-primary-200"
          )}
        >
          <Brain className="h-4 w-4 text-primary-600" />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm text-foreground">
              {formatAgentName(agentName)}
            </span>
            {isActiveAgent && (
              <span className="flex items-center gap-1 text-[10px] text-primary-600">
                <Activity className="h-3 w-3 animate-pulse" />
                Active
              </span>
            )}
          </div>
          <span className="text-xs text-muted-foreground">
            {events.length} thought{events.length !== 1 ? "s" : ""}
          </span>
        </div>
        {isExpanded ? (
          <ChevronDown className="h-4 w-4 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-4 w-4 text-muted-foreground" />
        )}
      </button>

      {/* Events List */}
      {isExpanded && (
        <div className="p-3 pt-0 space-y-2">
          {events.map((event) => (
            <ThinkingEventCard
              key={event.id}
              event={event}
              isLatest={event.id === latestEventId && isActiveAgent}
            />
          ))}
        </div>
      )}
    </div>
  );
};

// ────────────────────────────────────────────────────────────────────────────────
// Main Panel Component
// ────────────────────────────────────────────────────────────────────────────────

const formatAgentName = (name: string): string => {
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
};

export const AgentActivityPanel = ({
  thinkingEvents,
  isStreaming,
  className,
}: AgentActivityPanelProps) => {
  const [expandedAgents, setExpandedAgents] = useState<Set<string>>(new Set());

  // Group events by agent
  const groupedByAgent = thinkingEvents.reduce(
    (acc, event) => {
      if (!acc[event.agent]) {
        acc[event.agent] = [];
      }
      acc[event.agent].push(event);
      return acc;
    },
    {} as Record<string, ThinkingEvent[]>
  );

  // Determine active agent (most recent event)
  const activeAgent =
    thinkingEvents.length > 0 ? thinkingEvents[0].agent : null;

  // Auto-expand active agent
  const toggleAgent = (agentName: string) => {
    setExpandedAgents((prev) => {
      const next = new Set(prev);
      if (next.has(agentName)) {
        next.delete(agentName);
      } else {
        next.add(agentName);
      }
      return next;
    });
  };

  // Compute summary stats
  const stats = {
    totalThoughts: thinkingEvents.length,
    findings: thinkingEvents.filter((e) => e.eventType === "agent_finding").length,
    toolDecisions: thinkingEvents.filter(
      (e) => e.eventType === "agent_tool_deciding"
    ).length,
    uncertainties: thinkingEvents.filter(
      (e) => e.eventType === "agent_uncertainty"
    ).length,
  };

  if (thinkingEvents.length === 0) {
    return (
      <div
        className={cn(
          "flex flex-col items-center justify-center p-8 text-center",
          className
        )}
      >
        <div className="w-12 h-12 rounded-xl bg-slate-100 flex items-center justify-center mb-3">
          <Brain className="h-6 w-6 text-slate-400" />
        </div>
        <p className="text-sm text-muted-foreground">
          {isStreaming
            ? "Waiting for agent reasoning..."
            : "No agent activity yet"}
        </p>
        {isStreaming && (
          <div className="mt-2 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-primary-400 animate-bounce [animation-delay:0ms]" />
            <span className="w-1.5 h-1.5 rounded-full bg-primary-400 animate-bounce [animation-delay:150ms]" />
            <span className="w-1.5 h-1.5 rounded-full bg-primary-400 animate-bounce [animation-delay:300ms]" />
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col gap-4", className)}>
      {/* Summary Stats Bar */}
      <div className="flex items-center gap-3 px-1">
        <span className="text-xs font-medium text-foreground">
          Agent Activity
        </span>
        <div className="flex items-center gap-2 ml-auto">
          {stats.findings > 0 && (
            <Badge variant="secondary" className="text-[10px] gap-1">
              <Lightbulb className="h-3 w-3" />
              {stats.findings}
            </Badge>
          )}
          {stats.toolDecisions > 0 && (
            <Badge variant="secondary" className="text-[10px] gap-1">
              <Wrench className="h-3 w-3" />
              {stats.toolDecisions}
            </Badge>
          )}
          {stats.uncertainties > 0 && (
            <Badge variant="warning" className="text-[10px] gap-1">
              <AlertTriangle className="h-3 w-3" />
              {stats.uncertainties}
            </Badge>
          )}
        </div>
      </div>

      {/* Agent Groups */}
      <div className="space-y-3">
        {Object.entries(groupedByAgent).map(([agentName, events]) => (
          <AgentGroup
            key={agentName}
            agentName={agentName}
            events={events}
            isExpanded={
              expandedAgents.has(agentName) || agentName === activeAgent
            }
            onToggle={() => toggleAgent(agentName)}
            isActiveAgent={isStreaming && agentName === activeAgent}
          />
        ))}
      </div>
    </div>
  );
};

export default AgentActivityPanel;
