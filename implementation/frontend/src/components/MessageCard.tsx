import { useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  Brain,
  Sparkles,
  User,
  ChevronDown,
  ChevronRight,
  Lightbulb,
  Wrench,
  AlertTriangle,
  ArrowRight,
  ExternalLink,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "./ui/badge";

// ────────────────────────────────────────────────────────────────────────────────
// Types for Reasoning Data
// ────────────────────────────────────────────────────────────────────────────────

export interface ReasoningStep {
  step_type: string;
  thought: string;
  evidence?: string;
  confidence?: number;
  timestamp?: string;
}

export interface KeyFinding {
  claim: string;
  evidence?: string[];
  confidence?: number;
  source?: string;
  contradictions?: string[];
}

export interface ToolConsideration {
  tool_name: string;
  reason?: string;
  selected?: boolean;
  rejection_reason?: string;
}

export interface ReasoningData {
  reasoning_steps?: ReasoningStep[];
  key_findings?: KeyFinding[];
  tools_considered?: ToolConsideration[];
  uncertainties?: string[];
  suggested_followup?: string;
  rationale?: string;
}

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
  reasoning?: ReasoningData;
}

// ────────────────────────────────────────────────────────────────────────────────
// Sub-components for Reasoning Display
// ────────────────────────────────────────────────────────────────────────────────

const COMPONENT_ORDER = ["base", "evidence", "tool_reliability", "self_assessment"];

const STEP_TYPE_CONFIG: Record<string, { color: string; bgColor: string }> = {
  observation: { color: "text-blue-600", bgColor: "bg-blue-50" },
  analysis: { color: "text-purple-600", bgColor: "bg-purple-50" },
  hypothesis: { color: "text-indigo-600", bgColor: "bg-indigo-50" },
  decision: { color: "text-amber-600", bgColor: "bg-amber-50" },
  tool_selection: { color: "text-orange-600", bgColor: "bg-orange-50" },
  evaluation: { color: "text-emerald-600", bgColor: "bg-emerald-50" },
  synthesis: { color: "text-teal-600", bgColor: "bg-teal-50" },
  uncertainty: { color: "text-red-600", bgColor: "bg-red-50" },
};

interface ReasoningStepsProps {
  steps: ReasoningStep[];
}

const ReasoningSteps = ({ steps }: ReasoningStepsProps) => {
  if (steps.length === 0) return null;

  return (
    <div className="space-y-2">
      <h4 className="text-xs font-semibold text-slate-600 uppercase tracking-wider flex items-center gap-1.5">
        <Brain className="h-3.5 w-3.5" />
        Chain of Thought
      </h4>
      <div className="space-y-1.5">
        {steps.map((step, index) => {
          const config = STEP_TYPE_CONFIG[step.step_type] || STEP_TYPE_CONFIG.analysis;
          return (
            <div
              key={index}
              className={cn(
                "flex items-start gap-2 p-2.5 rounded-lg border border-slate-100",
                config.bgColor
              )}
            >
              <span
                className={cn(
                  "text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded",
                  config.color,
                  "bg-white/60"
                )}
              >
                {step.step_type.replace(/_/g, " ")}
              </span>
              <div className="flex-1 min-w-0">
                <p className="text-xs text-slate-700 leading-relaxed">{step.thought}</p>
                {step.evidence && (
                  <p className="text-[10px] text-slate-500 mt-1 italic">
                    Evidence: {step.evidence}
                  </p>
                )}
              </div>
              {typeof step.confidence === "number" && (
                <Badge variant="outline" className="text-[9px] shrink-0">
                  {(step.confidence * 100).toFixed(0)}%
                </Badge>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

interface KeyFindingsProps {
  findings: KeyFinding[];
}

const KeyFindings = ({ findings }: KeyFindingsProps) => {
  if (findings.length === 0) return null;

  return (
    <div className="space-y-2">
      <h4 className="text-xs font-semibold text-slate-600 uppercase tracking-wider flex items-center gap-1.5">
        <Lightbulb className="h-3.5 w-3.5" />
        Key Findings
      </h4>
      <div className="space-y-2">
        {findings.map((finding, index) => (
          <div
            key={index}
            className="p-2.5 rounded-lg bg-emerald-50 border border-emerald-100"
          >
            <p className="text-xs font-medium text-emerald-800">{finding.claim}</p>
            {finding.evidence && finding.evidence.length > 0 && (
              <div className="mt-1.5 flex flex-wrap gap-1">
                {finding.evidence.slice(0, 3).map((ev, i) => (
                  <span
                    key={i}
                    className="text-[10px] bg-white/60 text-emerald-700 px-1.5 py-0.5 rounded"
                  >
                    {ev.length > 50 ? ev.slice(0, 50) + "..." : ev}
                  </span>
                ))}
              </div>
            )}
            <div className="mt-1.5 flex items-center gap-2">
              {typeof finding.confidence === "number" && (
                <Badge
                  variant={finding.confidence >= 0.7 ? "success" : "warning"}
                  className="text-[9px]"
                >
                  {(finding.confidence * 100).toFixed(0)}% confident
                </Badge>
              )}
              {finding.source && (
                <span className="text-[10px] text-emerald-600 flex items-center gap-0.5">
                  <ExternalLink className="h-2.5 w-2.5" />
                  {finding.source}
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

interface ToolsConsideredProps {
  tools: ToolConsideration[];
}

const ToolsConsidered = ({ tools }: ToolsConsideredProps) => {
  if (tools.length === 0) return null;

  return (
    <div className="space-y-2">
      <h4 className="text-xs font-semibold text-slate-600 uppercase tracking-wider flex items-center gap-1.5">
        <Wrench className="h-3.5 w-3.5" />
        Tools Considered
      </h4>
      <div className="flex flex-wrap gap-2">
        {tools.map((tool, index) => (
          <div
            key={index}
            className={cn(
              "inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs",
              tool.selected
                ? "bg-primary-50 border border-primary-200 text-primary-700"
                : "bg-slate-50 border border-slate-200 text-slate-600"
            )}
          >
            <span className="font-medium">{tool.tool_name}</span>
            {tool.selected ? (
              <Badge variant="success" className="text-[9px] px-1 py-0">
                Used
              </Badge>
            ) : (
              <Badge variant="secondary" className="text-[9px] px-1 py-0">
                Skipped
              </Badge>
            )}
            {tool.reason && (
              <span className="text-[10px] text-slate-500 max-w-[150px] truncate">
                {tool.reason}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

interface UncertaintiesProps {
  uncertainties: string[];
}

const Uncertainties = ({ uncertainties }: UncertaintiesProps) => {
  if (uncertainties.length === 0) return null;

  return (
    <div className="space-y-2">
      <h4 className="text-xs font-semibold text-slate-600 uppercase tracking-wider flex items-center gap-1.5">
        <AlertTriangle className="h-3.5 w-3.5" />
        Uncertainties
      </h4>
      <div className="space-y-1.5">
        {uncertainties.map((uncertainty, index) => (
          <div
            key={index}
            className="flex items-start gap-2 p-2 rounded-lg bg-orange-50 border border-orange-100"
          >
            <AlertTriangle className="h-3.5 w-3.5 text-orange-500 mt-0.5 shrink-0" />
            <p className="text-xs text-orange-800">{uncertainty}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

// ────────────────────────────────────────────────────────────────────────────────
// Main MessageCard Component
// ────────────────────────────────────────────────────────────────────────────────

const MessageCard = ({
  role,
  content,
  timestamp,
  agentName,
  confidence,
  confidenceBreakdown,
  toolMetadata,
  reasoning,
}: MessageCardProps) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const isAssistant = role !== "user";
  const isSystem = role === "system";

  const hasReasoning =
    reasoning &&
    ((reasoning.reasoning_steps && reasoning.reasoning_steps.length > 0) ||
      (reasoning.key_findings && reasoning.key_findings.length > 0) ||
      (reasoning.tools_considered && reasoning.tools_considered.length > 0) ||
      (reasoning.uncertainties && reasoning.uncertainties.length > 0) ||
      reasoning.suggested_followup);

  const breakdownEntries = confidenceBreakdown
    ? Object.entries(confidenceBreakdown).sort((a, b) => {
        const left = COMPONENT_ORDER.indexOf(a[0]);
        const right = COMPONENT_ORDER.indexOf(b[0]);
        const fallback = COMPONENT_ORDER.length;
        return (left === -1 ? fallback : left) - (right === -1 ? fallback : right);
      })
    : undefined;

  const confidenceLevel = confidence
    ? confidence >= 0.8
      ? "high"
      : confidence >= 0.5
        ? "medium"
        : "low"
    : null;

  return (
    <div
      className={cn(
        "group flex gap-4 p-5 rounded-2xl transition-all duration-300 animate-fade-in",
        "hover:shadow-elevated",
        isSystem
          ? "bg-gradient-to-br from-slate-100 to-slate-50 border border-slate-200/60"
          : isAssistant
            ? "bg-gradient-to-br from-white to-slate-50/50 border border-slate-200/40 shadow-soft"
            : "bg-gradient-to-br from-primary-50 to-primary-100/60 border border-primary-200/40"
      )}
    >
      <div
        className={cn(
          "w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 transition-transform duration-300 group-hover:scale-105",
          isSystem
            ? "bg-gradient-to-br from-slate-200 to-slate-300"
            : isAssistant
              ? "bg-gradient-to-br from-primary-100 to-primary-200 shadow-sm"
              : "bg-gradient-to-br from-primary-500 to-primary-600 shadow-sm"
        )}
      >
        {isSystem ? (
          <Sparkles className="h-5 w-5 text-slate-600" />
        ) : isAssistant ? (
          <Brain className="h-5 w-5 text-primary-600" />
        ) : (
          <User className="h-5 w-5 text-white" />
        )}
      </div>

      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-sm font-semibold text-foreground">
            {isSystem ? "NeuraForge" : isAssistant ? agentName ?? "NeuraForge Agent" : "You"}
          </span>
          {typeof confidence === "number" && (
            <Badge
              variant={
                confidenceLevel === "high"
                  ? "success"
                  : confidenceLevel === "medium"
                    ? "warning"
                    : "secondary"
              }
              className="text-[10px] px-2 py-0.5"
            >
              {(confidence * 100).toFixed(0)}%
            </Badge>
          )}
          {hasReasoning && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className={cn(
                "flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full transition-colors",
                isExpanded
                  ? "bg-primary-100 text-primary-700"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              )}
            >
              {isExpanded ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
              {isExpanded ? "Hide" : "Show"} Reasoning
            </button>
          )}
          <span className="text-xs text-muted-foreground ml-auto">{timestamp}</span>
        </div>

        {/* Markdown-rendered content */}
        <div className="prose prose-slate prose-sm max-w-none text-foreground/90">
          <ReactMarkdown
            components={{
              // Custom rendering for cleaner output
              h1: ({ children }) => (
                <h1 className="text-xl font-bold text-foreground mt-4 mb-3 first:mt-0">{children}</h1>
              ),
              h2: ({ children }) => (
                <h2 className="text-lg font-semibold text-foreground mt-4 mb-2 first:mt-0">{children}</h2>
              ),
              h3: ({ children }) => (
                <h3 className="text-base font-semibold text-foreground mt-3 mb-2 first:mt-0">{children}</h3>
              ),
              p: ({ children }) => (
                <p className="text-[15px] leading-relaxed text-foreground/85 mb-3 last:mb-0">{children}</p>
              ),
              ul: ({ children }) => (
                <ul className="list-disc list-outside ml-5 space-y-1.5 text-[15px] text-foreground/85 mb-3">{children}</ul>
              ),
              ol: ({ children }) => (
                <ol className="list-decimal list-outside ml-5 space-y-1.5 text-[15px] text-foreground/85 mb-3">{children}</ol>
              ),
              li: ({ children }) => (
                <li className="leading-relaxed">{children}</li>
              ),
              strong: ({ children }) => (
                <strong className="font-semibold text-foreground">{children}</strong>
              ),
              em: ({ children }) => (
                <em className="italic text-foreground/80">{children}</em>
              ),
              code: ({ children }) => (
                <code className="bg-slate-100 text-primary-700 px-1.5 py-0.5 rounded text-sm font-mono">{children}</code>
              ),
              pre: ({ children }) => (
                <pre className="bg-slate-900 text-slate-100 p-4 rounded-lg overflow-x-auto text-sm font-mono mb-3">{children}</pre>
              ),
              blockquote: ({ children }) => (
                <blockquote className="border-l-4 border-primary-300 pl-4 py-1 my-3 bg-primary-50/50 rounded-r-lg italic text-foreground/80">{children}</blockquote>
              ),
              hr: () => <hr className="my-4 border-slate-200" />,
            }}
          >
            {content}
          </ReactMarkdown>
        </div>

        {/* Tool Metadata - only show tool name and latency, hide debug info */}
        {toolMetadata && (
          <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
            {toolMetadata.name && (
              <span className="inline-flex items-center gap-1 bg-slate-100 px-2 py-0.5 rounded-md">
                <span className="font-medium">Tool:</span> {toolMetadata.name}
              </span>
            )}
            {toolMetadata.resolved && toolMetadata.resolved !== toolMetadata.name && (
              <span className="inline-flex items-center gap-1 bg-slate-100 px-2 py-0.5 rounded-md">
                <span className="font-medium">Target:</span> {toolMetadata.resolved}
              </span>
            )}
            {typeof toolMetadata.latency === "number" && (
              <span className="inline-flex items-center gap-1 bg-slate-100 px-2 py-0.5 rounded-md">
                <span className="font-medium">Latency:</span> {(toolMetadata.latency * 1000).toFixed(0)}ms
              </span>
            )}
            {typeof toolMetadata.cached === "boolean" && (
              <Badge variant={toolMetadata.cached ? "secondary" : "outline"} className="text-[10px]">
                {toolMetadata.cached ? "Cached" : "Live"}
              </Badge>
            )}
          </div>
        )}

        {/* Expandable Reasoning Section */}
        {hasReasoning && isExpanded && (
          <div className="mt-4 pt-4 border-t border-slate-200/60 space-y-4">
            {/* Confidence Breakdown - only visible in expanded view */}
            {breakdownEntries && (
              <div className="space-y-2">
                <h4 className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
                  Confidence Breakdown
                </h4>
                <div className="flex flex-wrap gap-2">
                  {breakdownEntries.map(([key, value]) => (
                    <span
                      key={key}
                      className="text-[10px] uppercase tracking-wider font-medium bg-white/80 border border-slate-200 px-2.5 py-1 rounded-lg text-slate-600 shadow-xs"
                    >
                      {`${key.replace(/_/g, " ")}: ${(value * 100).toFixed(0)}%`}
                    </span>
                  ))}
                </div>
              </div>
            )}
            
            {reasoning.rationale && (
              <div className="text-xs text-slate-600 italic bg-slate-50 p-2.5 rounded-lg border border-slate-100">
                <span className="font-medium not-italic">Rationale:</span> {reasoning.rationale}
              </div>
            )}

            {reasoning.reasoning_steps && reasoning.reasoning_steps.length > 0 && (
              <ReasoningSteps steps={reasoning.reasoning_steps} />
            )}

            {reasoning.key_findings && reasoning.key_findings.length > 0 && (
              <KeyFindings findings={reasoning.key_findings} />
            )}

            {reasoning.tools_considered && reasoning.tools_considered.length > 0 && (
              <ToolsConsidered tools={reasoning.tools_considered} />
            )}

            {reasoning.uncertainties && reasoning.uncertainties.length > 0 && (
              <Uncertainties uncertainties={reasoning.uncertainties} />
            )}

            {reasoning.suggested_followup && (
              <div className="flex items-start gap-2 p-2.5 rounded-lg bg-primary-50 border border-primary-100">
                <ArrowRight className="h-3.5 w-3.5 text-primary-600 mt-0.5 shrink-0" />
                <div>
                  <span className="text-[10px] font-semibold text-primary-700 uppercase tracking-wider">
                    Suggested Follow-up
                  </span>
                  <p className="text-xs text-primary-800 mt-0.5">{reasoning.suggested_followup}</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageCard;
