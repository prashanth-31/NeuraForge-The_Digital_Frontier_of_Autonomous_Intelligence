import { Brain, User } from "lucide-react";
import { cn } from "@/lib/utils";

interface MessageCardProps {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: string;
  agentName?: string;
  confidence?: number;
}

const MessageCard = ({ role, content, timestamp, agentName, confidence }: MessageCardProps) => {
  const isAssistant = role !== "user";
  const isSystem = role === "system";

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
      </div>
    </div>
  );
};

export default MessageCard;
