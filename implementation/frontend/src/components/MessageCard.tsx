import { Brain, User } from "lucide-react";
import { cn } from "@/lib/utils";

interface MessageCardProps {
  role: "user" | "assistant";
  content: string;
  timestamp: string;
}

const MessageCard = ({ role, content, timestamp }: MessageCardProps) => {
  const isAssistant = role === "assistant";

  return (
    <div className={cn("flex gap-4 p-6 rounded-xl transition-smooth animate-fade-in", 
      isAssistant ? "bg-[hsl(var(--message-ai))]" : "bg-[hsl(var(--message-user))]"
    )}>
      <div className={cn(
        "w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0",
        isAssistant ? "bg-primary/10" : "bg-secondary/10"
      )}>
        {isAssistant ? (
          <Brain className="h-5 w-5 text-primary" />
        ) : (
          <User className="h-5 w-5 text-secondary" />
        )}
      </div>
      
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-sm font-medium text-foreground">
            {isAssistant ? "NeuraForge Intelligence System" : "You"}
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
