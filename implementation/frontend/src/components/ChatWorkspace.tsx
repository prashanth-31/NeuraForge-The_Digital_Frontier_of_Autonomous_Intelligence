import { useEffect, useRef } from "react";

import { useTaskContext } from "@/contexts/TaskContext";
import { emitWorkspaceState } from "@/lib/analytics";
import { ScrollArea } from "./ui/scroll-area";
import MessageCard from "./MessageCard";

const ChatWorkspace = () => {
  const { messages, isStreaming } = useTaskContext();
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: isStreaming ? "smooth" : "auto" });
  }, [messages, isStreaming]);

  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    emitWorkspaceState({
      messageCount: messages.length,
      isStreaming,
      timestamp: new Date().toISOString(),
      lastMessage: lastMessage
        ? {
            id: lastMessage.id,
            role: lastMessage.role,
            agentName: lastMessage.agentName,
            confidence: lastMessage.confidence,
          }
        : undefined,
    });
  }, [messages, isStreaming]);

  const hasMessages = messages.length > 0;

  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      <div className="border-b border-border p-4 bg-card shadow-soft">
        <h2 className="text-sm font-semibold text-foreground">Current Session</h2>
        <p className="text-xs text-muted-foreground mt-1">
          {isStreaming ? "Streaming live agent responses" : "Collaborative Intelligence Mode"}
        </p>
      </div>

      <ScrollArea className="flex-1 p-6 overflow-hidden">
        <div className="max-w-4xl mx-auto space-y-4 pb-20">
          {hasMessages ? (
            messages.map((message) => (
              <MessageCard
                key={message.id}
                role={message.role === "system" ? "system" : message.role}
                content={message.content}
                timestamp={message.timestamp}
                agentName={message.agentName}
                confidence={message.confidence}
                confidenceBreakdown={message.confidenceBreakdown}
                toolMetadata={message.toolMetadata}
              />
            ))
          ) : (
            <div className="p-10 rounded-xl border border-dashed border-border text-center text-sm text-muted-foreground bg-muted/30">
              <p>Your collaborative session is empty. Start by describing a task below to see real-time agent updates.</p>
            </div>
          )}
          {isStreaming && (
            <div className="text-xs text-muted-foreground italic text-center">
              Agents are preparing additional insights...
            </div>
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>
    </div>
  );
};

export default ChatWorkspace;
