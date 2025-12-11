import { useEffect, useRef } from "react";
import { Sparkles, MessageCircle, Loader2 } from "lucide-react";

import { useTaskContext } from "@/contexts/TaskContext";
import { emitWorkspaceState } from "@/lib/analytics";
import MessageCard from "./MessageCard";

const ChatWorkspace = () => {
  const { messages, isStreaming } = useTaskContext();
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: isStreaming ? "smooth" : "auto" });
    }
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
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-white">
      {/* Minimal Header */}
      <div className="flex-shrink-0 border-b border-slate-200/80 px-6 py-3 bg-white">
        <div className="max-w-3xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center">
              <MessageCircle className="h-4 w-4 text-white" />
            </div>
            <span className="text-sm font-medium text-foreground">NeuraForge</span>
          </div>
          {isStreaming && (
            <div className="flex items-center gap-2 text-xs text-primary-600">
              <Loader2 className="h-3.5 w-3.5 animate-spin" />
              <span className="font-medium">Processing...</span>
            </div>
          )}
        </div>
      </div>

      {/* Messages Container - Fixed height with scroll */}
      <div 
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto overscroll-contain"
        style={{ scrollBehavior: 'smooth' }}
      >
        <div className="max-w-3xl mx-auto px-4 py-6">
          {hasMessages ? (
            <div className="space-y-6">
              {messages.map((message) => (
                <MessageCard
                  key={message.id}
                  role={message.role === "system" ? "system" : message.role}
                  content={message.content}
                  timestamp={message.timestamp}
                  agentName={message.agentName}
                  confidence={message.confidence}
                  confidenceBreakdown={message.confidenceBreakdown}
                  toolMetadata={message.toolMetadata}
                  reasoning={message.reasoning}
                />
              ))}
              
              {/* Streaming indicator at bottom */}
              {isStreaming && (
                <div className="flex items-center justify-center gap-2 py-3">
                  <div className="flex items-center gap-2 text-sm text-slate-500 bg-slate-50 px-4 py-2.5 rounded-full border border-slate-200">
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-2 h-2 bg-primary-500 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                    <span>Agents are thinking...</span>
                  </div>
                </div>
              )}
              
              {/* Scroll anchor */}
              <div ref={bottomRef} className="h-1" />
            </div>
          ) : (
            /* Welcome Screen */
            <div className="flex flex-col items-center justify-center py-20 px-6">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center mb-6 shadow-lg">
                <Sparkles className="h-10 w-10 text-white" />
              </div>
              <h2 className="text-2xl font-bold text-foreground mb-3">How can I help you today?</h2>
              <p className="text-base text-muted-foreground text-center max-w-md leading-relaxed">
                Ask me anything — financial analysis, research, creative writing, or enterprise insights.
                I'll coordinate specialized agents to give you comprehensive answers.
              </p>
              
              {/* Example prompts */}
              <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-lg">
                {[
                  "Give me a financial analysis of Tesla",
                  "Research the latest AI trends",
                  "Help me write a business proposal",
                  "Analyze market opportunities in tech",
                ].map((prompt, idx) => (
                  <button
                    key={idx}
                    className="text-left px-4 py-3 rounded-xl border border-slate-200 hover:border-primary-300 hover:bg-primary-50/50 transition-all text-sm text-slate-700 hover:text-foreground"
                    onClick={() => {
                      // This could be wired to send the prompt
                    }}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatWorkspace;
