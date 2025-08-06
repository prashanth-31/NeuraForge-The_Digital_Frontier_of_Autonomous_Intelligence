import { useState, useRef, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { AgentBubble, type Agent } from "./AgentBubble";
import { Button } from "./ui/button";
import { Copy, CheckCircle, RotateCcw, ThumbsUp, ThumbsDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { useToast } from "./ui/use-toast";
import { useChatContext } from "@/contexts/ChatContext"; // Use the ChatContext instead
import { WelcomeMessage } from "./WelcomeMessage";

interface ChatInterfaceProps {
  className?: string;
  onSendExample?: (example: string) => void;
}

export const ChatInterface = ({ className, onSendExample }: ChatInterfaceProps) => {
  const { messages, sendMessage, connectionStatus } = useChatContext();
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [feedbackGiven, setFeedbackGiven] = useState<Record<string, 'up' | 'down' | null>>({});
  const [isAutoScrollEnabled, setIsAutoScrollEnabled] = useState(true);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();
  
  console.log("ChatInterface rendering with messages:", messages); // Debug log

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollAreaRef.current && isAutoScrollEnabled) {
      const scrollElement = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollElement) {
        // Use a small timeout to ensure DOM has updated
        setTimeout(() => {
          scrollElement.scrollTop = scrollElement.scrollHeight;
        }, 10);
      }
    }
  }, [messages, isAutoScrollEnabled]);

  const copyToClipboard = (id: string, text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
      toast({
        title: "Copied to clipboard",
        description: "The response has been copied to your clipboard.",
        duration: 3000,
      });
    });
  };

  const giveFeedback = (id: string, type: 'up' | 'down') => {
    setFeedbackGiven(prev => ({
      ...prev,
      [id]: type
    }));
    toast({
      title: type === 'up' ? "Positive feedback sent" : "Negative feedback sent",
      description: "Thank you for your feedback!",
      duration: 3000,
    });
  };

  const regenerateResponse = (id: string) => {
    // In a real implementation, this would trigger a regeneration
    toast({
      title: "Regenerating response",
      description: "The agent is generating a new response...",
      duration: 3000,
    });
  };

  return (
    <div className={cn("h-full overflow-hidden", className)} ref={scrollAreaRef}>
      <ScrollArea className="h-full p-6">
        {messages.length === 0 ? (
          <WelcomeMessage onExampleClick={(example) => {
            console.log("Example clicked:", example);
            if (onSendExample) {
              onSendExample(example);
            } else {
              // If no handler is provided, use the context directly
              sendMessage(example);
            }
          }} />
        ) : (
          <div className="max-w-4xl mx-auto space-y-1">
            {messages.map((message, index) => (
              <div key={message.id} className="group relative">
                <AgentBubble 
                  agent={message} 
                  isUserMessage={message.name === 'You'}
                />
                
                {!message.response?.endsWith('...') && !message.status?.includes('thinking') && message.name !== 'You' && (
                  <div className="absolute right-0 top-0 opacity-0 group-hover:opacity-100 transition-opacity flex gap-1">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 rounded-full"
                      onClick={() => copyToClipboard(message.id, message.response || '')}
                    >
                      {copiedId === message.id ? 
                        <CheckCircle className="h-4 w-4 text-green-500" /> : 
                        <Copy className="h-4 w-4" />
                      }
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className={cn(
                        "h-8 w-8 rounded-full",
                        feedbackGiven[message.id] === 'up' && "bg-green-500/10 text-green-500"
                      )}
                      onClick={() => giveFeedback(message.id, 'up')}
                    >
                      <ThumbsUp className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className={cn(
                        "h-8 w-8 rounded-full",
                        feedbackGiven[message.id] === 'down' && "bg-red-500/10 text-red-500"
                      )}
                      onClick={() => giveFeedback(message.id, 'down')}
                    >
                      <ThumbsDown className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 rounded-full"
                      onClick={() => regenerateResponse(message.id)}
                    >
                      <RotateCcw className="h-4 w-4" />
                    </Button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
};