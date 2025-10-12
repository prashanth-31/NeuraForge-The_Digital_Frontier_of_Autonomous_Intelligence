import { Send, Paperclip, Settings2 } from "lucide-react";
import { useState } from "react";

import { useTaskContext } from "@/contexts/TaskContext";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";

const InputBar = () => {
  const [message, setMessage] = useState("");
  const { submitTask, isStreaming } = useTaskContext();

  const handleSubmit = async () => {
    if (!message.trim()) {
      return;
    }
    await submitTask(message);
    setMessage("");
  };

  return (
    <div className="border-t border-border bg-card shadow-soft sticky bottom-0">
      <div className="p-4">
        <div className="max-w-4xl mx-auto">
          <div className="relative flex items-end gap-3">
            <div className="flex-1 relative">
              <Textarea
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                placeholder="Ask NeuraForge anything..."
                className="min-h-[52px] max-h-32 resize-none pr-24 bg-muted/30 border-border/50 focus:bg-background transition-smooth"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void handleSubmit();
                  }
                }}
                disabled={isStreaming}
              />
              <div className="absolute right-2 bottom-2 flex items-center gap-1">
                <Button type="button" variant="ghost" size="icon" className="h-8 w-8" disabled>
                  <Paperclip className="h-4 w-4" />
                </Button>
                <Button type="button" variant="ghost" size="icon" className="h-8 w-8" disabled>
                  <Settings2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <Button
              size="icon"
              className="h-[52px] w-[52px] bg-primary hover:bg-primary/90 transition-smooth hover:scale-105"
              onClick={() => void handleSubmit()}
              disabled={isStreaming}
            >
              <Send className="h-5 w-5" />
            </Button>
          </div>
          <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
            <span>Shift + Enter for new line • Enter to send</span>
            <span>{isStreaming ? "Agents responding..." : "Collaborative Mode"}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InputBar;
