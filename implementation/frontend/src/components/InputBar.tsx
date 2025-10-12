import { Send, Paperclip, Settings2 } from "lucide-react";
import { Button } from "./ui/button";
import { Textarea } from "./ui/textarea";
import { useState } from "react";

const InputBar = () => {
  const [message, setMessage] = useState("");

  return (
    <div className="fixed bottom-0 left-56 right-72 bg-card border-t border-border shadow-soft z-40">
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
                    // Handle send
                  }
                }}
              />
              <div className="absolute right-2 bottom-2 flex items-center gap-1">
                <Button variant="ghost" size="icon" className="h-8 w-8">
                  <Paperclip className="h-4 w-4" />
                </Button>
                <Button variant="ghost" size="icon" className="h-8 w-8">
                  <Settings2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <Button 
              size="icon"
              className="h-[52px] w-[52px] bg-primary hover:bg-primary/90 transition-smooth hover:scale-105"
            >
              <Send className="h-5 w-5" />
            </Button>
          </div>
          <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
            <span>Shift + Enter for new line • Enter to send</span>
            <span>Collaborative Mode</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InputBar;
