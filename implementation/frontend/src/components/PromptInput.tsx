import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Send, Upload, Database, Paperclip, Sparkles, Bot, Settings, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { useIsMobile } from "@/hooks/use-mobile";
import { Card } from "./ui/card";
import { AnimatePresence, motion } from "framer-motion";
import { 
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

interface PromptInputProps {
  onSend?: (message: string, useMemory: boolean) => void;
  className?: string;
}

const promptSuggestions = [
  "Analyze market trends for renewable energy investments",
  "Summarize this research paper on AI ethics",
  "Compare investment opportunities in emerging markets",
  "Generate creative campaign ideas for sustainable products",
  "Create financial projections for Q3 2025"
];

export const PromptInput = ({ onSend, className }: PromptInputProps) => {
  const [message, setMessage] = useState("");
  const [useMemory, setUseMemory] = useState(true);
  const [useEnhance, setUseEnhance] = useState(true);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isMobile = useIsMobile();

  // Auto-adjust textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  const handleSend = () => {
    if (message.trim()) {
      onSend?.(message, useMemory);
      setMessage("");
      setShowSuggestions(false);
      
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={cn("border-t border-border bg-card/50 backdrop-blur-sm p-4 md:p-6", className)}>
      <div className="max-w-4xl mx-auto space-y-4">
        {/* Suggestions */}
        <AnimatePresence>
          {showSuggestions && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 10 }}
              className="flex gap-2 flex-wrap"
            >
              {promptSuggestions.map((suggestion, index) => (
                <Button
                  key={index}
                  variant="outline"
                  size="sm"
                  className="text-xs h-8 transition-all hover:bg-accent/80 hover:text-accent-foreground hover:border-neura-blue/40"
                  onClick={() => {
                    setMessage(suggestion);
                    setShowSuggestions(false);
                    if (textareaRef.current) {
                      textareaRef.current.focus();
                    }
                  }}
                >
                  {suggestion}
                </Button>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
        
        {/* Input Area */}
        <div className="relative">
          <Card className="overflow-hidden border-border/50 bg-background/50 backdrop-blur-sm">
            <div className="flex flex-col">
              <div className="flex-1 relative">
                <Textarea
                  ref={textareaRef}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  onKeyDown={handleKeyPress}
                  onFocus={() => setShowSuggestions(true)}
                  placeholder="Ask something, upload a file, or analyze data..."
                  className="min-h-[60px] max-h-[200px] resize-none rounded-none border-0 focus-visible:ring-0 focus-visible:ring-offset-0 bg-transparent"
                  rows={1}
                />
              </div>
              
              <div className="flex items-center justify-between px-3 py-2 border-t border-border/30">
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 rounded-full"
                    onClick={() => {}}
                  >
                    <Paperclip className="w-4 h-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 rounded-full"
                    onClick={() => {}}
                  >
                    <Upload className="w-4 h-4" />
                  </Button>
                  <Button
                    variant={useEnhance ? "default" : "ghost"}
                    size="sm"
                    className={cn(
                      "h-8 gap-1.5 text-xs rounded-full",
                      useEnhance && "bg-neura-blue text-white hover:bg-neura-blue/90"
                    )}
                    onClick={() => setUseEnhance(!useEnhance)}
                  >
                    <Sparkles className="w-3.5 h-3.5" />
                    {!isMobile && "Enhance"}
                  </Button>
                  <Button
                    variant={useMemory ? "default" : "ghost"}
                    size="sm"
                    className={cn(
                      "h-8 gap-1.5 text-xs rounded-full",
                      useMemory && "bg-neura-blue text-white hover:bg-neura-blue/90"
                    )}
                    onClick={() => setUseMemory(!useMemory)}
                  >
                    <Database className="w-3.5 h-3.5" />
                    {!isMobile && "Memory"}
                  </Button>
                  
                  <Popover open={showAdvancedSettings} onOpenChange={setShowAdvancedSettings}>
                    <PopoverTrigger asChild>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 gap-1.5 text-xs rounded-full"
                      >
                        <Settings className="w-3.5 h-3.5" />
                        {!isMobile && "Settings"}
                      </Button>
                    </PopoverTrigger>
                    <PopoverContent className="w-[280px] p-4">
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <h4 className="font-medium text-sm">Advanced Settings</h4>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6"
                            onClick={() => setShowAdvancedSettings(false)}
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                        
                        <div className="space-y-2">
                          <Badge variant="outline" className="text-xs w-full justify-between p-2 cursor-pointer hover:bg-accent/50">
                            <span>Auto-select agents</span>
                            <span className="font-semibold text-neura-blue">On</span>
                          </Badge>
                          <Badge variant="outline" className="text-xs w-full justify-between p-2 cursor-pointer hover:bg-accent/50">
                            <span>Multi-agent negotiation</span>
                            <span className="font-semibold text-neura-blue">On</span>
                          </Badge>
                          <Badge variant="outline" className="text-xs w-full justify-between p-2 cursor-pointer hover:bg-accent/50">
                            <span>Step-by-step reasoning</span>
                            <span className="font-semibold text-muted-foreground">Off</span>
                          </Badge>
                          <Badge variant="outline" className="text-xs w-full justify-between p-2 cursor-pointer hover:bg-accent/50">
                            <span>Web search</span>
                            <span className="font-semibold text-neura-blue">On</span>
                          </Badge>
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>
                </div>
                
                <Button
                  onClick={handleSend}
                  disabled={!message.trim()}
                  className="h-9 px-4 rounded-full bg-neura-blue hover:bg-neura-blue/90 text-white"
                >
                  <Send className="w-4 h-4 mr-2" />
                  Send
                </Button>
              </div>
            </div>
          </Card>
          
          {/* Controls */}
          <div className="flex justify-center mt-3">
            <div className="text-xs text-muted-foreground">
              Press Enter to send, Shift+Enter for new line
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};