import { Card } from "@/components/ui/card";
import { Brain, User } from "lucide-react";
import { motion } from "framer-motion";

export interface Agent {
  id: string;
  name: string;
  type: 'research' | 'creative' | 'finance' | 'enterprise';
  confidence: number;
  status: 'active' | 'thinking' | 'complete' | 'idle';
  response?: string;
  metadata?: string[] | Record<string, any>;
}

interface AgentBubbleProps {
  agent: Agent;
  isUserMessage?: boolean;
}

export const AgentBubble = ({ agent, isUserMessage = false }: AgentBubbleProps) => {
  const Icon = isUserMessage ? User : Brain;
  
  const bubbleVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { 
      opacity: 1, 
      y: 0,
      transition: { 
        duration: 0.3,
        ease: "easeOut"
      }
    }
  };
  
  const thinkingVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1 }
  };
  
  if (isUserMessage) {
    return (
      <motion.div 
        className="flex justify-end mb-4"
        initial="hidden"
        animate="visible"
        variants={bubbleVariants}
      >
        <div className="flex items-start gap-3">
          <Card className="max-w-[70%] p-4 bg-primary text-primary-foreground dark:bg-primary/90 shadow-md">
            <p>{agent.response}</p>
          </Card>
          <div className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 bg-primary text-primary-foreground">
            <User className="w-5 h-5" />
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div 
      className="flex gap-3 mb-6"
      initial="hidden"
      animate="visible"
      variants={bubbleVariants}
    >
      <div className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 border bg-muted/30 border-border/50">
        <Icon className="w-5 h-5 text-primary" />
      </div>
      
      <div className="flex-1 space-y-2">
        <Card className="p-4 space-y-3 bg-card border border-border/50 shadow-sm hover:shadow-md transition-shadow duration-200">
          <p className="text-card-foreground whitespace-pre-line">{agent.response}</p>
          {agent.metadata && typeof agent.metadata === 'object' && !Array.isArray(agent.metadata) && (
            <div className="pt-2 border-t border-border/30 text-xs space-y-1">
              {'citations' in agent.metadata && Array.isArray((agent.metadata as any).citations) && (
                <div>
                  <span className="font-semibold">Sources:</span>
                  <ul className="list-disc ml-4 mt-1 space-y-0.5">
                    {(agent.metadata as any).citations.slice(0,4).map((c:string, i:number) => (
                      <li key={i} className="truncate"><a href={c} target="_blank" rel="noopener noreferrer" className="text-primary underline">{c}</a></li>
                    ))}
                  </ul>
                </div>
              )}
              {'used_tools' in agent.metadata && Array.isArray((agent.metadata as any).used_tools) && (
                <div className="text-muted-foreground">Tools: {(agent.metadata as any).used_tools.join(', ')}</div>
              )}
              {'autobrowse' in agent.metadata && (agent.metadata as any).autobrowse && (
                <div className="text-green-600 dark:text-green-400">Auto-browsed ✅</div>
              )}
              {'autobrowse_skipped' in agent.metadata && (agent.metadata as any).autobrowse_skipped && (
                <div className="text-yellow-600 dark:text-yellow-400">Auto-browse skipped (fallback)</div>
              )}
            </div>
          )}
        </Card>
        
        {agent.status === 'thinking' && (
          <motion.div 
            className="flex items-center gap-2 text-sm text-muted-foreground"
            variants={thinkingVariants}
            initial="hidden"
            animate="visible"
            transition={{ duration: 0.5 }}
          >
            <div className="flex gap-1">
              <div className="w-2 h-2 bg-neura-blue rounded-full animate-bounce [animation-delay:-0.3s]"></div>
              <div className="w-2 h-2 bg-neura-blue rounded-full animate-bounce [animation-delay:-0.15s]"></div>
              <div className="w-2 h-2 bg-neura-blue rounded-full animate-bounce"></div>
            </div>
            <span>Processing...</span>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};