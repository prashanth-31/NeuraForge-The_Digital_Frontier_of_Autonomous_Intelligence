import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Brain, Search, Palette, TrendingUp, Building, User } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

export interface Agent {
  id: string;
  name: string;
  type: 'research' | 'creative' | 'finance' | 'enterprise';
  confidence: number;
  status: 'active' | 'thinking' | 'complete' | 'idle';
  response?: string;
  metadata?: string[];
}

const agentIcons = {
  research: Search,
  creative: Palette,
  finance: TrendingUp,
  enterprise: Building,
};

const agentColors = {
  research: 'agent-research',
  creative: 'agent-creative', 
  finance: 'agent-finance',
  enterprise: 'agent-enterprise',
};

const agentDescriptions = {
  research: 'Specialized in factual information and data analysis',
  creative: 'Specialized in creative content and innovative thinking',
  finance: 'Specialized in financial analysis and planning',
  enterprise: 'Specialized in business strategy and operations',
};

interface AgentBubbleProps {
  agent: Agent;
  isUserMessage?: boolean;
}

export const AgentBubble = ({ agent, isUserMessage = false }: AgentBubbleProps) => {
  const Icon = isUserMessage ? User : agentIcons[agent.type] || Brain;
  const colorClass = !isUserMessage ? agentColors[agent.type] : '';
  
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
      <div className={cn(
        "w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 border",
        agent.type === 'research' && "bg-agent-research/10 border-agent-research/20",
        agent.type === 'creative' && "bg-agent-creative/10 border-agent-creative/20",
        agent.type === 'finance' && "bg-agent-finance/10 border-agent-finance/20",
        agent.type === 'enterprise' && "bg-agent-enterprise/10 border-agent-enterprise/20"
      )}>
        <Icon className={cn(
          "w-5 h-5",
          agent.type === 'research' && "text-agent-research",
          agent.type === 'creative' && "text-agent-creative",
          agent.type === 'finance' && "text-agent-finance",
          agent.type === 'enterprise' && "text-agent-enterprise"
        )} />
      </div>
      
      <div className="flex-1 space-y-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-medium text-foreground">{agent.name}</span>
          <Badge 
            variant="secondary" 
            className={cn(
              "text-xs px-2 py-0.5 border",
              agent.type === 'research' && "bg-agent-research/10 text-agent-research border-agent-research/20",
              agent.type === 'creative' && "bg-agent-creative/10 text-agent-creative border-agent-creative/20",
              agent.type === 'finance' && "bg-agent-finance/10 text-agent-finance border-agent-finance/20",
              agent.type === 'enterprise' && "bg-agent-enterprise/10 text-agent-enterprise border-agent-enterprise/20"
            )}
          >
            {agent.confidence}% confidence
          </Badge>
          <Badge variant="outline" className="text-xs hidden sm:inline-flex">
            {agentDescriptions[agent.type] || 'NeuraForge Assistant'}
          </Badge>
          {agent.metadata?.map((tag, index) => (
            <Badge key={index} variant="outline" className="text-xs">
              {tag}
            </Badge>
          ))}
        </div>
        
        <Card className="p-4 bg-card border border-border/50 shadow-sm hover:shadow-md transition-shadow duration-200">
          <p className="text-card-foreground whitespace-pre-line">{agent.response}</p>
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