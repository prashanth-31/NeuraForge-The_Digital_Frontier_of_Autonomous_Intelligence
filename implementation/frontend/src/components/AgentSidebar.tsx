import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Brain, Search, Palette, TrendingUp, Building, Database, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Agent } from "./AgentBubble";

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

interface AgentSidebarProps {
  activeAgents: Agent[];
}

export const AgentSidebar = ({ activeAgents }: AgentSidebarProps) => {
  return (
    <div className="w-80 border-l border-border bg-card/30 backdrop-blur-sm p-6 space-y-6">
      <div className="space-y-3">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Brain className="w-5 h-5 text-neura-blue" />
          Active Agents
        </h3>
        
        <div className="space-y-3">
          {activeAgents.map((agent) => {
            const Icon = agentIcons[agent.type];
            const colorClass = agentColors[agent.type];
            
            return (
              <Card key={agent.id} className="p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <div className={cn(
                      "w-8 h-8 rounded-lg flex items-center justify-center border",
                      agent.type === 'research' && "bg-agent-research/10 border-agent-research/20",
                      agent.type === 'creative' && "bg-agent-creative/10 border-agent-creative/20",
                      agent.type === 'finance' && "bg-agent-finance/10 border-agent-finance/20",
                      agent.type === 'enterprise' && "bg-agent-enterprise/10 border-agent-enterprise/20"
                    )}>
                      <Icon className={cn(
                        "w-4 h-4",
                        agent.type === 'research' && "text-agent-research",
                        agent.type === 'creative' && "text-agent-creative",
                        agent.type === 'finance' && "text-agent-finance",
                        agent.type === 'enterprise' && "text-agent-enterprise"
                      )} />
                    </div>
                    <div>
                      <p className="font-medium text-sm">{agent.name}</p>
                      <p className="text-xs text-muted-foreground capitalize">{agent.status}</p>
                    </div>
                  </div>
                  <Badge 
                    variant="secondary" 
                    className={cn(
                      "text-xs border",
                      agent.type === 'research' && "bg-agent-research/10 text-agent-research border-agent-research/20",
                      agent.type === 'creative' && "bg-agent-creative/10 text-agent-creative border-agent-creative/20",
                      agent.type === 'finance' && "bg-agent-finance/10 text-agent-finance border-agent-finance/20",
                      agent.type === 'enterprise' && "bg-agent-enterprise/10 text-agent-enterprise border-agent-enterprise/20"
                    )}
                  >
                    {agent.confidence}%
                  </Badge>
                </div>
                
                <Progress value={agent.confidence} className="h-2" />
                
                {agent.metadata && (
                  <div className="flex flex-wrap gap-1">
                    {agent.metadata.map((tag, index) => (
                      <Badge key={index} variant="outline" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      </div>
      
      <Card className="p-4 space-y-3">
        <h4 className="font-medium flex items-center gap-2">
          <Database className="w-4 h-4" />
          Memory Usage
        </h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Redis Cache</span>
            <span>24.3 MB</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Qdrant Vector</span>
            <span>156 vectors</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">PostgreSQL</span>
            <span>89 records</span>
          </div>
        </div>
      </Card>
      
      <Card className="p-4 space-y-3">
        <h4 className="font-medium flex items-center gap-2">
          <Cpu className="w-4 h-4" />
          System Status
        </h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">LLaMA 3.1</span>
            <Badge variant="secondary" className="bg-green-500/10 text-green-400 border-green-500/20">
              Online
            </Badge>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Ollama</span>
            <Badge variant="secondary" className="bg-green-500/10 text-green-400 border-green-500/20">
              Ready
            </Badge>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Response Time</span>
            <span>1.2s avg</span>
          </div>
        </div>
      </Card>
    </div>
  );
};