import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Search, Palette, TrendingUp, Building, Clock } from "lucide-react";
import { cn } from "@/lib/utils";

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

interface ConversationItem {
  id: string;
  title: string;
  timestamp: string;
  agents: ('research' | 'creative' | 'finance' | 'enterprise')[];
  preview: string;
}

const mockConversations: ConversationItem[] = [
  {
    id: '1',
    title: 'Market Analysis Q4 2024',
    timestamp: '2 hours ago',
    agents: ['research', 'finance'],
    preview: 'Analyzed Q4 market trends and financial projections...'
  },
  {
    id: '2', 
    title: 'Creative Campaign Ideas',
    timestamp: '1 day ago',
    agents: ['creative', 'research'],
    preview: 'Generated marketing campaign concepts for product launch...'
  },
  {
    id: '3',
    title: 'Enterprise Integration Plan',
    timestamp: '3 days ago',
    agents: ['enterprise', 'research'],
    preview: 'Developed comprehensive integration strategy...'
  },
  {
    id: '4',
    title: 'Investment Portfolio Review',
    timestamp: '1 week ago',
    agents: ['finance', 'research'],
    preview: 'Reviewed current portfolio performance and recommendations...'
  },
];

interface ConversationHistoryProps {
  selectedId?: string;
  onSelect?: (id: string) => void;
}

export const ConversationHistory = ({ selectedId, onSelect }: ConversationHistoryProps) => {
  return (
    <div className="w-80 border-r border-border bg-card/30 backdrop-blur-sm p-6">
      <div className="space-y-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Clock className="w-5 h-5 text-neura-blue" />
          Conversations
        </h3>
        
        <ScrollArea className="h-[calc(100vh-200px)]">
          <div className="space-y-3">
            {mockConversations.map((conversation) => (
              <Card 
                key={conversation.id}
                className={cn(
                  "p-4 cursor-pointer transition-all hover:bg-accent/50",
                  selectedId === conversation.id && "bg-accent border-neura-blue/50"
                )}
                onClick={() => onSelect?.(conversation.id)}
              >
                <div className="space-y-2">
                  <h4 className="font-medium text-sm truncate">{conversation.title}</h4>
                  <p className="text-xs text-muted-foreground">{conversation.timestamp}</p>
                  <p className="text-xs text-muted-foreground line-clamp-2">{conversation.preview}</p>
                  
                  <div className="flex gap-1 flex-wrap">
                    {conversation.agents.map((agentType, index) => {
                      const Icon = agentIcons[agentType];
                      const colorClass = agentColors[agentType];
                      
                      return (
                        <div 
                          key={index}
                          className={cn(
                            "w-6 h-6 rounded-full flex items-center justify-center border",
                            agentType === 'research' && "bg-agent-research/10 border-agent-research/20",
                            agentType === 'creative' && "bg-agent-creative/10 border-agent-creative/20",
                            agentType === 'finance' && "bg-agent-finance/10 border-agent-finance/20",
                            agentType === 'enterprise' && "bg-agent-enterprise/10 border-agent-enterprise/20"
                          )}
                        >
                          <Icon className={cn(
                            "w-3 h-3",
                            agentType === 'research' && "text-agent-research",
                            agentType === 'creative' && "text-agent-creative",
                            agentType === 'finance' && "text-agent-finance",
                            agentType === 'enterprise' && "text-agent-enterprise"
                          )} />
                        </div>
                      );
                    })}
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
};