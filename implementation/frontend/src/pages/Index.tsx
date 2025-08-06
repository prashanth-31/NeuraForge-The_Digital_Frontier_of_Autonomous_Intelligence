import { useState } from "react";
import { NeuraForgeHeader } from "@/components/NeuraForgeHeader";
import { ConversationHistory } from "@/components/ConversationHistory";
import { ChatInterface } from "@/components/ChatInterface";
import { AgentSidebar } from "@/components/AgentSidebar";
import { PromptInput } from "@/components/PromptInput";
import type { Agent } from "@/components/AgentBubble";
import { useIsMobile } from "@/hooks/use-mobile";
import { Button } from "@/components/ui/button";
import { Menu, Users, X, Wifi, WifiOff } from "lucide-react";
import { cn } from "@/lib/utils";
import { useChat } from "@/hooks/useChat";
import { Badge } from "@/components/ui/badge";

const Index = () => {
  const [selectedConversation, setSelectedConversation] = useState<string>();
  const [showConversations, setShowConversations] = useState(false);
  const [showAgents, setShowAgents] = useState(false);
  const isMobile = useIsMobile();
  const { sendMessage, connectionStatus } = useChat();
  
  // Mock active agents
  const activeAgents: Agent[] = [
    {
      id: 'research-agent',
      name: 'Research Agent',
      type: 'research',
      confidence: 94,
      status: 'complete',
      metadata: ['Qdrant Vector DB', '12 sources', 'High confidence']
    },
    {
      id: 'creative-agent',
      name: 'Creative Agent',
      type: 'creative',
      confidence: 91,
      status: 'complete',
      metadata: ['Creative Database', '5 concepts', 'Collaborative']
    },
    {
      id: 'finance-agent',
      name: 'Finance Agent',
      type: 'finance',
      confidence: 89,
      status: 'thinking',
      metadata: ['Market Data', 'Risk Analysis']
    }
  ];

  const handleSendMessage = (message: string, useMemory: boolean) => {
    console.log('Sending message:', message, 'Use memory:', useMemory);
    // Send message to API
    sendMessage(message);
    
    // Close mobile sidebars when sending a message
    if (isMobile) {
      setShowConversations(false);
      setShowAgents(false);
    }
  };

  return (
    <div className="h-screen bg-background flex flex-col overflow-hidden">
      {/* Grid pattern background */}
      <div 
        className="fixed inset-0 opacity-20"
        style={{
          backgroundImage: 'radial-gradient(circle at 1px 1px, hsl(var(--neura-blue)) 1px, transparent 0)',
          backgroundSize: '20px 20px'
        }}
      />
      
      <div className="relative z-10 h-full flex flex-col overflow-hidden">
        <NeuraForgeHeader>
          {/* Connection status indicator */}
          <div className="flex items-center ml-auto">
            <Badge 
              variant={connectionStatus === 'connected' ? 'default' : 'destructive'}
              className="gap-1 ml-auto"
            >
              {connectionStatus === 'connected' ? (
                <>
                  <Wifi className="w-3 h-3" />
                  <span>Connected</span>
                </>
              ) : connectionStatus === 'connecting' ? (
                <>
                  <span className="animate-pulse">Connecting...</span>
                </>
              ) : (
                <>
                  <WifiOff className="w-3 h-3" />
                  <span>{connectionStatus === 'error' ? 'Connection Error' : 'Disconnected'}</span>
                </>
              )}
            </Badge>
          </div>
        </NeuraForgeHeader>
        
        <div className="flex-1 flex relative h-0 overflow-hidden">
          {/* Mobile: Conversation History Sidebar Toggle */}
          {isMobile && (
            <Button
              variant="outline"
              size="icon"
              className="absolute left-4 top-4 z-30 md:hidden"
              onClick={() => {
                setShowConversations(!showConversations);
                setShowAgents(false);
              }}
            >
              {showConversations ? (
                <X className="h-4 w-4" />
              ) : (
                <Menu className="h-4 w-4" />
              )}
            </Button>
          )}
          
          {/* Conversation History Sidebar */}
          <div
            className={cn(
              "fixed inset-y-16 left-0 z-20 w-80 transition-transform duration-300 ease-in-out md:relative md:inset-y-0 md:translate-x-0",
              showConversations ? "translate-x-0" : "-translate-x-full"
            )}
          >
            <ConversationHistory
              selectedId={selectedConversation}
              onSelect={(id) => {
                setSelectedConversation(id);
                if (isMobile) setShowConversations(false);
              }}
            />
          </div>
          
          {/* Mobile: Agent Sidebar Toggle */}
          {isMobile && (
            <Button
              variant="outline"
              size="icon"
              className="absolute right-4 top-4 z-30 md:hidden"
              onClick={() => {
                setShowAgents(!showAgents);
                setShowConversations(false);
              }}
            >
              {showAgents ? (
                <X className="h-4 w-4" />
              ) : (
                <Users className="h-4 w-4" />
              )}
            </Button>
          )}
          
          {/* Agent Sidebar */}
          <div
            className={cn(
              "fixed inset-y-16 right-0 z-20 w-80 transition-transform duration-300 ease-in-out md:relative md:inset-y-0 md:translate-x-0",
              showAgents ? "translate-x-0" : "translate-x-full"
            )}
          >
            <AgentSidebar activeAgents={activeAgents} />
          </div>
          
          {/* Main Chat Area */}
          <div className="flex-1 flex flex-col">
            <ChatInterface 
              className="flex-1" 
              onSendExample={(example) => handleSendMessage(example, true)}
            />
            <PromptInput onSend={handleSendMessage} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Index;
