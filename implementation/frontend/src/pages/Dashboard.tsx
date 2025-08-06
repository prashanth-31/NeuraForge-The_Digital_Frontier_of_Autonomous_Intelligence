import { useState, useEffect } from "react";
import { NeuraForgeHeader } from "@/components/NeuraForgeHeader";
import { ConversationHistory } from "@/components/ConversationHistory";
import { ChatInterface } from "@/components/ChatInterface";
import { AgentSidebar } from "@/components/AgentSidebar";
import { PromptInput } from "@/components/PromptInput";
import type { Agent } from "@/components/AgentBubble";
import { useIsMobile } from "@/hooks/use-mobile";
import { useChatContext } from "@/contexts/ChatContext"; // Import the ChatContext
import { Button } from "@/components/ui/button";
import { Menu, Users, X, Info } from "lucide-react";
import { cn } from "@/lib/utils";
import { useNavigate } from "react-router-dom";
import { FadeIn, SlideIn, PulseEffect, ScaleIn } from "@/components/animations";
import "@/components/animations.css";

const Dashboard = () => {
  const [selectedConversation, setSelectedConversation] = useState<string>();
  const [showConversations, setShowConversations] = useState(false);
  const [showAgents, setShowAgents] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const isMobile = useIsMobile();
  const navigate = useNavigate();
  
  // Use the chat hook to manage messages and send messages
  const { messages, sendMessage, connectionStatus } = useChatContext();
  
  // Simulate loading for smoother animations
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 100);
    return () => clearTimeout(timer);
  }, []);
  
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

  const handleSendMessage = (message: string, useMemory: boolean = true) => {
    console.log('Sending message:', message, 'Use memory:', useMemory);
    
    // Call the chat service with both parameters
    sendMessage(message, useMemory);
    
    // Close mobile sidebars when sending a message
    if (isMobile) {
      setShowConversations(false);
      setShowAgents(false);
    }
  };

  return (
    <FadeIn className="h-screen flex flex-col bg-background overflow-hidden">
      {/* Grid pattern background with animation */}
      <div 
        className="fixed inset-0 animate-fade-in"
        style={{
          backgroundImage: 'radial-gradient(circle at 1px 1px, hsl(var(--neura-blue)) 1px, transparent 0)',
          backgroundSize: '20px 20px',
        }}
      />
      
      <div className="relative z-10 flex flex-col h-full">
        <NeuraForgeHeader />
        
        <div className="flex-1 flex relative overflow-hidden">
          {/* Project Details Button with pulse effect */}
          <PulseEffect className="absolute right-4 top-4 z-30 md:hidden rounded-full">
            <Button
              variant="outline"
              size="icon"
              className="rounded-full border-neura-blue/30 pulse-animation"
              onClick={() => navigate("/project-details")}
            >
              <Info className="h-4 w-4 text-neura-blue" />
            </Button>
          </PulseEffect>
          
          {/* Mobile: Conversation History Sidebar Toggle with animation */}
          {isMobile && (
            <ScaleIn delay={0.3} className="absolute left-4 top-4 z-30 md:hidden">
              <Button
                variant="outline"
                size="icon"
                className="rounded-full button-hover"
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
            </ScaleIn>
          )}
          
          {/* Conversation History Sidebar with slide animation */}
          <div
            className={cn(
              "fixed inset-y-16 left-0 z-20 w-80 transition-transform duration-300 ease-in-out md:relative md:inset-y-0 md:translate-x-0 backdrop-blur-sm",
              showConversations ? "translate-x-0" : "-translate-x-full"
            )}
          >
            <SlideIn direction="left" delay={0.2} className="h-full">
              <ConversationHistory
                selectedId={selectedConversation}
                onSelect={(id) => {
                  setSelectedConversation(id);
                  if (isMobile) setShowConversations(false);
                }}
              />
            </SlideIn>
          </div>
          
          {/* Mobile: Agent Sidebar Toggle with animation */}
          {isMobile && (
            <ScaleIn delay={0.4} className="absolute right-16 top-4 z-30 md:hidden">
              <Button
                variant="outline"
                size="icon"
                className="rounded-full button-hover"
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
            </ScaleIn>
          )}
          
          {/* Agent Sidebar with slide animation */}
          <div
            className={cn(
              "fixed inset-y-16 right-0 z-20 w-80 transition-transform duration-300 ease-in-out md:relative md:inset-y-0 md:translate-x-0 backdrop-blur-sm",
              showAgents ? "translate-x-0" : "translate-x-full"
            )}
          >
            <SlideIn direction="right" delay={0.2} className="h-full">
              <AgentSidebar activeAgents={activeAgents} />
            </SlideIn>
          </div>
          
          {/* Main Chat Area with fade in animation */}
          <FadeIn className="flex-1 flex flex-col" delay={0.3}>
            <ChatInterface className="flex-1" onSendExample={handleSendMessage} />
            <ScaleIn delay={0.5}>
              <PromptInput onSend={handleSendMessage} />
            </ScaleIn>
          </FadeIn>
        </div>
      </div>
    </FadeIn>
  );
};

export default Dashboard;
