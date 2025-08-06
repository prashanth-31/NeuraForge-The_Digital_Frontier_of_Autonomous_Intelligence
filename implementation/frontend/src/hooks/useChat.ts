import { useState, useEffect, useCallback, useRef } from 'react';
import { api, ChatResponse, Message, NeuraForgeAPI } from '@/services/api';
import { Agent } from '@/components/AgentBubble';

interface UseChatReturn {
  messages: Agent[];
  sendMessage: (content: string, useMemory?: boolean) => void;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Agent[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>(
    api.getStatus()
  );
  const messagesRef = useRef<Agent[]>([]);

  // Keep a reference to the current messages array
  useEffect(() => {
    messagesRef.current = messages;
    console.log("Messages updated:", messages); // Debug log
  }, [messages]);

  // Connect to WebSocket
  useEffect(() => {
    console.log("Setting up WebSocket connection..."); // Debug log
    api.connect();

    // Handle connection status changes
    const unsubscribeStatus = api.onStatusChange((status) => {
      console.log("WebSocket status changed:", status); // Debug log
      setConnectionStatus(status);
    });

    // Handle incoming messages
    const unsubscribeMessage = api.onMessage((response: ChatResponse) => {
      console.log("Received WebSocket message:", response); // Debug log
      
      if (response.done === false) {
        // Handle streaming response if implemented in the future
        console.log("Received partial response (streaming)"); // Debug log
        return;
      }

      const agentType = determineAgentType(response.content);
      const newAgent = NeuraForgeAPI.responseToAgent(
        response, 
        `agent-${Date.now()}`,
        agentType
      );

      console.log("Created agent from response:", newAgent); // Debug log

      setMessages((prevMessages) => {
        // Filter out any temporary thinking messages first
        const filteredMessages = prevMessages.filter(
          m => !m.status?.includes('thinking')
        );
        return [...filteredMessages, newAgent];
      });
    });

    return () => {
      unsubscribeStatus();
      unsubscribeMessage();
      api.disconnect();
      console.log("WebSocket cleanup completed"); // Debug log
    };
  }, []);

  // Determine agent type based on content (simplified logic)
  const determineAgentType = (content: string): 'research' | 'creative' | 'finance' | 'enterprise' => {
    const contentLower = content.toLowerCase();
    
    if (contentLower.includes('data') || contentLower.includes('research') || contentLower.includes('analysis')) {
      return 'research';
    } else if (contentLower.includes('creative') || contentLower.includes('design') || contentLower.includes('ideas')) {
      return 'creative';
    } else if (contentLower.includes('finance') || contentLower.includes('budget') || contentLower.includes('investment')) {
      return 'finance';
    } else {
      return 'enterprise';
    }
  };

  // Send a message
  const sendMessage = useCallback((content: string, useMemory: boolean = true) => {
    // Ignore empty messages
    if (!content.trim()) return;
    
    console.log("Sending message:", content, "Use memory:", useMemory); // Debug log
    
    // Create user message
    const userMessage: Agent = {
      id: `user-${Date.now()}`,
      name: 'You',
      type: 'research',
      confidence: 100,
      status: 'complete',
      response: content
    };

    // Add user message to state
    setMessages(prevMessages => [...prevMessages, userMessage]);
    console.log("Added user message to state:", userMessage); // Debug log

    // Create temporary agent message
    const tempAgentId = `agent-temp-${Date.now()}`;
    const tempAgent: Agent = {
      id: tempAgentId,
      name: 'NeuraForge',
      type: 'research',
      confidence: 0,
      status: 'thinking',
      response: 'Thinking...'
    };

    // Add temporary message
    setMessages(prevMessages => [...prevMessages, tempAgent]);
    console.log("Added temporary agent message:", tempAgent); // Debug log

    // Format messages for API
    // If useMemory is true, include more context from previous messages
    const currentMessages = messagesRef.current;
    const contextSize = useMemory ? 6 : 2; // Use more context when memory is enabled
    
    // Start with either all messages or just the last few based on useMemory
    const relevantMessages = useMemory 
      ? currentMessages.filter(m => m.status !== 'thinking') 
      : currentMessages.slice(-contextSize).filter(m => m.status !== 'thinking');
    
    // Add the new user message if it's not already included
    if (!relevantMessages.find(m => m.id === userMessage.id)) {
      relevantMessages.push(userMessage);
    }
    
    // Format for API
    const apiMessages: Message[] = relevantMessages.map(m => ({
      role: m.name === 'You' ? 'user' : 'assistant',
      content: m.response || ''
    }));

    console.log('Sending to backend:', apiMessages); // Debug log

    // Try WebSocket first, fall back to REST API
    if (connectionStatus === 'connected') {
      try {
        console.log("Using WebSocket for message"); // Debug log
        api.sendWebSocketMessage({ messages: apiMessages });
      } catch (err) {
        console.error('WebSocket send failed, falling back to REST:', err);
        sendViaRest(apiMessages, tempAgentId);
      }
    } else {
      // Use REST API
      console.log("Using REST API for message (WebSocket not connected)"); // Debug log
      sendViaRest(apiMessages, tempAgentId);
    }
  }, [connectionStatus]);

  // Helper function to send via REST API
  const sendViaRest = (apiMessages: Message[], tempAgentId: string) => {
    api.sendChatMessage({ messages: apiMessages })
      .then(response => {
        console.log('Response from backend REST API:', response); // Debug log
        
        // Remove the temporary message
        setMessages(prevMessages => prevMessages.filter(m => m.id !== tempAgentId));
        console.log("Removed temporary message:", tempAgentId); // Debug log
        
        const agentType = determineAgentType(response.content);
        const newAgent = NeuraForgeAPI.responseToAgent(
          response, 
          `agent-${Date.now()}`,
          agentType
        );
        
        console.log("Created agent from REST response:", newAgent); // Debug log
        setMessages(prevMessages => [...prevMessages, newAgent]);
      })
      .catch(error => {
        console.error('Error sending message:', error);
        
        // Remove the temporary message
        setMessages(prevMessages => prevMessages.filter(m => m.id !== tempAgentId));
        console.log("Removed temporary message after error:", tempAgentId); // Debug log
        
        // Add error message
        const errorAgent: Agent = {
          id: `error-${Date.now()}`,
          name: 'System',
          type: 'research',
          confidence: 0,
          status: 'complete',
          response: 'Sorry, there was an error processing your request. Please try again or check if the backend server is running.'
        };
        
        console.log("Added error message:", errorAgent); // Debug log
        setMessages(prevMessages => [...prevMessages, errorAgent]);
      });
  };

  return {
    messages,
    sendMessage,
    connectionStatus
  };
}
