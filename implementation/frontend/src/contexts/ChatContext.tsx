import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { api, ChatResponse, Message, NeuraForgeAPI } from '@/services/api';
import { Agent } from '@/components/AgentBubble';

interface ChatContextType {
  messages: Agent[];
  sendMessage: (content: string, useMemory?: boolean) => void;
  connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error';
}

// Create the context with a default value
const ChatContext = createContext<ChatContextType>({
  messages: [],
  sendMessage: () => {},
  connectionStatus: 'disconnected'
});

// Hook to use the chat context
export const useChatContext = () => useContext(ChatContext);

// Provider component
export const ChatProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [messages, setMessages] = useState<Agent[]>([]);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'error'>(
    api.getStatus()
  );
  const messagesRef = useRef<Agent[]>([]);

  // Keep a reference to the current messages array
  useEffect(() => {
    messagesRef.current = messages;
    console.log("Messages updated in context:", messages); // Debug log
  }, [messages]);

  // Connect to WebSocket
  useEffect(() => {
    console.log("Setting up WebSocket connection in context..."); // Debug log
    api.connect();

    // Handle connection status changes
    const unsubscribeStatus = api.onStatusChange((status) => {
      console.log("WebSocket status changed:", status); // Debug log
      setConnectionStatus(status);
    });

    // Handle incoming messages
    const unsubscribeMessage = api.onMessage((response: ChatResponse) => {
      console.log("Received WebSocket message in context:", response); // Debug log
      
      if (response.done === false) {
        // Handle streaming response if implemented in the future
        console.log("Received partial response (streaming)"); // Debug log
        return;
      }

      // Create agent from response
      const newAgent = NeuraForgeAPI.responseToAgent(
        response, 
        `agent-${Date.now()}`
      );

      console.log("Created agent from response in context:", newAgent); // Debug log

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
      console.log("WebSocket cleanup completed in context"); // Debug log
    };
  }, []);

  // Send a message
  const sendMessage = useCallback((content: string, useMemory: boolean = true) => {
    // Ignore empty messages
    if (!content.trim()) return;
    
    console.log("Sending message from context:", content, "Use memory:", useMemory); // Debug log
    
    // Create user message
    const userMessage: Agent = {
      id: `user-${Date.now()}`,
      name: 'You',
      type: 'research', // Type doesn't matter for user messages
      confidence: 100,
      status: 'complete',
      response: content
    };

    // Add user message to state
    setMessages(prevMessages => [...prevMessages, userMessage]);
    console.log("Added user message to state in context:", userMessage); // Debug log

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
    console.log("Added temporary agent message in context:", tempAgent); // Debug log

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

    // Generate a unique user ID if not available
    const userId = localStorage.getItem('userId') || `user-${Date.now()}`;
    if (!localStorage.getItem('userId')) {
      localStorage.setItem('userId', userId);
    }

    console.log('Sending to backend from context:', apiMessages); // Debug log

    // Prepare the request object
    const request = {
      messages: apiMessages,
      user_id: userId,
      metadata: {
        client: 'web',
        useMemory: useMemory,
        clientTimestamp: new Date().toISOString()
      }
    };

    // Try WebSocket first, fall back to REST API
    if (connectionStatus === 'connected') {
      try {
        console.log("Using WebSocket for message from context"); // Debug log
        api.sendWebSocketMessage(request);
      } catch (err) {
        console.error('WebSocket send failed in context, falling back to REST:', err);
        sendViaRest(request, tempAgentId);
      }
    } else {
      // Use REST API
      console.log("Using REST API for message from context (WebSocket not connected)"); // Debug log
      sendViaRest(request, tempAgentId);
    }
  }, [connectionStatus]);

  // Helper function to send via REST API
  const sendViaRest = (apiRequest: any, tempAgentId: string) => {
    api.sendChatMessage(apiRequest)
      .then(response => {
        console.log('Response from backend REST API in context:', response); // Debug log
        
        // Remove the temporary message
        setMessages(prevMessages => prevMessages.filter(m => m.id !== tempAgentId));
        console.log("Removed temporary message in context:", tempAgentId); // Debug log
        
        // Create agent from response
        const newAgent = NeuraForgeAPI.responseToAgent(
          response, 
          `agent-${Date.now()}`
        );
        
        console.log("Created agent from REST response in context:", newAgent); // Debug log
        setMessages(prevMessages => [...prevMessages, newAgent]);
      })
      .catch(error => {
        console.error('Error sending message in context:', error);
        
        // Remove the temporary message
        setMessages(prevMessages => prevMessages.filter(m => m.id !== tempAgentId));
        console.log("Removed temporary message after error in context:", tempAgentId); // Debug log
        
        // Add error message
        const errorAgent: Agent = {
          id: `error-${Date.now()}`,
          name: 'System',
          type: 'research',
          confidence: 0,
          status: 'complete',
          response: 'Sorry, there was an error processing your request. Please try again or check if the backend server is running.'
        };
        
        console.log("Added error message in context:", errorAgent); // Debug log
        setMessages(prevMessages => [...prevMessages, errorAgent]);
      });
  };

  const contextValue: ChatContextType = {
    messages,
    sendMessage,
    connectionStatus
  };

  return (
    <ChatContext.Provider value={contextValue}>
      {children}
    </ChatContext.Provider>
  );
};
