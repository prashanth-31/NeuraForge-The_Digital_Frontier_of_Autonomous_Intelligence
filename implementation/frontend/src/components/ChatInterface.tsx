'use client';

import { useState, useRef, useEffect } from 'react';

// Define API URL as a constant that can be easily changed
const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

type Message = {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  agentInfo?: string;
  confidenceScore?: number;
  timestamp?: number;
  isError?: boolean;
};

// Function to format code blocks in messages
const formatMessageContent = (content: string): JSX.Element => {
  // Simple markdown-like formatter for code blocks and line breaks
  const parts = content.split(/(```[\s\S]*?```)/g);
  
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith('```') && part.endsWith('```')) {
          // Extract the code and language
          const code = part.slice(3, -3);
          const firstLineBreak = code.indexOf('\n');
          const language = firstLineBreak > 0 ? code.slice(0, firstLineBreak).trim() : '';
          const codeContent = firstLineBreak > 0 ? code.slice(firstLineBreak + 1) : code;
          
          return (
            <pre key={i} className="bg-gray-900 dark:bg-gray-950 text-gray-100 p-3 rounded my-2 overflow-x-auto">
              {language && (
                <div className="text-xs text-gray-400 pb-1 border-b border-gray-700 mb-2">
                  {language}
                </div>
              )}
              <code>{codeContent}</code>
            </pre>
          );
        } else {
          // Process line breaks and other text
          return (
            <span key={i} style={{ whiteSpace: "pre-wrap" }}>
              {part.split('\n').map((line, j, arr) => (
                <span key={j}>
                  {line}
                  {j < arr.length - 1 && <br />}
                </span>
              ))}
            </span>
          );
        }
      })}
    </>
  );
};

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '0',
      role: 'assistant',
      content: 'Hello! I\'m NeuraForge, your AI assistant. How can I help you today?',
      agentInfo: 'welcome_agent',
      confidenceScore: 1.0,
      timestamp: Date.now(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'error'>('connected');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Focus input on load
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  // Check API connection on load
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const response = await fetch(`${API_URL}/`);
        if (response.ok) {
          setConnectionStatus('connected');
        } else {
          setConnectionStatus('error');
        }
      } catch (error) {
        console.error('API connection error:', error);
        setConnectionStatus('error');
      }
    };
    
    checkConnection();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isProcessing) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsProcessing(true);

    try {
      // In Phase 1, we'll just use the REST API
      // In later phases, we'll switch to WebSockets for streaming
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          messages: [...messages, userMessage].map(({ role, content }) => ({
            role,
            content,
          })),
          stream: false,
        }),
      });

      if (!response.ok) {
        throw new Error(`API responded with status: ${response.status}`);
      }

      const data = await response.json();

      const assistantMessage: Message = {
        id: Date.now().toString(),
        role: 'assistant',
        content: data.content,
        agentInfo: data.agent_info,
        confidenceScore: data.confidence_score,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setConnectionStatus('connected');
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Add error message
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: 'assistant',
          content: 'Sorry, there was an error connecting to the NeuraForge API. Please make sure the backend server is running.',
          isError: true,
          timestamp: Date.now(),
        },
      ]);
      setConnectionStatus('error');
    } finally {
      setIsProcessing(false);
    }
  };

  // Function to render the typing indicator
  const renderTypingIndicator = () => {
    return (
      <div className="flex items-center space-x-2 p-2">
        <div className="h-6 w-6 rounded-full bg-gradient-to-br from-neural-light to-neural-dark flex items-center justify-center text-white font-medium shadow-md pulse-animation text-xs">
          NF
        </div>
        <div className="glassmorphism px-4 py-2 rounded-xl typing-indicator">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full max-w-5xl mx-auto relative">
      {/* Neural network pattern background */}
      <div className="absolute inset-0 neural-bg opacity-30 pointer-events-none"></div>
      
      {/* Connection status indicator */}
      {connectionStatus === 'error' && (
        <div className="bg-gradient-to-r from-red-500 to-red-600 text-white px-6 py-3 text-center font-medium shadow-md z-10">
          <div className="flex items-center justify-center space-x-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
            </svg>
            <span>Error connecting to the NeuraForge API. Make sure the backend server is running at <span className="font-mono bg-red-600/50 px-2 py-0.5 rounded">{API_URL}</span></span>
          </div>
        </div>
      )}
      
      {/* Connection status success */}
      {connectionStatus === 'connected' && (
        <div className="bg-gradient-to-r from-neural-main to-forge-main text-white px-4 py-2 text-center font-medium shadow-md z-10">
          <div className="flex items-center justify-center">
            <span className="text-sm">
              <span className="inline-block h-2 w-2 rounded-full bg-green-300 mr-2 animate-pulse"></span>
              Connected to NeuraForge API with LLaMA 3.1:8b model
            </span>
          </div>
        </div>
      )}
      
      <div className="flex-1 overflow-y-auto px-4 md:px-6 py-6 space-y-6 mb-2 scroll-smooth relative z-10">
        {/* Decorative floating elements */}
        <div className="absolute right-10 top-20 w-32 h-32 bg-neural-main/5 dark:bg-neural-main/10 rounded-full filter blur-3xl float-animation"></div>
        <div className="absolute left-10 top-40 w-24 h-24 bg-forge-main/5 dark:bg-forge-main/10 rounded-full filter blur-2xl float-animation" style={{ animationDelay: '2s' }}></div>
        
        {messages.map((message, idx) => (
          <div
            key={message.id}
            className={`flex items-end ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            } ${message.isError ? 'opacity-90' : ''}`}
          >
            {message.role !== 'user' && (
              <div className="flex-shrink-0 mr-2 self-end mb-1">
                <div className="h-6 w-6 rounded-lg bg-gradient-to-br from-neural-light to-neural-dark flex items-center justify-center text-white font-medium shadow-md text-xs">
                  NF
                </div>
              </div>
            )}
            
            <div
              className={`max-w-[85%] md:max-w-[75%] rounded-2xl p-4 shadow-message ${
                message.role === 'user'
                  ? 'bg-gradient-to-r from-neural-main to-neural-dark text-white message-user'
                  : message.isError 
                    ? 'bg-red-50 text-red-800 dark:bg-red-900/40 dark:text-red-200 border border-red-200 dark:border-red-800/30'
                    : 'glassmorphism message-assistant'
              }`}
            >
              {formatMessageContent(message.content)}
              
              {message.agentInfo && (
                <div className="mt-3 pt-2 border-t border-white/20 dark:border-gray-700 text-xs flex flex-wrap justify-between items-center gap-2">
                  <div className="flex items-center">
                    <svg className="w-3.5 h-3.5 mr-1 opacity-70" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                      <path fillRule="evenodd" d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" clipRule="evenodd"></path>
                    </svg>
                    <span className="font-medium gradient-text">{message.agentInfo}</span>
                  </div>
                  
                  <div className="flex space-x-3 opacity-80">
                    {message.confidenceScore && (
                      <div className="flex items-center">
                        <div className="w-10 bg-gray-300 dark:bg-gray-700 h-1.5 rounded-full mr-1.5 overflow-hidden">
                          <div 
                            className="h-full bg-neural-accent dark:bg-neural-light rounded-full" 
                            style={{width: `${(message.confidenceScore * 100).toFixed(0)}%`}}
                          ></div>
                        </div>
                        <span>{(message.confidenceScore * 100).toFixed(0)}%</span>
                      </div>
                    )}
                    
                    {message.timestamp && (
                      <span className="font-mono text-[10px]" suppressHydrationWarning>
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
            
            {message.role === 'user' && (
              <div className="flex-shrink-0 ml-2 self-end mb-1">
                <div className="h-6 w-6 rounded-lg bg-gradient-to-br from-forge-light to-forge-dark flex items-center justify-center text-white shadow-md">
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path>
                  </svg>
                </div>
              </div>
            )}
          </div>
        ))}
        
        {isProcessing && renderTypingIndicator()}
        
        <div ref={messagesEndRef} />
      </div>

      <div className="relative border-t dark:border-gray-700/50 p-4 md:p-6 bg-white/80 dark:bg-gray-900/80 backdrop-blur-lg z-10">
        {/* Decorative elements */}
        <div className="absolute left-6 -top-12 w-24 h-24 bg-neural-main/5 dark:bg-neural-main/10 rounded-full filter blur-2xl"></div>
        <div className="absolute right-6 -top-8 w-16 h-16 bg-forge-main/5 dark:bg-forge-main/10 rounded-full filter blur-2xl"></div>
        
        <form onSubmit={handleSubmit} className="flex space-x-3 relative">
          <div className="relative flex-1">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isProcessing}
              className="input-primary w-full pl-10 pr-4 py-3 text-base"
              placeholder="Type your message..."
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  if (!isProcessing && input.trim()) {
                    handleSubmit(e);
                  }
                }
              }}
            />
            <div className="absolute left-3 top-1/2 transform -translate-y-1/2 text-neural-main opacity-70">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path>
              </svg>
            </div>
          </div>
          
          <button
            type="submit"
            disabled={isProcessing || !input.trim()}
            className="btn-primary min-w-16 flex items-center justify-center"
            aria-label="Send message"
          >
            {isProcessing ? (
              <span className="flex items-center">
                <svg
                  className="animate-spin mr-1.5 h-4 w-4 text-white"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                <span className="text-sm">Processing</span>
              </span>
            ) : (
              <span className="flex items-center">
                <svg 
                  className="mr-1 h-4 w-4" 
                  fill="currentColor" 
                  viewBox="0 0 20 20"
                >
                  <path 
                    fillRule="evenodd" 
                    d="M10.293 5.293a1 1 0 011.414 0l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414-1.414L12.586 11H5a1 1 0 110-2h7.586l-2.293-2.293a1 1 0 010-1.414z" 
                    clipRule="evenodd"
                  />
                </svg>
                <span className="text-sm">Send</span>
              </span>
            )}
          </button>
        </form>
        
        {/* Features hint */}
        <div className="mt-4 p-3 bg-white/30 dark:bg-gray-800/30 backdrop-blur-sm rounded-xl shadow-sm border border-gray-100 dark:border-gray-700/50">
          <div className="text-xs text-center text-gray-700 dark:text-gray-300 font-medium mb-2">
            NeuraForge Features
          </div>
          <div className="flex flex-wrap justify-center gap-2 text-xs text-gray-500 dark:text-gray-400">
            <span className="flex items-center bg-white/50 dark:bg-gray-700/50 px-2 py-1 rounded-md shadow-sm">
              <svg className="w-2.5 h-2.5 mr-1" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z"></path>
              </svg>
              LLaMA 3.1:8b
            </span>
            <span className="flex items-center bg-white/50 dark:bg-gray-700/50 px-2 py-1 rounded-md shadow-sm">
              <svg className="w-2.5 h-2.5 mr-1" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                <path fillRule="evenodd" d="M12.316 3.051a1 1 0 01.633 1.265l-4 12a1 1 0 11-1.898-.632l4-12a1 1 0 011.265-.633zM5.707 6.293a1 1 0 010 1.414L3.414 10l2.293 2.293a1 1 0 11-1.414 1.414l-3-3a1 1 0 010-1.414l3-3a1 1 0 011.414 0zm8.586 0a1 1 0 011.414 0l3 3a1 1 0 010 1.414l-3 3a1 1 0 11-1.414-1.414L16.586 10l-2.293-2.293a1 1 0 010-1.414z" clipRule="evenodd"></path>
              </svg>
              Code formatting
            </span>
            <span className="flex items-center bg-white/50 dark:bg-gray-700/50 px-2 py-1 rounded-md shadow-sm">
              <svg className="w-2.5 h-2.5 mr-1" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5 1 1 0 11-1.731-1A3 3 0 0113 8a3.001 3.001 0 01-2 2.83V11a1 1 0 11-2 0v-1a1 1 0 011-1 1 1 0 100-2zm0 8a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd"></path>
              </svg>
              Ask anything
            </span>
            <span className="flex items-center bg-white/50 dark:bg-gray-700/50 px-2 py-1 rounded-md shadow-sm">
              <svg className="w-2.5 h-2.5 mr-1" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                <path d="M5 4a1 1 0 00-2 0v7.268a2 2 0 000 3.464V16a1 1 0 102 0v-1.268a2 2 0 000-3.464V4zM11 4a1 1 0 10-2 0v1.268a2 2 0 000 3.464V16a1 1 0 102 0V8.732a2 2 0 000-3.464V4zM16 3a1 1 0 011 1v7.268a2 2 0 010 3.464V16a1 1 0 11-2 0v-1.268a2 2 0 010-3.464V4a1 1 0 011-1z"></path>
              </svg>
              Advanced settings
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
