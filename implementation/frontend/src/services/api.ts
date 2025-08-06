import { Agent } from '@/components/AgentBubble';
import { ENV } from '@/utils/env';

// Interfaces for API requests and responses
export interface Message {
  role: string;
  content: string;
}

export interface AgentInfo {
  id: string;
  name: string;
  type: string;
  confidence: number;
}

export interface ChatRequest {
  messages: Message[];
  user_id?: string;
  metadata?: Record<string, any>;
  stream?: boolean;
}

export interface ChatResponse {
  content: string;
  agent: AgentInfo;
  metadata?: Record<string, any>;
  done?: boolean;
}

/**
 * API client for NeuraForge backend
 */
export class NeuraForgeAPI {
  private baseUrl: string;
  private websocket: WebSocket | null = null;
  private messageCallbacks: ((message: ChatResponse) => void)[] = [];
  private statusCallbacks: ((status: 'connecting' | 'connected' | 'disconnected' | 'error') => void)[] = [];
  private connectionStatus: 'connecting' | 'connected' | 'disconnected' | 'error' = 'disconnected';

  constructor(baseUrl: string = ENV.API_URL) {
    this.baseUrl = baseUrl;
  }

  /**
   * Get the current connection status
   */
  getStatus(): 'connecting' | 'connected' | 'disconnected' | 'error' {
    return this.connectionStatus;
  }

  /**
   * Connect to the WebSocket API
   */
  connect(): void {
    if (this.websocket && (this.websocket.readyState === WebSocket.OPEN || this.websocket.readyState === WebSocket.CONNECTING)) {
      return; // Already connected or connecting
    }

    try {
      this.updateStatus('connecting');
      this.websocket = new WebSocket(`${this.baseUrl.replace('http', 'ws')}/ws/chat`);

      this.websocket.onopen = () => {
        this.updateStatus('connected');
      };

      this.websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as ChatResponse;
          this.notifyMessageCallbacks(data);
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      this.websocket.onclose = () => {
        this.updateStatus('disconnected');
      };

      this.websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        this.updateStatus('error');
      };
    } catch (err) {
      console.error('Error connecting to WebSocket:', err);
      this.updateStatus('error');
    }
  }

  /**
   * Disconnect from the WebSocket API
   */
  disconnect(): void {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
      this.updateStatus('disconnected');
    }
  }

  /**
   * Send a chat message via WebSocket
   */
  sendWebSocketMessage(request: ChatRequest): void {
    if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
      console.log("WebSocket not open, connecting and trying again in 1s");
      this.connect();
      // Queue the message to be sent when connection is established
      setTimeout(() => this.sendWebSocketMessage(request), 1000);
      return;
    }

    console.log("Sending message via WebSocket:", request);
    this.websocket.send(JSON.stringify(request));
  }

  /**
   * Send a chat message via REST API
   */
  async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    try {
      console.log("Sending REST API request to:", `${this.baseUrl}/chat`);
      console.log("Request body:", JSON.stringify(request));
      
      const response = await fetch(`${this.baseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`API error (${response.status}):`, errorText);
        throw new Error(`API error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log("REST API response:", data);
      return data as ChatResponse;
    } catch (error) {
      console.error('Error sending chat message:', error);
      throw error;
    }
  }

  /**
   * Convert API response to Agent format
   */
  static responseToAgent(response: ChatResponse, id: string = 'api-agent'): Agent {
    // Map agent type from backend to frontend
    let agentType: 'research' | 'creative' | 'finance' | 'enterprise' = 'enterprise';
    
    if (response.agent?.type) {
      const type = response.agent.type.toLowerCase();
      if (type === 'research') {
        agentType = 'research';
      } else if (type === 'creative') {
        agentType = 'creative'; 
      } else if (type === 'financial' || type === 'finance') {
        agentType = 'finance';
      } else if (type === 'enterprise' || type === 'business') {
        agentType = 'enterprise';
      }
    }
    
    return {
      id: id,
      name: response.agent?.name || 'NeuraForge',
      type: agentType,
      confidence: response.agent?.confidence ? Math.round(response.agent.confidence * 100) : 95,
      status: 'complete',
      response: response.content,
      metadata: response.metadata ? Object.keys(response.metadata) : undefined
    };
  }

  /**
   * Register a callback for WebSocket messages
   */
  onMessage(callback: (message: ChatResponse) => void): () => void {
    this.messageCallbacks.push(callback);
    return () => {
      this.messageCallbacks = this.messageCallbacks.filter(cb => cb !== callback);
    };
  }

  /**
   * Register a callback for connection status changes
   */
  onStatusChange(callback: (status: 'connecting' | 'connected' | 'disconnected' | 'error') => void): () => void {
    this.statusCallbacks.push(callback);
    // Immediately notify with current status
    callback(this.connectionStatus);
    return () => {
      this.statusCallbacks = this.statusCallbacks.filter(cb => cb !== callback);
    };
  }

  /**
   * Update connection status and notify callbacks
   */
  private updateStatus(status: 'connecting' | 'connected' | 'disconnected' | 'error'): void {
    this.connectionStatus = status;
    this.notifyStatusCallbacks();
  }

  /**
   * Notify all message callbacks
   */
  private notifyMessageCallbacks(message: ChatResponse): void {
    this.messageCallbacks.forEach(callback => callback(message));
  }

  /**
   * Notify all status callbacks
   */
  private notifyStatusCallbacks(): void {
    this.statusCallbacks.forEach(callback => callback(this.connectionStatus));
  }
}

// Singleton instance
export const api = new NeuraForgeAPI();

export default api;
