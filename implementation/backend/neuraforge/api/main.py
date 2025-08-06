"""Main FastAPI application for NeuraForge."""

import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from neuraforge.llm import LLMConfig, NeuraForgeLLM

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="NeuraForge API",
    description="API for the NeuraForge intelligent multi-agent system",
    version="0.1.0",
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (not recommended for production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize LLM
llm_config = LLMConfig()
neuraforge_llm = NeuraForgeLLM(config=llm_config)


# Define models
class Message(BaseModel):
    """Message model for chat."""
    
    role: str
    content: str


class ChatRequest(BaseModel):
    """Chat request model."""
    
    messages: List[Message]
    stream: bool = True


class ChatResponse(BaseModel):
    """Chat response model."""
    
    content: str
    agent_info: Optional[str] = None
    confidence_score: Optional[float] = None


# API endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to NeuraForge API"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for non-streaming responses."""
    try:
        if not request.messages:
            return ChatResponse(
                content="No messages provided. Please send at least one message.",
                agent_info="system",
                confidence_score=1.0,
            )
        
        # Format messages for LLM
        formatted_prompt = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
        
        # Generate response from LLM
        response = neuraforge_llm.generate(formatted_prompt)
        
        return ChatResponse(
            content=response,
            agent_info="Financial Agent",
            confidence_score=0.95,
        )
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return ChatResponse(
            content="I apologize, but I encountered an error while processing your request. Please try again.",
            agent_info="system",
            confidence_score=0.0,
        )


# WebSocket connection for streaming responses
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for streaming chat responses."""
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process the message using the LLM
            messages = data.get("messages", [])
            if not messages:
                await websocket.send_json({
                    "content": "No messages provided. Please send at least one message.",
                    "agent_info": "system",
                    "confidence_score": 1.0,
                })
                continue
                
            # Format messages for LLM
            formatted_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            
            # Generate response from LLM
            try:
                response = neuraforge_llm.generate(formatted_prompt)
                
                # Send response to client
                await websocket.send_json({
                    "content": response,
                    "agent_info": "Financial Agent",
                    "confidence_score": 0.95,
                })
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                await websocket.send_json({
                    "content": "I apologize, but I encountered an error while processing your request. Please try again.",
                    "agent_info": "system",
                    "confidence_score": 0.0,
                })
    
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)
