# NeuraForge: Advanced Multi-Agent Interface

## Overview
NeuraForge is a powerful multi-agent AI interface that enables seamless interaction between various specialized AI agents. The system combines a modern, responsive UI with robust functionality to create an intuitive and efficient user experience.

## Features
- **Multi-Agent Collaboration**: Orchestrate interactions between specialized AI agents (Research, Creative, Finance, Enterprise)
- **Responsive Design**: Full mobile/tablet support with optimized sidebar interactions
- **Theme Support**: Light and dark mode with automatic system preference detection
- **Interactive UI**: Modern animations and transitions with Framer Motion
- **Advanced Chat Interface**: Message reactions, copy functionality, and regeneration options
- **Memory Integration**: Persistent context across conversations
- **Enhanced Input**: Rich text input with suggestions and advanced settings
- **Real-time WebSocket Communication**: Streaming responses from the backend API
- **API Integration**: Seamless connection with the NeuraForge backend services

## Tech Stack
- **Frontend**: React, TypeScript, Tailwind CSS, Shadcn UI
- **State Management**: React Query for server state, React hooks for local state
- **Styling**: Tailwind CSS with custom theme variables
- **Animation**: Framer Motion for fluid UI transitions
- **Routing**: React Router for navigation
- **API Communication**: Fetch API and WebSockets for backend integration

## Getting Started

### Prerequisites

- Node.js 18+ (Latest LTS recommended)
- npm or yarn
- NeuraForge backend running (see backend setup instructions)

### Installation

1. Navigate to the frontend directory:

```bash
cd implementation/frontend
```

2. Install dependencies:

```bash
npm install
# or
yarn
```

3. Create a `.env.local` file with your backend API URL:

```
VITE_API_URL=http://localhost:8000
```

### Development

Start the development server:

```bash
npm run dev
# or
yarn dev
```

The application will be available at [http://localhost:3000](http://localhost:3000).

## Architecture

### Key Components

- **ChatInterface**: Main chat interface with message display
- **PromptInput**: Input area for sending messages with options
- **AgentBubble**: Message bubble with agent information
- **AgentSidebar**: Shows active agents and their status
- **ConversationHistory**: Displays past conversations
- **API Service**: Handles communication with the backend

### API Integration

The frontend communicates with the NeuraForge backend API:

- REST API for stateless requests
- WebSocket for real-time message streaming
- Environment variables for configuration

## Troubleshooting

### Connection Issues

If you experience connection issues:

1. Ensure the backend server is running
2. Check your `.env.local` configuration
3. Verify network connectivity and CORS settings
4. Check the browser console for any error messages

### Build Problems

If you encounter build issues:

1. Clear node_modules and reinstall dependencies
2. Ensure you're using a compatible Node.js version
3. Check for TypeScript errors in your code
