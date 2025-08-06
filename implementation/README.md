# NeuraForge Implementation

This repository contains the implementation of the NeuraForge system, an intelligent multi-agent architecture.

## Project Structure

The project is organized according to the layered architecture defined in the architecture document:

```
implementation/
├── backend/              # Backend services
│   ├── core/             # Core server and shared utilities
│   ├── agents/           # Domain-specific agent implementations
│   ├── llm/              # LLM integration layer
│   ├── memory/           # Memory systems implementation
│   └── rag/              # Retrieval-Augmented Generation implementation
├── frontend/             # Next.js frontend application
└── infrastructure/       # Deployment and infrastructure configuration
```

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker and Docker Compose
- Git

### Setup Instructions

1. Clone this repository
2. Set up the Python environment:
   ```
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Set up the frontend:
   ```
   cd frontend
   npm install
   ```
4. Configure environment variables (see `.env.example` files)
5. Start development services:
   ```
   docker-compose up -d
   ```

## Phase 1 Implementation

Currently implementing Phase 1 (Foundation):

- Development environment setup
- Core LLM integration using LangChain and Ollama
- Basic UI framework with React and Next.js

## Architecture Reference

For detailed architecture information, refer to the [NeuraForge Architecture Document](../NeuraForge_Architecture.md).
