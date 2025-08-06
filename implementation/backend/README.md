# NeuraForge Backend

This is the backend component of the NeuraForge project, implementing the AI agent framework, API, and core functionality.

## Project Structure

```
backend/
├── myenv/                  # Python virtual environment
├── neuraforge/             # Main package
│   ├── __init__.py
│   ├── api/                # API implementation
│   │   ├── __init__.py
│   │   └── main.py
│   ├── llm/                # LLM integration
│   │   ├── __init__.py
│   │   └── core.py
│   └── agents/             # Agent implementations
│       ├── __init__.py
│       ├── base_agent.py
│       ├── creative_agent.py
│       ├── enterprise_agent.py
│       ├── financial_agent.py
│       └── research_agent.py
├── pyproject.toml          # Project metadata
├── requirements.txt        # Dependencies
├── server.py               # Main server entry point
├── setup.py                # Package setup script
└── tests/                  # Test suite
    ├── __init__.py
    ├── conftest.py
    ├── test_orchestration.py
    └── agents/             # Agent-specific tests
        ├── __init__.py
        ├── test_financial_agent.py
        ├── test_financial_agent_manual.py
        ├── test_financial_agent_with_tools.py
        └── simple_financial_test.py
```

## Getting Started

1. Activate the virtual environment:
   ```
   myenv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start the server:
   ```
   python server.py
   ```

## Testing

For detailed testing instructions, see [tests/README.md](tests/README.md).

Quick testing options:

1. Setup the test environment:
   ```
   setup_tests.bat
   ```

2. Run Financial Agent tests:
   ```
   cd tests/agents
   run_financial_tests.bat
   ```

3. Run a simple Financial Agent test with minimal dependencies:
   ```
   python tests/agents/simple_financial_test.py
   ```

## Financial Agent

The Financial Agent is a specialized agent for financial analysis, planning, and investment guidance. It can:

1. Answer questions about financial concepts and strategies
2. Provide budgeting and financial planning advice
3. Offer general investment guidance
4. Perform financial calculations (in the enhanced version)

For detailed testing instructions for the Financial Agent, see [tests/agents/README_FINANCIAL_TESTING.md](tests/agents/README_FINANCIAL_TESTING.md).

## Dependencies

- Python 3.9+
- FastAPI
- LangChain
- Ollama (for LLM access)
- Uvicorn (ASGI server)

## Configuration

The backend can be configured through environment variables or a `.env` file.
