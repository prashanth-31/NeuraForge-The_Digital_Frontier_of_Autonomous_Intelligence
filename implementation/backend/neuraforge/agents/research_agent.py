"""ResearchAgent: focuses on factual queries and retrieval-like responses."""
from __future__ import annotations

from typing import Any, Dict
from datetime import datetime, timezone

from .base_agent import BaseAgent, AgentInput, AgentOutput
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from ..tools import (
    wikipedia_summary_tool,
    arxiv_search_tool,
    crossref_search_tool,
    papers_search_tool,
    web_scrape_tool,
    web_search_tool,
)

REACT_PROMPT = PromptTemplate(
    input_variables=["input", "tool_names", "tools", "today"],
    template=(
        "You are a precise research assistant. Today is {today}.\n"
        "Always use tools for factual or recent data.\n"
    "If the query asks for papers, arXiv results, citations, or 'latest', you MUST first use arxiv_search.\n"
    "If the user provides a URL or asks to summarize a web page/site, use web_scrape.\n"
    "If the query is open-ended or asks for the latest info, first use web_search to find sources, then web_scrape top results.\n"
        "Do not fabricate papers or links. Do NOT say you can't provide real-time information—browse instead using the tools.\n"
        "\nAvailable tools: {tool_names}\n"
        "{tools}\n"
        "Question: {input}\n"
        "Follow the ReAct format strictly:\n"
        "Thought: Decide if a tool is needed.\n"
        "Action: one of [{tool_names}]\n"
        "Action Input: <args as JSON>\n"
        "Observation: <tool output>\n"
        "... (repeat up to 5 steps)\n"
        "Final Answer: A concise, up-to-date answer using verified data with 2-4 citations (titles + direct URLs). Start with 'As of {today}, ...'.\n"
    ),
)


class ResearchAgent(BaseAgent):
    def __init__(self, agent_id: str, llm, callbacks=None) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_name="Research Agent",
            agent_type="research",
            llm=llm,
            callbacks=callbacks,
            system_prompt=(
                "You are a precise research assistant."
                " Answer with accurate facts, cite assumptions, and avoid speculation."
            ),
        )
        tools: list[BaseTool] = [
            wikipedia_summary_tool,
            arxiv_search_tool,
            crossref_search_tool,
            papers_search_tool,
            web_scrape_tool,
            web_search_tool,
        ]
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prompt_with_date = REACT_PROMPT.partial(today=today_str)
        agent = create_react_agent(self.llm, tools, prompt_with_date)
        self.executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5, verbose=False)

    async def process(self, agent_input: AgentInput) -> AgentOutput:
        question = agent_input.messages[-1]["content"] if agent_input.messages else ""
        # Prefer tools when papers=true metadata is set
        force_papers = False
        try:
            force_papers = bool((agent_input.metadata or {}).get("papers"))
        except Exception:
            force_papers = False
        ql = question.lower()
        if force_papers or any(k in ql for k in ["arxiv", "paper", "papers", "latest", "rag", "retrieval-augmented"]):
            question = f"{question}\n\nTool hint: Use papers_search (and/or arxiv_search) to fetch current papers before answering."
        # If a URL is present, hint to use web_scrape
        if any(ql.count(proto) for proto in ["http://", "https://"]):
            question = f"{question}\n\nTool hint: Use web_scrape to extract factual content from the URL before answering."
        # If it's open-ended and time-sensitive, hint to search then scrape
        if any(
            k in ql for k in [
                "latest",
                "news",
                "update",
                "today",
                "release",
                "launch",
                "announce",
                "timeline",
                "date",
            ]
        ) and "http" not in ql:
            question = (
                f"{question}\n\nTool hint: Use web_search to find 2-3 relevant sources, then web_scrape to verify details before the final answer."
            )
        try:
            result = await self.executor.ainvoke({"input": question})
            answer = result.get("output") or result
            md = {"used_tools": [t.name for t in self.executor.tools]}
            return AgentOutput(content=str(answer), confidence_score=0.9, metadata=md)
        except Exception as e:
            return AgentOutput(
                content=f"Research agent error: {str(e)}",
                confidence_score=0.0,
                metadata={"error": str(e)},
            )
