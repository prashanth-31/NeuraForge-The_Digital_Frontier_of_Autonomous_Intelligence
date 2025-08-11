"""EnterpriseAgent: business strategy and operations."""
from __future__ import annotations

from typing import Any, Dict

from .base_agent import BaseAgent, AgentInput, AgentOutput
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from ..tools import wikidata_search_tool

REACT_PROMPT = PromptTemplate(
    input_variables=["input", "tool_names", "tools"],
    template=(
        "You are an enterprise strategy assistant. Use tools for entity data.\n"
        "Available tools: {tool_names}\n"
        "{tools}\n"
        "Task: {input}\n"
        "Follow the ReAct format:\nThought: ...\nAction: one of [{tool_names}]\nAction Input: <args as JSON>\nObservation: ...\n... (repeat up to 5 steps)\nFinal Answer: <your answer>\n"
    ),
)


class EnterpriseAgent(BaseAgent):
    def __init__(self, agent_id: str, llm, callbacks=None) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_name="Enterprise Agent",
            agent_type="enterprise",
            llm=llm,
            callbacks=callbacks,
            system_prompt=(
                "You are an enterprise strategy assistant."
                " Provide pragmatic, actionable recommendations with clear steps."
            ),
        )
        tools: list[BaseTool] = [wikidata_search_tool]
        agent = create_react_agent(self.llm, tools, REACT_PROMPT)
        self.executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5, verbose=False)

    async def process(self, agent_input: AgentInput) -> AgentOutput:
        prompt = agent_input.messages[-1]["content"] if agent_input.messages else ""
        try:
            result = await self.executor.ainvoke({"input": prompt})
            answer = result.get("output") or result
            md = {"used_tools": [t.name for t in self.executor.tools]}
            return AgentOutput(content=str(answer), confidence_score=0.8, metadata=md)
        except Exception as e:
            return AgentOutput(
                content=f"Enterprise agent error: {str(e)}",
                confidence_score=0.0,
                metadata={"error": str(e)},
            )
