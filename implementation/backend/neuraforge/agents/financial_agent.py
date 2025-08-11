"""FinancialAgent: computations and financially-oriented analysis."""
from __future__ import annotations

import math
from typing import Any, Dict

from .base_agent import BaseAgent, AgentInput, AgentOutput

from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from ..tools import price_series_tool, fx_convert_tool

REACT_PROMPT = PromptTemplate(
    input_variables=["input", "tool_names", "tools"],
    template=(
        "You are a financial analysis assistant. Use tools for data where possible.\n"
        "Available tools: {tool_names}\n"
        "{tools}\n"
        "Request: {input}\n"
        "Follow the ReAct format:\nThought: ...\nAction: one of [{tool_names}]\nAction Input: <args as JSON>\nObservation: ...\n... (repeat up to 5 steps)\nFinal Answer: <your answer>\n"
    ),
)


class FinancialAgent(BaseAgent):
    def __init__(self, agent_id: str, llm, callbacks=None) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_name="Financial Agent",
            agent_type="financial",
            llm=llm,
            callbacks=callbacks,
            system_prompt=(
                "You are a financial analysis assistant."
                " Provide clear reasoning, formulas, and avoid speculative advice."
            ),
        )
        tools: list[BaseTool] = [price_series_tool, fx_convert_tool]
        agent = create_react_agent(self.llm, tools, REACT_PROMPT)
        self.executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5, verbose=False)

    def _npv(self, rate: float, cashflows):
        return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cashflows))

    def _irr(self, cashflows, guess=0.1):
        rate = guess
        for _ in range(50):
            npv = 0.0
            d_npv = 0.0
            for t, cf in enumerate(cashflows):
                npv += cf / ((1 + rate) ** t)
                if t > 0:
                    d_npv += -t * cf / ((1 + rate) ** (t + 1))
            if abs(d_npv) < 1e-8:
                break
            new_rate = rate - npv / d_npv
            if abs(new_rate - rate) < 1e-7:
                rate = new_rate
                break
            rate = new_rate
        return rate

    async def process(self, agent_input: AgentInput) -> AgentOutput:
        last = (agent_input.messages or [])[-1]
        text = (last.get("content") or "").lower()

        # Calculator fast-paths
        try:
            if "npv(" in text:
                rate_str = text.split("rate=")[1].split(",")[0].strip().strip(")")
                cashflows_str = text.split("cashflows=")[1].split(")")[0].strip()
                rate = float(rate_str)
                cashflows = eval(cashflows_str, {"__builtins__": {}}, {})
                result = self._npv(rate, cashflows)
                content = f"NPV at rate {rate:.4f} is {result:.4f}"
                return AgentOutput(content=content, confidence_score=0.95, metadata={"calculator": "npv"})

            if "irr(" in text:
                cashflows_str = text.split("cashflows=")[1].split(")")[0].strip()
                cashflows = eval(cashflows_str, {"__builtins__": {}}, {})
                result = self._irr(cashflows)
                content = f"IRR is {result:.4%}"
                return AgentOutput(content=content, confidence_score=0.9, metadata={"calculator": "irr"})
        except Exception as e:
            return AgentOutput(
                content=f"Financial calculator error: {str(e)}",
                confidence_score=0.0,
                metadata={"error": str(e)},
            )

        # Tool-using ReAct path
        try:
            result = await self.executor.ainvoke({"input": last.get("content", "")})
            answer = result.get("output") or result
            md = {"used_tools": [t.name for t in self.executor.tools]}
            return AgentOutput(content=str(answer), confidence_score=0.8, metadata=md)
        except Exception:
            # Fallback to plain LLM
            try:
                prompt = self.build_prompt(agent_input, "Provide a careful financial analysis with assumptions.")
                llm_text = self.llm.invoke(prompt)
                return AgentOutput(content=llm_text, confidence_score=0.8)
            except Exception as e:
                return AgentOutput(
                    content=f"Financial agent error: {str(e)}",
                    confidence_score=0.0,
                    metadata={"error": str(e)},
                )
