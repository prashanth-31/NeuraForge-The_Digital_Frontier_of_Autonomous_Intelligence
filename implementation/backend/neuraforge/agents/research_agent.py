"""ResearchAgent: focuses on factual queries and retrieval-like responses."""
from __future__ import annotations

from typing import Any, Dict, List
from datetime import datetime, timezone

from .base_agent import BaseAgent, AgentInput, AgentOutput
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
import re
import asyncio
import logging
from ..services import search as search_service, scraper as scraper_service
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
        "MANDATORY PRINCIPLES:\n"
        "1. Always consult tools BEFORE answering if the question involves factual claims, recency, data, metrics, launches, updates, comparisons, papers, citations, or URLs.\n"
        "2. NEVER produce a purely hypothetical answer when real sources can be fetched.\n"
        "3. NEVER say you cannot provide real-time or up-to-date info — instead perform web_search then web_scrape.\n"
        "4. NEVER fabricate papers, numbers, or links; only cite what was actually retrieved.\n"
        "5. If tools return nothing, explicitly say 'No reliable sources found for X after searching.' rather than giving a hypothetical.\n"
        "6. Cite 2-4 real sources with direct URLs.\n"
    "7. For financial projections / forecasts (projection, forecast, estimate, guidance, Q* YYYY), NEVER invent figures. Only report figures explicitly present in sources (earnings releases, SEC filings, reputable news, analyst consensus) and cite each cluster of figures. If no sourced figures found, state that and DO NOT fabricate.\n"
        "If the query asks for papers, arXiv results, citations, or 'latest', you MUST first use arxiv_search or papers_search.\n"
        "If the user provides a URL or asks to summarize a web page/site, use web_scrape.\n"
        "If the query is open-ended or asks for the latest/current/recent info, first use web_search to find sources, then web_scrape top results.\n"
        "Do not fabricate papers or links. Do NOT say you can't provide real-time information—browse instead using the tools.\n"
        "\nAvailable tools: {tool_names}\n"
        "{tools}\n"
        "Question: {input}\n"
        "Follow the ReAct format strictly:\n"
        "Thought: Decide if a tool is needed. (It almost always is.)\n"
        "Action: one of [{tool_names}]\n"
        "Action Input: <args as JSON>\n"
        "Observation: <tool output>\n"
        "... (repeat up to 5 steps)\n"
        "Final Answer: Start with 'As of {today}, ...' and provide ONLY supported facts with 2-4 citations (Title – URL). No hypotheticals.\n"
    ),
)


logger = logging.getLogger(__name__)


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
        # Detect URLs and time-sensitive intent
        url_pattern = r"https?://\S+"
        urls = re.findall(url_pattern, question)
        time_tokens = [
            "latest","news","update","today","release","launch","announce","timeline","date","current","recent","this week","this month","2025","2024","2023"
        ]
        is_time_sensitive = any(k in ql for k in time_tokens)

        # Detect financial / projection intent
        financial_projection_tokens = [
            "projection","projections","forecast","forecasts","estimate","estimates","guidance","eps","revenue","net income","operating income","gross margin","q1","q2","q3","q4","fy","fiscal","segment","datacenter","gaming","nvidia","outlook"
        ]
        is_financial_projection = (
            any(t in ql for t in financial_projection_tokens)
            and any(qtok in ql for qtok in ["q1","q2","q3","q4","fy","2025","2024","guidance","forecast","projection","outlook"])
        )
        if is_financial_projection:
            is_time_sensitive = True  # always treat as time sensitive

        disclaimer_phrases = [
            "hypothetical","i cannot provide real-time","i can't provide real-time","as an ai language model","i do not have access to real-time","cannot provide real time","can't provide real time","possible scenario","purely illustrative","for illustrative purposes","could ","might ","would ","may "
        ]

        async def forced_autobrowse_synthesis(original_question: str, reason: str, financial: bool | None = None) -> AgentOutput:
            """Force search+scrape regardless of triggers (used if disclaimer detected)."""
            try:
                logger.info("Forced autobrowse synthesis invoked (reason=%s)", reason)
                results = await search_service.ddg_search(original_question, max_results=3)
                top_urls = [r.get("url") for r in results if r.get("url")] or []
                if not top_urls:
                    return AgentOutput(
                        content=f"As of {datetime.now(timezone.utc).strftime('%Y-%m-%d')}, no reliable sources found for: {original_question}",
                        confidence_score=0.6,
                        metadata={"autobrowse": True, "forced": True, "reason": reason, "citations": [], "used_tools":["web_search"]},
                    )
                async def scrape(u: str) -> Dict[str,str]:
                    try:
                        text = await scraper_service.scrape_url(u, include_links=False, max_chars=1600)
                        if len(text) < 300:
                            text = await scraper_service.scrape_url(u, include_links=False, max_chars=1600, render_js=True, wait_until="networkidle", wait_ms=1500)
                        return {"url": u, "text": text}
                    except Exception as e:
                        return {"url": u, "text": f"Error scraping {u}: {e}"}
                scraped = await asyncio.gather(*(scrape(u) for u in top_urls[:3]))
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                src_text = []
                for i,s in enumerate(scraped, start=1):
                    snippet = s.get("text","")[:1200]
                    src_text.append(f"[{i}] {s.get('url','')}\n{snippet}")
                if financial is None:
                    financial = is_financial_projection
                if financial:
                    synthesis_prompt = (
                        "You are a precise research assistant. Today is {today}.\n"
                        "FINANCIAL RULES: Use ONLY explicit figures appearing in SOURCES (earnings releases, filings, reputable news).\n"
                        "If figures (segment revenue, EPS, margins, guidance) are NOT present verbatim in sources, state that no sourced Q&A financial guidance figures were found and DO NOT invent estimates.\n"
                        "If sources lack concrete numbers, output: 'As of {today}, no sourced figures for <query> were found in retrieved sources.'\n"
                        "Otherwise list sourced figures with inline [n] citations after each cluster.\n"
                        "Begin answer with 'As of {today}, ...'\n"
                        "SOURCES:\n{sources}\n\nQuestion: {question}\nAnswer:"
                    )
                else:
                    synthesis_prompt = (
                        "You are a precise research assistant. Today is {today}.\n"
                        "Write an evidence-grounded answer ONLY using the SOURCES below. If sources are errors or empty, acknowledge lack of data.\n"
                        "Begin with 'As of {today}, ...' and include 2-4 inline [n] citations.\n"
                        "SOURCES:\n{sources}\n\nQuestion: {question}\nAnswer:"
                    )
                prompt = synthesis_prompt.format(today=today_str, sources="\n\n".join(src_text), question=original_question)
                try:
                    answer = await self.llm.apredict(prompt)  # type: ignore[attr-defined]
                except Exception:
                    answer = await asyncio.to_thread(self.llm.predict, prompt)
                md = {"used_tools":["web_search","web_scrape"],"citations":[s.get("url") for s in scraped if s.get("url")],"autobrowse":True,"forced":True,"reason":reason}
                return AgentOutput(content=str(answer).strip(), confidence_score=0.9, metadata=md)
            except Exception as e:
                logger.exception("Forced autobrowse failed: %s", e)
                return AgentOutput(content=f"Autobrowse failure: {e}", confidence_score=0.0, metadata={"error":str(e)})

        def _needs_financial_regeneration(ans: str, financial: bool) -> bool:
            if not financial:
                return False
            low = ans.lower()
            speculative_tokens = [
                "hypothetical", "possible scenario", "illustrative", "for illustrative purposes", "might ", "could ", "would ", "may ", "should not be considered", "not investment advice"
            ]
            if any(tok in low for tok in speculative_tokens):
                return True
            # Large numeric clusters without citation markers
            numbers = re.findall(r"\b\d{3,}(?:\.\d+)?", ans)
            if numbers and ("[" not in ans or not re.search(r"\[\d+\]", ans)):
                return True
            return False

        async def _regenerate_financial(answer: str, question_text: str, src_text: str, today_str: str) -> str:
            regen_prompt = (
                "You previously produced a speculative / hypothetical financial projection which is not allowed. Today is {today}.\n"
                "Rewrite STRICTLY using ONLY figures explicitly present in the SOURCES text. If SOURCES do not contain explicit numeric guidance figures (segment revenue, EPS, margins) for the query, respond exactly with: 'As of {today}, no sourced figures for the query were found in retrieved public sources.'\n"
                "Do NOT add any investment advice, disclaimers, hypothetical language, or invented numbers. Provide inline [n] citations after each numeric cluster you DO include.\n"
                "SOURCES:\n{sources}\n\nQuestion: {question}\nRewritten Answer:"
            )
            prompt = regen_prompt.format(today=today_str, sources=src_text, question=question_text)
            try:
                try:
                    return await self.llm.apredict(prompt)  # type: ignore[attr-defined]
                except Exception:
                    return await asyncio.to_thread(self.llm.predict, prompt)
            except Exception as e:
                logger.warning("Financial regeneration failed: %s", e)
                return f"As of {today_str}, no sourced figures for the query were found in retrieved public sources."

        # EARLY deterministic browse for financial projection queries or time-sensitive or URL
        if urls or is_time_sensitive or is_financial_projection:
            logger.info(
                "ResearchAgent autobrowse trigger: urls=%s is_time_sensitive=%s financial=%s question=%s",
                bool(urls), is_time_sensitive, is_financial_projection, question[:120]
            )
            try:
                sources: List[Dict[str, str]] = []
                reason = "urls provided" if urls else ("financial" if is_financial_projection else "time-sensitive")
                if urls:
                    top_urls = urls[:2]
                else:
                    try:
                        results = await search_service.ddg_search(question, max_results=3)
                        top_urls = [r["url"] for r in results[:3] if r.get("url")]
                        if not top_urls:
                            logger.warning("Autobrowse search produced no URLs; skipping autobrowse synthesis")
                            raise RuntimeError("no_search_results")
                    except Exception as se:
                        logger.exception("Search failed in autobrowse path: %s", se)
                        raise

                async def scrape(u: str) -> Dict[str, str]:
                    try:
                        text = await scraper_service.scrape_url(u, include_links=False, max_chars=1600)
                        if len(text) < 300:
                            logger.info("Re-scraping with JS rendering due to sparse content: %s", u)
                            text = await scraper_service.scrape_url(
                                u,
                                include_links=False,
                                max_chars=1600,
                                render_js=True,
                                wait_until="networkidle",
                                wait_ms=1500,
                            )
                        return {"url": u, "text": text}
                    except Exception as e:
                        logger.warning("Scrape failed for %s: %s", u, e)
                        return {"url": u, "text": f"Error scraping {u}: {e}"}

                scraped = await asyncio.gather(*(scrape(u) for u in top_urls))
                sources.extend(scraped)
                today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if is_financial_projection:
                    synthesis_prompt = (
                        "You are a precise research assistant. Today is {today}.\n"
                        "FINANCIAL QUERY: Use ONLY figures explicitly appearing in sources. DO NOT fabricate projections.\n"
                        "If no explicit numeric guidance figures for the query are present, state this clearly and do not invent scenarios.\n"
                        "Output structure: Begin 'As of {today}, ...' then either sourced figures with [n] citations OR a statement that no sourced figures were found.\n"
                        "SOURCES:\n{sources}\n\nQuestion: {question}\nAnswer:"
                    )
                else:
                    synthesis_prompt = (
                        "You are a precise research assistant. Today is {today}.\n"
                        "Write a concise, up-to-date answer using ONLY the SOURCES below.\n"
                        "- Begin with 'As of {today}, ...'\n"
                        "- Include 2-4 inline citations as [n] referring to source numbers.\n"
                        "- Provide direct URLs for citations in a short list at the end.\n"
                        "SOURCES:\n{sources}\n\nQuestion: {question}\nAnswer:"
                    )
                src_text = []
                for i, s in enumerate(sources, start=1):
                    snippet = s.get("text", "")[:1200]
                    src_text.append(f"[{i}] {s.get('url','')}\n{snippet}")
                prompt = synthesis_prompt.format(today=today_str, sources="\n\n".join(src_text), question=question)
                try:
                    answer = await self.llm.apredict(prompt)  # type: ignore[attr-defined]
                except Exception:
                    answer = await asyncio.to_thread(self.llm.predict, prompt)
                # Post-generation financial validation
                regenerated = False
                if _needs_financial_regeneration(answer, is_financial_projection):
                    logger.info("Regenerating financial answer to remove speculative content / enforce sourcing")
                    src_block = "\n\n".join(src_text)
                    answer = await _regenerate_financial(answer, question, src_block, today_str)
                    regenerated = True
                md = {
                    "used_tools": ["web_search" if not urls else "", "web_scrape"],
                    "citations": [s.get("url") for s in sources if s.get("url")],
                    "autobrowse": True,
                    "autobrowse_reason": reason,
                    "financial": is_financial_projection,
                    "financial_regenerated": regenerated,
                }
                md["used_tools"] = [t for t in md["used_tools"] if t]
                return AgentOutput(content=str(answer).strip(), confidence_score=0.92, metadata=md)
            except Exception as e:
                logger.exception("Autobrowse path failed; falling back to ReAct: %s", e)
                pass
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
            answer_str = str(answer)
            md = {"used_tools": [t.name for t in self.executor.tools], "autobrowse": False}
            if urls or is_time_sensitive:
                md["autobrowse_skipped"] = True
            # Disclaimer / hypothetical detection
            lower_ans = answer_str.lower()
            if any(p in lower_ans for p in disclaimer_phrases):
                logger.info("Disclaimer/hypothetical phrasing detected; forcing autobrowse re-run.")
                forced = await forced_autobrowse_synthesis(question, reason="disclaimer_detected", financial=is_financial_projection)
                # Merge metadata
                forced.metadata["original_answer_flagged"] = True
                return forced
            # If no citations pattern and time-sensitive, attempt forced browse
            has_real_url = "http://" in lower_ans or "https://" in lower_ans
            if is_time_sensitive and not has_real_url:
                logger.info("Time-sensitive answer lacks sources; forcing autobrowse re-run.")
                forced = await forced_autobrowse_synthesis(question, reason="missing_sources_time_sensitive", financial=is_financial_projection)
                forced.metadata["original_answer_flagged"] = True
                return forced
            # Financial projection numeric hallucination guard
            if is_financial_projection:
                # Extract large numeric tokens (ignore years) for validation
                nums = re.findall(r"\b\d{2,4}\.\d+|\b\d{3,}(?:\.\d+)?", answer_str)
                # Filter out common years
                nums_filtered = [n for n in nums if n not in ["2023","2024","2025"]]
                citation_present = any("http://" in l or "https://" in l for l in answer_str.splitlines())
                if nums_filtered and not citation_present:
                    logger.info("Financial projection numbers without citations detected; forcing autobrowse.")
                    forced = await forced_autobrowse_synthesis(question, reason="financial_numbers_without_citations", financial=True)
                    forced.metadata["original_answer_flagged"] = True
                    return forced
            return AgentOutput(content=answer_str, confidence_score=0.9, metadata=md)
        except Exception as e:
            logger.exception("ResearchAgent ReAct failure: %s", e)
            return AgentOutput(
                content=f"Research agent error: {str(e)}",
                confidence_score=0.0,
                metadata={"error": str(e)},
            )
