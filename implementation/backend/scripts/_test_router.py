from app.orchestration.routing import HeuristicAgentRouter
from app.agents.research import ResearchAgent
from app.agents.finance import FinanceAgent
from app.agents.creative import CreativeAgent
from app.agents.enterprise import EnterpriseAgent
import asyncio

async def main():
    router = HeuristicAgentRouter()
    agents = [ResearchAgent(), FinanceAgent(), CreativeAgent(), EnterpriseAgent()]
    decision = await router.select(task={'prompt':'Hi'}, agents=agents)
    print('names=', decision.names)
    print('reason=', decision.reason)

asyncio.run(main())
