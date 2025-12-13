"""
Comprehensive test script for LLM Planner agent and tool selection.
Tests all agent types and their expected tool selections.
"""
import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1"

# Test cases: (prompt, expected_agent, expected_tools, description)
TEST_CASES = [
    # ========== RESEARCH AGENT TESTS ==========
    {
        "prompt": "Find research papers on transformer architecture in deep learning",
        "expected_agent": "research_agent",
        "expected_tools": ["research.arxiv"],
        "fallback_tools": ["research.search"],
        "description": "Research: Academic papers (arxiv)"
    },
    {
        "prompt": "What is the history of artificial intelligence?",
        "expected_agent": "research_agent",
        "expected_tools": ["research.wikipedia", "research.search"],
        "fallback_tools": ["research.search", "research.arxiv"],
        "description": "Research: General knowledge"
    },
    {
        "prompt": "Search for recent news about quantum computing breakthroughs",
        "expected_agent": "research_agent",
        "expected_tools": ["research.search"],
        "fallback_tools": [],
        "description": "Research: News/web search"
    },
    {
        "prompt": "Summarize the key points about machine learning algorithms",
        "expected_agent": "research_agent",
        "expected_tools": ["research.summarizer", "research.search"],
        "fallback_tools": ["research.search"],
        "description": "Research: Summarization"
    },
    {
        "prompt": "What are the best practices for software testing?",
        "expected_agent": "research_agent",
        "expected_tools": ["research.search"],
        "fallback_tools": [],
        "description": "Research: Best practices inquiry"
    },
    {
        "prompt": "Find academic papers about stock market prediction using AI",
        "expected_agent": "research_agent",
        "expected_tools": ["research.arxiv"],
        "fallback_tools": ["research.search"],
        "description": "Research: Academic finance papers"
    },
    
    # ========== FINANCE AGENT TESTS ==========
    {
        "prompt": "What is the current stock price of Tesla?",
        "expected_agent": "finance_agent",
        "expected_tools": ["finance.snapshot"],
        "fallback_tools": ["finance.snapshot.alpha"],
        "description": "Finance: Stock price query"
    },
    {
        "prompt": "Give me a financial analysis of Apple Inc earnings",
        "expected_agent": "finance_agent",
        "expected_tools": ["finance.snapshot", "finance.analytics"],
        "fallback_tools": [],
        "description": "Finance: Company analysis"
    },
    {
        "prompt": "What are the latest market trends for NVIDIA stock?",
        "expected_agent": "finance_agent",
        "expected_tools": ["finance.snapshot", "finance.news"],
        "fallback_tools": [],
        "description": "Finance: Market trends"
    },
    {
        "prompt": "Compare the P/E ratios of Microsoft and Google",
        "expected_agent": "finance_agent",
        "expected_tools": ["finance.snapshot"],
        "fallback_tools": [],
        "description": "Finance: Comparative analysis"
    },
    {
        "prompt": "How is AAPL performing today?",
        "expected_agent": "finance_agent",
        "expected_tools": ["finance.snapshot"],
        "fallback_tools": [],
        "description": "Finance: Ticker symbol query"
    },
    {
        "prompt": "What's the dividend yield for Microsoft stock?",
        "expected_agent": "finance_agent",
        "expected_tools": ["finance.snapshot"],
        "fallback_tools": [],
        "description": "Finance: Dividend query"
    },
    {
        "prompt": "Should I invest in cryptocurrency right now?",
        "expected_agent": "finance_agent",
        "expected_tools": ["finance.snapshot", "finance.news"],
        "fallback_tools": [],
        "description": "Finance: Investment advice"
    },
    {
        "prompt": "What's the stock outlook for AMD?",
        "expected_agent": "finance_agent",
        "expected_tools": ["finance.snapshot"],
        "fallback_tools": [],
        "description": "Finance: Stock outlook"
    },
    
    # ========== CREATIVE AGENT TESTS ==========
    {
        "prompt": "Write a short poem about the sunset",
        "expected_agent": "creative_agent",
        "expected_tools": ["creative.tonecheck"],
        "fallback_tools": [],
        "description": "Creative: Poetry writing"
    },
    {
        "prompt": "Help me brainstorm ideas for a mobile app startup",
        "expected_agent": "creative_agent",
        "expected_tools": ["creative.tonecheck"],
        "fallback_tools": [],
        "description": "Creative: Brainstorming"
    },
    {
        "prompt": "Create a marketing slogan for an eco-friendly water bottle",
        "expected_agent": "creative_agent",
        "expected_tools": ["creative.tonecheck"],
        "fallback_tools": [],
        "description": "Creative: Marketing content"
    },
    {
        "prompt": "Write a short story about a robot learning to feel emotions",
        "expected_agent": "creative_agent",
        "expected_tools": ["creative.tonecheck"],
        "fallback_tools": [],
        "description": "Creative: Story writing"
    },
    {
        "prompt": "Compose lyrics for a song about friendship",
        "expected_agent": "creative_agent",
        "expected_tools": ["creative.tonecheck"],
        "fallback_tools": [],
        "description": "Creative: Song lyrics"
    },
    {
        "prompt": "Write a catchy tagline for a fitness app",
        "expected_agent": "creative_agent",
        "expected_tools": ["creative.tonecheck"],
        "fallback_tools": [],
        "description": "Creative: Tagline creation"
    },
    {
        "prompt": "Draft a creative product description for wireless headphones",
        "expected_agent": "creative_agent",
        "expected_tools": ["creative.tonecheck"],
        "fallback_tools": [],
        "description": "Creative: Product description"
    },
    
    # ========== GENERAL AGENT TESTS ==========
    {
        "prompt": "Hello, how are you today?",
        "expected_agent": "general_agent",
        "expected_tools": [],
        "fallback_tools": [],
        "description": "General: Greeting"
    },
    {
        "prompt": "What can you help me with?",
        "expected_agent": "general_agent",
        "expected_tools": [],
        "fallback_tools": [],
        "description": "General: Capabilities inquiry"
    },
    {
        "prompt": "Explain how to calculate compound interest",
        "expected_agent": "general_agent",
        "expected_tools": [],
        "fallback_tools": [],
        "description": "General: Educational explanation"
    },
    {
        "prompt": "Convert 100 miles to kilometers",
        "expected_agent": "general_agent",
        "expected_tools": [],
        "fallback_tools": [],
        "description": "General: Unit conversion"
    },
    {
        "prompt": "What's the capital of France?",
        "expected_agent": "general_agent",
        "expected_tools": [],
        "fallback_tools": [],
        "description": "General: Simple factual question"
    },
    {
        "prompt": "Thank you for your help!",
        "expected_agent": "general_agent",
        "expected_tools": [],
        "fallback_tools": [],
        "description": "General: Gratitude"
    },
    
    # ========== ENTERPRISE AGENT TESTS ==========
    {
        "prompt": "Generate a quarterly business report for our sales data",
        "expected_agent": "enterprise_agent",
        "expected_tools": ["enterprise.playbook", "enterprise.report"],
        "fallback_tools": ["enterprise.playbook"],
        "description": "Enterprise: Business reporting"
    },
    {
        "prompt": "Analyze our company's CRM data for customer trends",
        "expected_agent": "enterprise_agent",
        "expected_tools": ["enterprise.playbook", "enterprise.analytics"],
        "fallback_tools": ["enterprise.playbook"],
        "description": "Enterprise: CRM data analytics"
    },
    {
        "prompt": "Help me draft a professional business proposal",
        "expected_agent": "enterprise_agent",
        "expected_tools": ["enterprise.playbook"],
        "fallback_tools": [],
        "description": "Enterprise: Business proposal"
    },
    {
        "prompt": "Create a go-to-market strategy for our new product",
        "expected_agent": "enterprise_agent",
        "expected_tools": ["enterprise.playbook"],
        "fallback_tools": [],
        "description": "Enterprise: GTM strategy"
    },
    {
        "prompt": "Build an executive summary for the board meeting",
        "expected_agent": "enterprise_agent",
        "expected_tools": ["enterprise.playbook"],
        "fallback_tools": [],
        "description": "Enterprise: Executive summary"
    },
    {
        "prompt": "Help me create a pitch deck for investors",
        "expected_agent": "enterprise_agent",
        "expected_tools": ["enterprise.playbook"],
        "fallback_tools": [],
        "description": "Enterprise: Pitch deck"
    },
    
    # ========== MIXED INTENT TESTS ==========
    {
        "prompt": "Research Tesla stock performance and write a summary report",
        "expected_agent": "finance_agent",
        "expected_tools": ["finance.snapshot"],
        "fallback_tools": ["research.search"],
        "description": "Mixed: Finance + Research (finance primary)"
    },
    {
        "prompt": "Write a creative blog post about investing in tech stocks",
        "expected_agent": "creative_agent",
        "expected_tools": ["creative.tonecheck"],
        "fallback_tools": [],
        "description": "Mixed: Creative + Finance (creative primary)"
    },
    {
        "prompt": "Analyze the latest AI trends and summarize key findings",
        "expected_agent": "research_agent",
        "expected_tools": ["research.search", "research.summarizer"],
        "fallback_tools": [],
        "description": "Mixed: Research + Summarize"
    },
    {
        "prompt": "Draft a creative company newsletter about our Q3 results",
        "expected_agent": "creative_agent",
        "expected_tools": ["creative.tonecheck"],
        "fallback_tools": [],
        "description": "Mixed: Creative + Enterprise"
    },
    
    # ========== EDGE CASES ==========
    {
        "prompt": "I need help with something",
        "expected_agent": "general_agent",
        "expected_tools": [],
        "fallback_tools": [],
        "description": "Edge: Vague request"
    },
    {
        "prompt": "Tell me about NVIDIA's new GPU",
        "expected_agent": "research_agent",
        "expected_tools": ["research.search"],
        "fallback_tools": [],
        "description": "Edge: Product info (not finance)"
    },
    {
        "prompt": "How do I write better code?",
        "expected_agent": "general_agent",
        "expected_tools": [],
        "fallback_tools": ["research.search"],
        "description": "Edge: Self-improvement question"
    },
]


def submit_task(prompt: str, max_retries: int = 3) -> dict:
    """Submit a task and return the task_id, with rate-limit retry"""
    body = {
        "prompt": prompt,
        "session_id": f"test-routing-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    }
    for attempt in range(max_retries):
        response = requests.post(f"{BASE_URL}/submit_task", json=body)
        if response.status_code == 429:  # Rate limited
            retry_after = int(response.headers.get("retry-after", 7))
            print(f"    Rate limited, waiting {retry_after}s...")
            time.sleep(retry_after + 1)
            continue
        return response.json()
    return {"error": "Rate limit exceeded after retries"}


def get_task_result(task_id: str, max_wait: int = 120) -> dict:
    """Poll for task completion"""
    start = time.time()
    while time.time() - start < max_wait:
        response = requests.get(f"{BASE_URL}/tasks/{task_id}")
        result = response.json()
        if result.get("status") in ["completed", "failed"]:
            return result
        time.sleep(2)
    return {"status": "timeout", "task_id": task_id}


def extract_routing_info(result: dict) -> dict:
    """Extract agent and tool selection from result"""
    info = {
        "status": result.get("status"),
        "selected_agents": [],
        "skipped_agents": [],
        "planned_tools": [],
        "fallback_tools": [],
        "executed_tools": [],
        "planner_reason": "",
        "raw_response": ""
    }
    
    for event in result.get("events", []):
        if event.get("event_type") == "routing_decided":
            payload = event.get("payload", {})
            info["selected_agents"] = payload.get("selected_agents", [])
            info["skipped_agents"] = payload.get("skipped_agents", [])
            
            metadata = payload.get("metadata", {}).get("planner", {})
            info["raw_response"] = metadata.get("raw_response", "")
            
            steps = metadata.get("steps", [])
            if steps:
                step = steps[0]
                info["planned_tools"] = step.get("tools", [])
                info["fallback_tools"] = step.get("fallback_tools", [])
                info["planner_reason"] = step.get("reason", "")
        
        if event.get("event_type") == "agent_completed":
            payload = event.get("payload", {}).get("output", {}).get("metadata", {})
            tools_used = payload.get("tools_used", [])
            info["executed_tools"] = [t.get("tool") for t in tools_used]
    
    return info


def evaluate_result(test_case: dict, routing_info: dict) -> dict:
    """Evaluate if the routing matches expectations"""
    evaluation = {
        "agent_correct": False,
        "tools_correct": False,
        "issues": []
    }
    
    # Check agent selection
    selected = routing_info.get("selected_agents", [])
    expected_agent = test_case["expected_agent"]
    
    if expected_agent in selected:
        evaluation["agent_correct"] = True
        if len(selected) > 1 and expected_agent != selected[0]:
            evaluation["issues"].append(f"Expected {expected_agent} as primary, got {selected[0]}")
    else:
        evaluation["issues"].append(f"Expected {expected_agent}, got {selected}")
    
    # Check tool selection
    planned_tools = routing_info.get("planned_tools", [])
    expected_tools = test_case["expected_tools"]
    
    if not expected_tools:
        evaluation["tools_correct"] = True  # No specific tool expected
    elif any(t in planned_tools for t in expected_tools):
        evaluation["tools_correct"] = True
    else:
        # Check if fallback was reasonably used
        fallback_tools = test_case.get("fallback_tools", [])
        if fallback_tools and any(t in planned_tools for t in fallback_tools):
            evaluation["tools_correct"] = True
            evaluation["issues"].append(f"Used fallback tool instead of primary")
        else:
            evaluation["issues"].append(f"Expected tools {expected_tools}, got {planned_tools}")
    
    return evaluation


def run_tests():
    """Run all test cases and collect results"""
    print("=" * 80)
    print("NEURAFORGE PLANNER ROUTING TEST SUITE")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)
    print()
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] {test_case['description']}")
        print(f"    Prompt: {test_case['prompt'][:60]}...")
        print(f"    Expected: {test_case['expected_agent']} -> {test_case['expected_tools']}")
        
        try:
            # Submit task
            submit_result = submit_task(test_case["prompt"])
            task_id = submit_result.get("task_id")
            
            if not task_id:
                print(f"    ❌ FAILED: Could not submit task")
                results.append({
                    "test": test_case,
                    "status": "submit_failed",
                    "routing_info": {},
                    "evaluation": {"agent_correct": False, "tools_correct": False}
                })
                continue
            
            print(f"    Task ID: {task_id}")
            
            # Wait for result
            task_result = get_task_result(task_id)
            routing_info = extract_routing_info(task_result)
            evaluation = evaluate_result(test_case, routing_info)
            
            # Print result
            agent_status = "✅" if evaluation["agent_correct"] else "❌"
            tools_status = "✅" if evaluation["tools_correct"] else "❌"
            
            print(f"    Agent: {agent_status} {routing_info.get('selected_agents', [])}")
            print(f"    Tools: {tools_status} {routing_info.get('planned_tools', [])}")
            print(f"    Reason: {routing_info.get('planner_reason', 'N/A')}")
            
            if evaluation["issues"]:
                print(f"    Issues: {evaluation['issues']}")
            
            results.append({
                "test": test_case,
                "status": task_result.get("status"),
                "routing_info": routing_info,
                "evaluation": evaluation
            })
            
        except Exception as e:
            print(f"    ❌ ERROR: {str(e)}")
            results.append({
                "test": test_case,
                "status": "error",
                "error": str(e),
                "routing_info": {},
                "evaluation": {"agent_correct": False, "tools_correct": False, "issues": [str(e)]}
            })
        
        # Delay between tests to avoid rate limiting (backend requires 6+ seconds)
        time.sleep(7)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total = len(results)
    agent_correct = sum(1 for r in results if r["evaluation"]["agent_correct"])
    tools_correct = sum(1 for r in results if r["evaluation"]["tools_correct"])
    fully_correct = sum(1 for r in results if r["evaluation"]["agent_correct"] and r["evaluation"]["tools_correct"])
    
    print(f"\nTotal Tests: {total}")
    print(f"Agent Selection Correct: {agent_correct}/{total} ({100*agent_correct/total:.1f}%)")
    print(f"Tool Selection Correct:  {tools_correct}/{total} ({100*tools_correct/total:.1f}%)")
    print(f"Fully Correct:           {fully_correct}/{total} ({100*fully_correct/total:.1f}%)")
    
    # Group by agent
    print("\n--- Results by Agent Type ---")
    agents = ["research_agent", "finance_agent", "creative_agent", "general_agent", "enterprise_agent"]
    for agent in agents:
        agent_results = [r for r in results if r["test"]["expected_agent"] == agent]
        if agent_results:
            correct = sum(1 for r in agent_results if r["evaluation"]["agent_correct"])
            print(f"  {agent}: {correct}/{len(agent_results)} correct")
    
    # List failures
    failures = [r for r in results if not r["evaluation"]["agent_correct"] or not r["evaluation"]["tools_correct"]]
    if failures:
        print("\n--- Failed/Partial Tests ---")
        for r in failures:
            print(f"  • {r['test']['description']}")
            print(f"    Expected: {r['test']['expected_agent']} -> {r['test']['expected_tools']}")
            print(f"    Got: {r['routing_info'].get('selected_agents', [])} -> {r['routing_info'].get('planned_tools', [])}")
            if r["evaluation"]["issues"]:
                print(f"    Issues: {r['evaluation']['issues']}")
    
    print("\n" + "=" * 80)
    print(f"Test completed: {datetime.now().isoformat()}")
    print("=" * 80)
    
    # Save detailed results to file
    with open("test_routing_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nDetailed results saved to: test_routing_results.json")
    
    return results


if __name__ == "__main__":
    run_tests()
