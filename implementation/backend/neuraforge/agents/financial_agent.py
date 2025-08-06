"""Financial Agent for NeuraForge.

This module implements a specialized agent for handling financial tasks,
including financial analysis, planning, forecasting, and investment guidance.
"""

import logging
import re
from typing import Dict, List, Optional, Any, Union

from langchain.llms.base import BaseLLM
from langchain.callbacks.base import BaseCallbackHandler

from .base_agent import BaseAgent, AgentInput, AgentOutput

logger = logging.getLogger(__name__)


class FinancialCalculator:
    """Financial calculator tool for performing common financial calculations."""
    
    @staticmethod
    def calculate_compound_interest(principal: float, rate: float, time: float, compounds_per_year: int = 1) -> float:
        """
        Calculate compound interest.
        
        Args:
            principal: Principal amount
            rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
            time: Time in years
            compounds_per_year: Number of times interest is compounded per year
            
        Returns:
            float: The future value
        """
        return principal * (1 + rate/compounds_per_year)**(compounds_per_year*time)
    
    @staticmethod
    def calculate_loan_payment(principal: float, rate: float, time: float) -> float:
        """
        Calculate monthly loan payment.
        
        Args:
            principal: Loan amount
            rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
            time: Time in years
            
        Returns:
            float: Monthly payment amount
        """
        monthly_rate = rate / 12
        num_payments = time * 12
        return principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    
    @staticmethod
    def calculate_retirement_savings(monthly_contribution: float, rate: float, time: float) -> float:
        """
        Calculate retirement savings.
        
        Args:
            monthly_contribution: Monthly contribution amount
            rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
            time: Time in years
            
        Returns:
            float: The future value of the retirement account
        """
        monthly_rate = rate / 12
        num_months = time * 12
        return monthly_contribution * ((1 + monthly_rate)**num_months - 1) / monthly_rate
    
    @staticmethod
    def calculate_mortgage_affordability(income: float, debt: float, interest_rate: float, term: float) -> float:
        """
        Calculate mortgage affordability.
        
        Args:
            income: Annual income
            debt: Monthly debt payments
            interest_rate: Annual mortgage interest rate (as decimal)
            term: Mortgage term in years
            
        Returns:
            float: Affordable mortgage amount
        """
        # Use 28% of income for mortgage (conventional rule)
        monthly_income = income / 12
        max_piti = monthly_income * 0.28
        # Subtract estimated taxes and insurance (approx 25% of payment)
        max_pi = max_piti * 0.75
        # Account for existing debt (total DTI should be under 36%)
        if debt + max_pi > monthly_income * 0.36:
            max_pi = (monthly_income * 0.36) - debt
        
        # Calculate affordable mortgage amount
        monthly_rate = interest_rate / 12
        num_payments = term * 12
        return max_pi * ((1 + monthly_rate)**num_payments - 1) / (monthly_rate * (1 + monthly_rate)**num_payments)


class FinancialAgent(BaseAgent):
    """Specialized agent for financial analysis and planning tasks."""
    
    def __init__(
        self, 
        agent_id: str,
        llm: BaseLLM,
        callbacks: List[BaseCallbackHandler] = None
    ):
        """Initialize the financial agent.
        
        Args:
            agent_id: Unique identifier for the agent
            llm: The language model to use
            callbacks: Optional list of callback handlers
        """
        # Financial agent-specific prompt template
        prompt_template = """
        You are a Financial Agent specializing in financial analysis, planning, and investment guidance.
        Your goal is to provide clear, accurate financial insights and recommendations.
        
        You excel at:
        - Analyzing financial data and trends
        - Providing budgeting and financial planning advice
        - Explaining financial concepts in simple terms
        - Offering general investment strategies
        - Discussing economic concepts and market dynamics
        
        Important: You are not a licensed financial advisor. Always clarify that your guidance is 
        educational and users should consult with licensed professionals for personalized advice.
        
        Chat history:
        {chat_history}
        
        Human: {input}
        Financial Agent:
        """
        
        super().__init__(
            agent_id=agent_id,
            agent_name="Financial Agent",
            agent_type="financial",
            llm=llm,
            prompt_template=prompt_template,
            callbacks=callbacks
        )
        
        # Initialize financial calculator tool
        self.calculator = FinancialCalculator()
        
        # Track tool usage
        self.tools_used = []
        
        # Define financial keywords for confidence boosting and tool selection
        self.financial_keywords = [
            "budget", "invest", "financial", "money", "savings", 
            "debt", "loan", "mortgage", "retirement", "tax", 
            "interest", "stock", "bond", "mutual fund", "etf",
            "roth", "ira", "401k", "credit", "insurance"
        ]
        
        logger.info(f"Financial Agent initialized with ID: {agent_id}")
    
    async def process(self, agent_input: AgentInput) -> AgentOutput:
        """Process the input and generate a financial-focused response.
        
        Args:
            agent_input: The input to process
            
        Returns:
            AgentOutput: The financial agent's response
        """
        # Extract the latest user message
        user_messages = [msg["content"] for msg in agent_input.messages if msg["role"] == "user"]
        if not user_messages:
            return AgentOutput(
                content="I don't see any financial questions. How can I help you with financial analysis or planning?",
                agent_info="Financial Agent: Specialized in financial analysis and planning",
                confidence_score=0.9
            )
        
        latest_user_message = user_messages[-1]
        
        # Reset tools used for this request
        self.tools_used = []
        
        # Check for calculation requests
        calculation_result = self._handle_calculation_request(latest_user_message)
        if calculation_result:
            return calculation_result
        
        # Format chat history for context
        chat_history = self._format_chat_history(agent_input.messages[:-1])  # Exclude the latest message
        
        # Special handling for financial projections
        if any(term in latest_user_message.lower() for term in ["financial projection", "financial forecast", "financial outlook", "revenue projection"]):
            return await self._handle_financial_projections(latest_user_message, chat_history)
        
        # Process with LLM
        response = await self.chain.arun(
            input=latest_user_message,
            chat_history=chat_history
        )
        
        # Add disclaimer to responses
        disclaimer = "\n\nNote: This information is for educational purposes only. Please consult with a licensed financial professional for personalized advice."
        if len(response) + len(disclaimer) < 8000:  # Avoid excessively long responses
            response += disclaimer
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(response)
        
        # Special adjustment for financial agent confidence
        # Financial agents should be more confident on financial questions
        if any(keyword in latest_user_message.lower() for keyword in self.financial_keywords):
            confidence_score = min(0.95, confidence_score + 0.1)
        
        return AgentOutput(
            content=response,
            agent_info="Financial Agent: Specialized in financial analysis and planning",
            confidence_score=confidence_score,
            metadata={
                "agent_type": "financial",
                "query_type": "advisory",
                "tools_used": self.tools_used
            }
        )
    
    def _handle_calculation_request(self, user_message: str) -> Optional[AgentOutput]:
        """Handle financial calculation requests.
        
        Args:
            user_message: The user's message
            
        Returns:
            Optional[AgentOutput]: The calculation result or None if no calculation needed
        """
        # Check for financial projections - this is a specialized request that needs LLM handling
        if "financial projection" in user_message.lower() or "financial forecast" in user_message.lower():
            self.tools_used.append("financial_projections")
            return None  # Let the LLM handle this complex request
            
        # Check for compound interest calculation
        compound_interest_pattern = r"(?:calculate\s+)?(?:compound\s+interest|interest).*?(\$?[\d,]+\.?\d*|\d+\.?\d*\s*(?:dollars|USD)?).*?(\d+\.?\d*)%.*?(\d+\.?\d*)\s*years?.*?(?:(\d+)?\s*(?:compounds?(?:\s*per\s*year)?)?|annually|monthly|quarterly|daily)?"
        compound_match = re.search(compound_interest_pattern, user_message, re.IGNORECASE)
        
        if compound_match:
            # Extract values from either the first or second capture group
            groups = compound_match.groups()
            principal_str = groups[0] if groups[0] else groups[4]
            rate_str = groups[1] if groups[1] else groups[5]
            time_str = groups[2] if groups[2] else groups[6]
            compounds_str = groups[3] if groups[3] else groups[7]
            
            if not principal_str or not rate_str or not time_str:
                return None  # Couldn't extract all required values
            
            # Clean and convert values
            principal = float(principal_str.replace('$', '').replace(',', ''))
            rate = float(rate_str) / 100  # Convert percentage to decimal
            time = float(time_str)
            compounds = int(compounds_str) if compounds_str else 1
            
            # Calculate
            result = self.calculator.calculate_compound_interest(principal, rate, time, compounds)
            self.tools_used.append("compound_interest_calculator")
            
            # Format response
            response = f"""
I've calculated the compound interest for you:

Principal: ${principal:,.2f}
Annual Interest Rate: {rate*100:.2f}%
Time Period: {time} years
Compounds Per Year: {compounds}

Future Value: ${result:,.2f}

This means your investment of ${principal:,.2f} will grow to ${result:,.2f} after {time} years at {rate*100:.2f}% interest compounded {compounds} time(s) per year.

Note: This information is for educational purposes only. Please consult with a licensed financial professional for personalized advice.
"""
            
            return AgentOutput(
                content=response,
                agent_info="Financial Calculator: Compound Interest Calculation",
                confidence_score=0.98,
                metadata={
                    "agent_type": "financial",
                    "query_type": "calculation",
                    "tools_used": self.tools_used
                }
            )
        
        # Check for loan payment calculation
        loan_pattern = r"(?:calculate|what\s+(?:is|are|would\s+be))?\s*(?:monthly)?\s*(?:loan|mortgage)?\s*payments?\s*(?:on|for)?\s*(?:a|loan|mortgage)?\s*(?:of)?\s*(\$?[\d,]+\.?\d*).*?(\d+\.?\d*)%.*?(\d+\.?\d*)\s*years?"
        loan_match = re.search(loan_pattern, user_message, re.IGNORECASE)
        
        if loan_match:
            # Extract values
            principal_str, rate_str, time_str = loan_match.groups()
            
            if not principal_str or not rate_str or not time_str:
                return None  # Couldn't extract all required values
            
            # Clean and convert values
            principal = float(principal_str.replace('$', '').replace(',', ''))
            rate = float(rate_str) / 100  # Convert percentage to decimal
            time = float(time_str)
            
            # Calculate
            result = self.calculator.calculate_loan_payment(principal, rate, time)
            self.tools_used.append("loan_payment_calculator")
            
            # Format response
            total_paid = result * time * 12
            interest_paid = total_paid - principal
            
            response = f"""
I've calculated the loan payment for you:

Loan Amount: ${principal:,.2f}
Annual Interest Rate: {rate*100:.2f}%
Loan Term: {time} years

Monthly Payment: ${result:,.2f}

Over the life of the loan:
Total Payments: ${total_paid:,.2f}
Total Interest: ${interest_paid:,.2f}

Note: This information is for educational purposes only. Please consult with a licensed financial professional for personalized advice.
"""
            
            return AgentOutput(
                content=response,
                agent_info="Financial Calculator: Loan Payment Calculation",
                confidence_score=0.98,
                metadata={
                    "agent_type": "financial",
                    "query_type": "calculation",
                    "tools_used": self.tools_used
                }
            )
        
        # Check for retirement savings calculation
        retirement_pattern = r"(?:calculate|how\s+much|what).*?(?:retirement|save).*?(?:contributing|invest|save).*?(\$?[\d,]+\.?\d*).*?(?:month|monthly).*?(\d+\.?\d*)%.*?(\d+\.?\d*)\s*years?"
        retirement_match = re.search(retirement_pattern, user_message, re.IGNORECASE)
        
        if retirement_match:
            # Extract values
            contribution_str, rate_str, time_str = retirement_match.groups()
            
            if not contribution_str or not rate_str or not time_str:
                return None  # Couldn't extract all required values
            
            # Clean and convert values
            contribution = float(contribution_str.replace('$', '').replace(',', ''))
            rate = float(rate_str) / 100  # Convert percentage to decimal
            time = float(time_str)
            
            # Calculate
            result = self.calculator.calculate_retirement_savings(contribution, rate, time)
            self.tools_used.append("retirement_calculator")
            
            # Format response
            total_contributions = contribution * time * 12
            interest_earned = result - total_contributions
            
            response = f"""
I've calculated your retirement savings projection:

Monthly Contribution: ${contribution:,.2f}
Annual Interest Rate: {rate*100:.2f}%
Investment Period: {time} years

Projected Future Value: ${result:,.2f}

Breakdown:
Total Contributions: ${total_contributions:,.2f}
Interest Earned: ${interest_earned:,.2f}

This projection assumes consistent monthly contributions of ${contribution:,.2f} and a steady annual return of {rate*100:.2f}% over {time} years.

Note: This information is for educational purposes only. Please consult with a licensed financial professional for personalized advice.
"""
            
            return AgentOutput(
                content=response,
                agent_info="Financial Calculator: Retirement Savings Projection",
                confidence_score=0.98,
                metadata={
                    "agent_type": "financial",
                    "query_type": "calculation",
                    "tools_used": self.tools_used
                }
            )
        
        # No calculation needed
        return None
    
    async def _handle_financial_projections(self, user_message: str, chat_history: str) -> AgentOutput:
        """Handle financial projection requests with specialized prompt.
        
        Args:
            user_message: The user's message
            chat_history: The conversation history
            
        Returns:
            AgentOutput: The financial projection response
        """
        # Special prompt template for financial projections
        projection_prompt = f"""
        You are a Financial Agent specializing in financial analysis, forecasting, and projections.
        
        TASK: Create a detailed financial projection based on the following request:
        "{user_message}"
        
        Previous conversation context:
        {chat_history}
        
        Follow these guidelines:
        1. Create realistic financial projections that include revenue, expenses, and profit
        2. Use current economic conditions from 2025 as baseline assumptions
        3. Provide quarter-by-quarter breakdown of key financial metrics
        4. Include growth rates and assumptions used in your model
        5. Highlight key financial risks and opportunities
        6. Format the projection in a clear, tabular format when appropriate
        
        IMPORTANT: This is a financial projection for educational purposes only. Always clarify that this is not financial advice.
        """
        
        # Track tool usage
        self.tools_used.append("financial_projections_generator")
        
        # Process with LLM using specialized prompt
        try:
            response = await self.llm.agenerate(prompts=[projection_prompt])
            projection_text = response.generations[0][0].text
            
            # Add disclaimer
            disclaimer = "\n\nNote: This financial projection is for educational purposes only. These figures are estimates based on general assumptions and should not be used for actual financial decisions. Please consult with a licensed financial professional for personalized advice."
            if len(projection_text) + len(disclaimer) < 8000:
                projection_text += disclaimer
            
            return AgentOutput(
                content=projection_text,
                agent_info="Financial Agent: Specialized in financial projections and forecasting",
                confidence_score=0.95,
                metadata={
                    "agent_type": "financial",
                    "query_type": "projection",
                    "tools_used": self.tools_used
                }
            )
        except Exception as e:
            logger.error(f"Error generating financial projection: {str(e)}", exc_info=True)
            error_message = f"I apologize, but I encountered an error while creating the financial projection. The error was: {str(e)}. Please try a more specific request or contact support."
            
            return AgentOutput(
                content=error_message,
                agent_info="Financial Agent: Error in financial projections",
                confidence_score=0.5,
                metadata={
                    "agent_type": "financial",
                    "query_type": "projection",
                    "tools_used": self.tools_used,
                    "error": str(e)
                }
            )
