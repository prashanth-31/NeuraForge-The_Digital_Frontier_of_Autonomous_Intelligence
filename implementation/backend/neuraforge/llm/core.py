"""LLM Integration Layer for NeuraForge.

This module provides the core LLM integration functionality using LangChain.
"""

import os
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
# Use imports compatible with langchain 0.1.0, langchain-core 0.1.23, and langchain-community 0.0.10
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
# Try to import from the newer location first, then fall back to the older location if needed
try:
    from langchain_core.exceptions import OutputParserException
except ImportError:
    # Fall back to the older location
    from langchain_core.output_parsers import OutputParserException

# Try to use the newer OllamaLLM from langchain_ollama if available
try:
    from langchain_ollama import OllamaLLM as LLMClass
    USE_NEW_OLLAMA = True
except ImportError:
    # Fall back to the older Ollama from langchain_community
    from langchain_community.llms import Ollama as LLMClass
    USE_NEW_OLLAMA = False

from langchain.chains import LLMChain
from pydantic import BaseModel, Field

load_dotenv()


def get_llm_model(model_name=None, temperature=0.7, streaming=True):
    """Get an instance of the LLM model for use in agents and other components.
    
    Args:
        model_name: Name of the model to use, defaults to the LLM_MODEL env var or 'llama3.1:8b'
        temperature: Temperature setting for the model (0-1)
        streaming: Whether to enable streaming output
        
    Returns:
        An instance of the LLM ready for use with LangChain
    """
    config = LLMConfig(
        model_name=model_name or os.getenv("LLM_MODEL", "llama3.1:8b"),
        temperature=temperature,
        streaming=streaming
    )
    
    # Use the direct LLM instance rather than the NeuraForgeLLM wrapper
    # for better compatibility with agent frameworks
    if USE_NEW_OLLAMA:
        # Use the newer OllamaLLM from langchain_ollama
        llm = LLMClass(
            model=config.model_name,
            temperature=config.temperature,
            top_p=config.top_p, 
            streaming=config.streaming,
            base_url=config.base_url,
        )
    else:
        # Use the older Ollama from langchain_community with compatible parameters
        llm = LLMClass(
            model=config.model_name,
            temperature=config.temperature,
            top_p=config.top_p,
            base_url=config.base_url,
        )
    
    return llm


class LLMConfig(BaseModel):
    """Configuration for LLM."""

    model_name: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.1:8b"))
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    streaming: bool = Field(default=True)
    base_url: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )


class NeuraForgeLLM:
    """Core LLM integration for NeuraForge.
    
    This class provides a unified interface to interact with the underlying LLM,
    handling prompt templates, streaming, and response parsing.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        callbacks: Optional[List[BaseCallbackHandler]] = None,
    ):
        """Initialize the NeuraForge LLM integration."""
        self.config = config or LLMConfig()
        
        # Set default callbacks for streaming if none provided
        if callbacks is None and self.config.streaming:
            callbacks = [StreamingStdOutCallbackHandler()]
            
        # Initialize the LLM based on which implementation is available
        if USE_NEW_OLLAMA:
            # Use the newer OllamaLLM from langchain_ollama
            self.llm = LLMClass(
                model=self.config.model_name,
                temperature=self.config.temperature,
                top_p=self.config.top_p, 
                streaming=self.config.streaming,
                base_url=self.config.base_url,
                callbacks=callbacks,
            )
        else:
            # Use the older Ollama from langchain_community with compatible parameters
            # Remove streaming if it causes validation errors
            self.llm = LLMClass(
                model=self.config.model_name,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                base_url=self.config.base_url,
                callbacks=callbacks,
            )

    def create_prompt_template(
        self, template: str, input_variables: List[str]
    ) -> PromptTemplate:
        """Create a prompt template for the LLM."""
        return PromptTemplate(template=template, input_variables=input_variables)

    def create_chain(
        self, 
        prompt_template: PromptTemplate, 
        output_parser: Optional[BaseOutputParser] = None
    ) -> LLMChain:
        """Create an LLM chain with the given prompt template."""
        return LLMChain(
            llm=self.llm,
            prompt=prompt_template,
            output_parser=output_parser,
        )

    def generate(
        self, prompt: Union[str, PromptTemplate], variables: Optional[Dict] = None
    ) -> str:
        """Generate text from the LLM using a prompt."""
        if isinstance(prompt, str):
            return self.llm.invoke(prompt)
        
        if variables is None:
            variables = {}
            
        chain = self.create_chain(prompt)
        return chain.invoke(variables)
