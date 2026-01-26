"""
API Client Wrapper for Claude/OpenAI API calls.

Provides a unified interface for making LLM API calls.
"""

from anthropic import Anthropic
from config import ANTHROPIC_API_KEY, MODEL_NAME


class LLMClient:
    """Wrapper for LLM API interactions."""
    
    def __init__(self):
        """Initialize the API client."""
        self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = MODEL_NAME
    
    def generate(self, prompt, system_prompt=None, max_tokens=4096):
        """
        Generate text using the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        pass
    
    def batch_generate(self, prompts, system_prompt=None, max_tokens=4096):
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of prompts
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of generated texts
        """
        pass
