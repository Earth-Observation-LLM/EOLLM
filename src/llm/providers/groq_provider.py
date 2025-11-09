"""Groq API provider implementation.

This module implements the LLMProvider interface for Groq's cloud API,
which provides access to vision-capable Llama models.
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple

from groq import Groq

from ..base import LLMProvider
from ..registry import register_provider


@register_provider('groq')
class GroqProvider(LLMProvider):
    """Groq API provider for vision-capable LLM inference.
    
    This provider uses Groq's cloud API which supports multimodal inputs
    (text + images) and provides fast inference with Llama models.
    
    The provider handles:
    - API authentication via environment variables
    - Message formatting in OpenAI-compatible format
    - Automatic retry with exponential backoff (built into Groq client)
    - Token usage tracking from response metadata
    
    Attributes:
        client: Groq API client instance
        model: Model name to use for inference
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Groq provider.
        
        Args:
            config: Configuration dictionary or Pydantic model with:
                - model: Model name (e.g., 'meta-llama/llama-4-maverick-17b-128e-instruct')
                - api_key_env: Name of environment variable containing API key
                
        Raises:
            ValueError: If API key environment variable is not set
            RuntimeError: If Groq client initialization fails
        """
        super().__init__(config)
        
        # Get API key from environment
        api_key_env = self._get_api_key_env_name(config)
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValueError(
                f"Groq API key not found. Please set the {api_key_env} "
                f"environment variable."
            )
        
        # Initialize Groq client
        try:
            self.client = Groq(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq client: {e}")
    
    def _get_api_key_env_name(self, config: Dict[str, Any]) -> str:
        """Extract API key environment variable name from config.
        
        Args:
            config: Configuration dictionary or object
            
        Returns:
            Environment variable name
        """
        if hasattr(config, 'api_key_env'):
            return config.api_key_env
        elif isinstance(config, dict):
            return config.get('api_key_env', 'GROQ_API_KEY')
        return 'GROQ_API_KEY'
    
    def send_message(self, messages: List[Dict]) -> Tuple[str, Dict[str, Any]]:
        """Send messages to Groq API and return response with statistics.
        
        This method sends messages to Groq's chat completion API using
        the OpenAI-compatible format. It supports multimodal inputs
        (text + base64-encoded images).
        
        Args:
            messages: List of message dictionaries in OpenAI format
            
        Returns:
            Tuple of (response_text, stats_dict) where stats includes:
            - model: Model name used
            - latency_ms: Request latency in milliseconds
            - tokens_used: Total tokens (prompt + completion)
            - timestamp: ISO timestamp of the call
            
        Raises:
            RuntimeError: If API call fails
        """
        # Validate messages
        self.validate_messages(messages)
        
        # Record start time
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # Make API call with default settings (no temperature/max_tokens override)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=False
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Extract token usage
            tokens_used = -1
            if hasattr(response, 'usage') and response.usage:
                tokens_used = (
                    response.usage.prompt_tokens + 
                    response.usage.completion_tokens
                )
            
            # Build statistics dictionary
            stats = {
                'model': self.model,
                'latency_ms': latency_ms,
                'tokens_used': tokens_used,
                'timestamp': timestamp,
                'success': True,
                'error': None
            }
            
            # Add tokens per second if available
            if tokens_used > 0:
                stats['tokens_per_second'] = tokens_used / (latency_ms / 1000)
            
            return response_text, stats
            
        except Exception as e:
            # Calculate latency even for failed requests
            latency_ms = (time.time() - start_time) * 1000
            
            # Build error statistics
            stats = {
                'model': self.model,
                'latency_ms': latency_ms,
                'tokens_used': -1,
                'timestamp': timestamp,
                'success': False,
                'error': str(e)
            }
            
            raise RuntimeError(f"Groq API call failed: {e}") from e
    
    def supports_vision(self) -> bool:
        """Check if this provider supports vision inputs.
        
        Returns:
            True (Groq's Llama models support vision)
        """
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'provider': 'groq',
            'model': self.model,
            'supports_vision': True,
            'api_format': 'OpenAI-compatible'
        }

