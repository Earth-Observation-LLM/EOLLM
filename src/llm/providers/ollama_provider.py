"""Ollama local API provider implementation.

This module implements the LLMProvider interface for Ollama's local API,
which provides access to locally-hosted vision-capable models.
"""

import time
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

import requests

from ..base import LLMProvider
from ..registry import register_provider


@register_provider('ollama')
class OllamaProvider(LLMProvider):
    """Ollama local API provider for vision-capable LLM inference.
    
    This provider uses Ollama's local HTTP API to interact with models
    running on the local machine. It supports multimodal inputs (text + images).
    
    The provider handles:
    - HTTP communication with local Ollama server
    - Message formatting in Ollama's format
    - Connection error handling (server not running)
    - Base64 image handling
    
    Note: Ollama does not provide token usage statistics, so tokens_used
    will always be -1 in the stats.
    
    Attributes:
        base_url: Base URL for Ollama API (e.g., 'http://localhost:11434')
        model: Model name to use for inference
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Ollama provider.
        
        Args:
            config: Configuration dictionary or Pydantic model with:
                - model: Model name (e.g., 'llama3.2-vision:11b')
                - base_url: Ollama API base URL (default: 'http://localhost:11434')
                
        Raises:
            ValueError: If configuration is invalid
        """
        super().__init__(config)
        
        # Get base URL
        if hasattr(config, 'base_url'):
            self.base_url = config.base_url
        elif isinstance(config, dict):
            self.base_url = config.get('base_url', 'http://localhost:11434')
        else:
            self.base_url = 'http://localhost:11434'
        
        # Remove trailing slash
        self.base_url = self.base_url.rstrip('/')
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to Ollama server.
        
        Raises:
            RuntimeError: If server is not accessible
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                f"Please ensure Ollama is running."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Ollama server at {self.base_url} is not responding. "
                f"Connection timed out."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}")
    
    def _convert_messages_to_ollama_format(
        self,
        messages: List[Dict]
    ) -> List[Dict]:
        """Convert OpenAI-style messages to Ollama format.
        
        Ollama expects images as separate 'images' field with base64 strings.
        
        Args:
            messages: OpenAI-format messages
            
        Returns:
            Ollama-format messages
        """
        ollama_messages = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            ollama_msg = {'role': role}
            images = []
            text_parts = []
            
            # Parse content
            if isinstance(content, str):
                ollama_msg['content'] = content
            elif isinstance(content, list):
                # Extract text and images
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                        elif item.get('type') == 'image_url':
                            # Extract base64 from data URI
                            url = item.get('image_url', {}).get('url', '')
                            if 'base64,' in url:
                                base64_str = url.split('base64,')[1]
                                images.append(base64_str)
                    elif isinstance(item, str):
                        text_parts.append(item)
                
                ollama_msg['content'] = ' '.join(text_parts)
                
                if images:
                    ollama_msg['images'] = images
            else:
                ollama_msg['content'] = str(content)
            
            ollama_messages.append(ollama_msg)
        
        return ollama_messages
    
    def send_message(self, messages: List[Dict]) -> Tuple[str, Dict[str, Any]]:
        """Send messages to Ollama API and return response with statistics.
        
        This method sends messages to Ollama's chat API endpoint.
        It converts OpenAI-style messages to Ollama's format.
        
        Args:
            messages: List of message dictionaries in OpenAI format
            
        Returns:
            Tuple of (response_text, stats_dict) where stats includes:
            - model: Model name used
            - latency_ms: Request latency in milliseconds
            - tokens_used: Always -1 (Ollama doesn't provide this)
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
            # Convert messages to Ollama format
            ollama_messages = self._convert_messages_to_ollama_format(messages)
            
            # Prepare request payload
            payload = {
                'model': self.model,
                'messages': ollama_messages,
                'stream': False
            }
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=300  # 5 minute timeout for inference
            )
            
            response.raise_for_status()
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse response
            response_data = response.json()
            response_text = response_data.get('message', {}).get('content', '')
            
            # Build statistics dictionary
            stats = {
                'model': self.model,
                'latency_ms': latency_ms,
                'tokens_used': -1,  # Ollama doesn't provide token counts
                'timestamp': timestamp,
                'success': True,
                'error': None
            }
            
            return response_text, stats
            
        except requests.exceptions.ConnectionError as e:
            # Calculate latency even for failed requests
            latency_ms = (time.time() - start_time) * 1000
            
            stats = {
                'model': self.model,
                'latency_ms': latency_ms,
                'tokens_used': -1,
                'timestamp': timestamp,
                'success': False,
                'error': f"Connection error: {e}"
            }
            
            raise RuntimeError(
                f"Cannot connect to Ollama server at {self.base_url}. "
                f"Please ensure Ollama is running."
            ) from e
            
        except requests.exceptions.Timeout as e:
            latency_ms = (time.time() - start_time) * 1000
            
            stats = {
                'model': self.model,
                'latency_ms': latency_ms,
                'tokens_used': -1,
                'timestamp': timestamp,
                'success': False,
                'error': f"Timeout: {e}"
            }
            
            raise RuntimeError(
                f"Ollama request timed out after {latency_ms/1000:.1f}s"
            ) from e
            
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
            
            raise RuntimeError(f"Ollama API call failed: {e}") from e
    
    def supports_vision(self) -> bool:
        """Check if this provider supports vision inputs.
        
        Returns:
            True (Ollama supports vision models)
        """
        return True
    
    def list_available_models(self) -> List[str]:
        """List models available on the Ollama server.
        
        Returns:
            List of model names
            
        Raises:
            RuntimeError: If unable to fetch models
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            raise RuntimeError(f"Failed to list Ollama models: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'provider': 'ollama',
            'model': self.model,
            'base_url': self.base_url,
            'supports_vision': True,
            'api_format': 'Ollama native'
        }

