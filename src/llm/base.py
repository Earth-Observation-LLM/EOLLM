"""Abstract base class for LLM providers.

This module defines the interface that all LLM providers must implement,
ensuring consistent behavior across different API backends (Groq, Ollama, etc.).
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class LLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All LLM providers (Groq, Ollama, etc.) must inherit from this class
    and implement its abstract methods. This ensures a consistent interface
    for the rest of the pipeline.
    
    The provider is responsible for:
    - Managing API connections
    - Formatting messages for the specific API
    - Handling retries and error recovery
    - Extracting usage statistics
    
    Attributes:
        config: Provider-specific configuration dictionary
        model: Name of the model being used
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with its configuration.
        
        Args:
            config: Provider-specific configuration dictionary.
                   Must contain at least 'model' key.
        
        Raises:
            ValueError: If required config keys are missing
        """
        self.config = config
        self.model = self._extract_model_name(config)
    
    def _extract_model_name(self, config: Dict[str, Any]) -> str:
        """Extract model name from config.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Model name string
            
        Raises:
            ValueError: If model name not found
        """
        if hasattr(config, 'model'):
            return config.model
        elif isinstance(config, dict) and 'model' in config:
            return config['model']
        else:
            raise ValueError("Configuration must include 'model' key")
    
    @abstractmethod
    def send_message(self, messages: List[Dict]) -> Tuple[str, Dict[str, Any]]:
        """Send messages to the LLM and return response with statistics.
        
        This is the core method that must be implemented by all providers.
        It sends a list of messages (which may include images) to the LLM
        and returns both the response text and usage statistics.
        
        Args:
            messages: List of message dictionaries in OpenAI format.
                     Each message has 'role' and 'content'.
                     Content can be string or list of content items
                     (text, images).
        
        Returns:
            Tuple of (response_text, stats_dict) where stats_dict must include:
            - model: str - Model name used
            - latency_ms: float - Request latency in milliseconds
            - tokens_used: int - Total tokens used (or -1 if unavailable)
            - timestamp: str - ISO format timestamp of the call
            
        Raises:
            RuntimeError: If API call fails after retries
            ValueError: If message format is invalid
        
        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello!"}
            ... ]
            >>> response, stats = provider.send_message(messages)
            >>> print(f"Response: {response}")
            >>> print(f"Latency: {stats['latency_ms']}ms")
        """
        pass
    
    @abstractmethod
    def supports_vision(self) -> bool:
        """Check if this provider supports vision/image inputs.
        
        Returns:
            True if the provider can handle image inputs, False otherwise
        
        Example:
            >>> if provider.supports_vision():
            ...     # Can send images
            ...     pass
        """
        pass
    
    def format_image_message(
        self,
        text: str,
        image_paths: List[str],
        image_base64_list: List[str]
    ) -> Dict[str, Any]:
        """Format a message containing text and images.
        
        This helper method provides a default implementation for formatting
        multimodal messages. Providers can override if they need custom formatting.
        
        Args:
            text: Text content of the message
            image_paths: List of image file paths (for logging)
            image_base64_list: List of base64-encoded images
            
        Returns:
            Message dictionary with text and images
        """
        content = [{"type": "text", "text": text}]
        
        for base64_str in image_base64_list:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_str}"
                }
            })
        
        return {
            "role": "user",
            "content": content
        }
    
    def extract_text_from_message(self, message: Dict[str, Any]) -> str:
        """Extract text content from a message.
        
        Helper method to extract just the text from a message that may
        contain multiple content types (text, images).
        
        Args:
            message: Message dictionary
            
        Returns:
            Extracted text content
        """
        content = message.get("content", "")
        
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            # Extract text from content list
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            return " ".join(text_parts)
        
        return str(content)
    
    def validate_messages(self, messages: List[Dict]) -> None:
        """Validate message format.
        
        Ensures messages follow the expected format before sending to API.
        
        Args:
            messages: List of message dictionaries
            
        Raises:
            ValueError: If message format is invalid
        """
        if not messages:
            raise ValueError("Messages list cannot be empty")
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            
            if "role" not in msg:
                raise ValueError(f"Message {i} missing 'role' field")
            
            if "content" not in msg:
                raise ValueError(f"Message {i} missing 'content' field")
            
            valid_roles = ["system", "user", "assistant"]
            if msg["role"] not in valid_roles:
                raise ValueError(
                    f"Message {i} has invalid role '{msg['role']}'. "
                    f"Must be one of {valid_roles}"
                )
    
    def __repr__(self) -> str:
        """String representation of provider."""
        return f"{self.__class__.__name__}(model='{self.model}')"

