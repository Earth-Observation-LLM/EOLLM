"""Unified LLM client facade.

This module provides a high-level interface for LLM operations that abstracts
away provider-specific details and integrates with the logging system.
"""

from typing import List, Dict, Any, Tuple, Optional

from .base import LLMProvider
from .registry import ProviderRegistry


class LLMClient:
    """Unified client for LLM operations across different providers.
    
    This class provides a facade over the provider system, offering a
    consistent interface regardless of which LLM provider is being used.
    It also integrates with the logging system to track all LLM calls.
    
    Attributes:
        provider: The underlying LLM provider instance
        logger: Logger instance for tracking LLM calls (optional)
        provider_name: Name of the provider being used
    """
    
    def __init__(
        self,
        provider_name: str,
        config: Any,
        logger: Optional[Any] = None
    ):
        """Initialize LLM client with a specific provider.
        
        Args:
            provider_name: Name of the provider ('groq', 'ollama', etc.)
            config: Provider-specific configuration
            logger: Optional logger instance for tracking calls
            
        Raises:
            ValueError: If provider not found or invalid
            RuntimeError: If provider initialization fails
        """
        self.provider_name = provider_name.lower()
        self.logger = logger
        
        # Create provider instance
        try:
            self.provider = ProviderRegistry.create(provider_name, config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize LLM client with provider '{provider_name}': {e}"
            ) from e
    
    def send_message(
        self,
        messages: List[Dict],
        phase: Optional[str] = None,
        pair_id: Optional[str] = None,
        tools: Optional[List[Dict]] = None
    ) -> Tuple[str, Dict[str, Any], Optional[List]]:
        """Send messages to LLM and return response with statistics.
        
        This method sends messages to the underlying provider and optionally
        logs the call details for debugging and analysis.
        
        Args:
            messages: List of message dictionaries in OpenAI format
            phase: Optional phase name for logging (e.g., 'satellite_thinking')
            pair_id: Optional pair ID for logging context
            tools: Optional list of tool definitions for function calling
            
        Returns:
            Tuple of (response_text, stats_dict, tool_calls)
            
        Raises:
            RuntimeError: If LLM call fails
            
        Example:
            >>> client = LLMClient('groq', config, logger)
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> response, stats, tool_calls = client.send_message(
            ...     messages,
            ...     phase='greeting',
            ...     pair_id='test_001'
            ... )
        """
        try:
            # Send message to provider
            response_text, stats, tool_calls = self.provider.send_message(messages, tools)
            
            # Log the call if logger is available
            if self.logger is not None:
                self._log_call(messages, response_text, stats, phase, pair_id, tool_calls)
            
            return response_text, stats, tool_calls
            
        except Exception as e:
            # Log failed call if logger is available
            if self.logger is not None:
                error_stats = {
                    'success': False,
                    'error': str(e),
                    'model': self.provider.model,
                    'latency_ms': 0,
                    'tokens_used': -1
                }
                self._log_call(messages, None, error_stats, phase, pair_id, None)
            
            raise RuntimeError(f"LLM call failed: {e}") from e
    
    def _log_call(
        self,
        messages: List[Dict],
        response: Optional[str],
        stats: Dict[str, Any],
        phase: Optional[str],
        pair_id: Optional[str],
        tool_calls: Optional[List] = None
    ) -> None:
        """Log an LLM call.
        
        Args:
            messages: Messages sent to LLM
            response: Response from LLM (None if failed)
            stats: Statistics dictionary
            phase: Phase name
            pair_id: Pair ID
            tool_calls: Tool calls made by LLM (None if no tools)
        """
        if not hasattr(self.logger, 'log_llm_call'):
            return
        
        try:
            # Add tool_calls to stats if present (for logging only)
            if tool_calls:
                stats['tool_calls'] = tool_calls
            
            self.logger.log_llm_call(
                messages=messages,
                response=response,
                stats=stats,
                phase=phase,
                pair_id=pair_id
            )
        except Exception as e:
            # Don't fail the pipeline if logging fails
            print(f"Warning: Failed to log LLM call: {e}")
    
    def supports_vision(self) -> bool:
        """Check if the current provider supports vision inputs.
        
        Returns:
            True if provider supports vision, False otherwise
        """
        return self.provider.supports_vision()
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider.
        
        Returns:
            Dictionary with provider information
        """
        info = {
            'provider_name': self.provider_name,
            'model': self.provider.model,
            'supports_vision': self.provider.supports_vision()
        }
        
        # Add provider-specific info if available
        if hasattr(self.provider, 'get_model_info'):
            info.update(self.provider.get_model_info())
        
        return info
    
    def format_message_with_images(
        self,
        text: str,
        image_paths: List[str],
        encode_images: callable
    ) -> Dict[str, Any]:
        """Format a message containing text and images.
        
        Helper method to create a properly formatted message with images.
        
        Args:
            text: Text content
            image_paths: List of image file paths
            encode_images: Function to encode images to base64
            
        Returns:
            Formatted message dictionary
            
        Example:
            >>> from src.core.image_utils import encode_image_to_base64
            >>> msg = client.format_message_with_images(
            ...     "Describe this image",
            ...     ["/path/to/image.png"],
            ...     encode_image_to_base64
            ... )
        """
        # Encode images
        image_base64_list = []
        for img_path in image_paths:
            try:
                base64_str = encode_images(img_path)
                image_base64_list.append(base64_str)
            except Exception as e:
                raise ValueError(f"Failed to encode image {img_path}: {e}")
        
        # Use provider's formatting method
        return self.provider.format_image_message(
            text, image_paths, image_base64_list
        )
    
    def __repr__(self) -> str:
        """String representation of client."""
        return f"LLMClient(provider='{self.provider_name}', model='{self.provider.model}')"

