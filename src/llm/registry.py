"""Provider registry for LLM provider management.

This module implements a registry pattern for managing LLM providers,
allowing dynamic provider registration and instantiation.
"""

from typing import Dict, Type, Any, Optional
from .base import LLMProvider


class ProviderRegistry:
    """Singleton registry for LLM providers.
    
    This class maintains a global registry of available LLM providers
    and provides factory methods for creating provider instances.
    
    Providers are automatically registered when imported, and can be
    created by name using the create() factory method.
    """
    
    _providers: Dict[str, Type[LLMProvider]] = {}
    _instance: Optional['ProviderRegistry'] = None
    
    def __new__(cls) -> 'ProviderRegistry':
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """Register a provider class.
        
        Args:
            name: Provider name (e.g., 'groq', 'ollama')
            provider_class: Provider class that inherits from LLMProvider
            
        Raises:
            ValueError: If provider_class doesn't inherit from LLMProvider
            
        Example:
            >>> ProviderRegistry.register('groq', GroqProvider)
        """
        if not issubclass(provider_class, LLMProvider):
            raise ValueError(
                f"Provider class must inherit from LLMProvider, "
                f"got {provider_class.__name__}"
            )
        
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create(cls, name: str, config: Any) -> LLMProvider:
        """Create a provider instance by name.
        
        Factory method that instantiates a provider with the given config.
        
        Args:
            name: Provider name (e.g., 'groq', 'ollama')
            config: Provider-specific configuration
            
        Returns:
            Instantiated LLMProvider
            
        Raises:
            ValueError: If provider name not found in registry
            
        Example:
            >>> config = {"model": "llama-3", "api_key": "..."}
            >>> provider = ProviderRegistry.create('groq', config)
        """
        name_lower = name.lower()
        
        if name_lower not in cls._providers:
            available = ', '.join(cls._providers.keys())
            raise ValueError(
                f"Provider '{name}' not found in registry. "
                f"Available providers: {available}"
            )
        
        provider_class = cls._providers[name_lower]
        
        try:
            return provider_class(config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create provider '{name}': {e}"
            )
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names.
        
        Returns:
            List of registered provider names
            
        Example:
            >>> providers = ProviderRegistry.list_providers()
            >>> print(providers)
            ['groq', 'ollama']
        """
        return list(cls._providers.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered.
        
        Args:
            name: Provider name to check
            
        Returns:
            True if provider is registered, False otherwise
        """
        return name.lower() in cls._providers
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a provider.
        
        Mainly used for testing purposes.
        
        Args:
            name: Provider name to unregister
        """
        name_lower = name.lower()
        if name_lower in cls._providers:
            del cls._providers[name_lower]
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers.
        
        Mainly used for testing purposes.
        """
        cls._providers.clear()


def register_provider(name: str) -> callable:
    """Decorator for automatic provider registration.
    
    This decorator automatically registers a provider class when the
    module is imported.
    
    Args:
        name: Provider name to register under
        
    Returns:
        Decorator function
        
    Example:
        >>> @register_provider('groq')
        ... class GroqProvider(LLMProvider):
        ...     pass
    """
    def decorator(cls: Type[LLMProvider]) -> Type[LLMProvider]:
        ProviderRegistry.register(name, cls)
        return cls
    return decorator

