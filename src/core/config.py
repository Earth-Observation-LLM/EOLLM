"""Configuration management with YAML loading and Pydantic validation.

This module provides type-safe configuration loading with environment variable
interpolation and comprehensive validation.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class GroqProviderConfig(BaseModel):
    """Configuration for Groq API provider."""
    
    enabled: bool = True
    model: str = "meta-llama/llama-4-maverick-17b-128e-instruct"
    api_key_env: str = "GROQ_API_KEY"
    
    @field_validator('api_key_env')
    @classmethod
    def validate_api_key_env(cls, v: str) -> str:
        """Validate that API key environment variable is set."""
        if not os.getenv(v):
            raise ValueError(f"Environment variable {v} is not set")
        return v


class OllamaProviderConfig(BaseModel):
    """Configuration for Ollama local provider."""
    
    enabled: bool = True
    model: str = "llama3.2-vision:11b"
    base_url: str = "http://localhost:11434"


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    
    default_provider: str = Field(
        default="groq",
        description="Default LLM provider to use"
    )
    providers: Dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific configurations"
    )
    
    @field_validator('default_provider')
    @classmethod
    def validate_default_provider(cls, v: str) -> str:
        """Ensure default provider is valid."""
        valid_providers = ['groq', 'ollama']
        if v not in valid_providers:
            raise ValueError(f"default_provider must be one of {valid_providers}")
        return v
    
    @model_validator(mode='after')
    def validate_default_provider_enabled(self) -> 'LLMConfig':
        """Ensure default provider is enabled."""
        if self.default_provider not in self.providers:
            raise ValueError(f"Default provider '{self.default_provider}' not found in providers config")
        
        provider_config = self.providers[self.default_provider]
        if isinstance(provider_config, dict) and not provider_config.get('enabled', True):
            raise ValueError(f"Default provider '{self.default_provider}' is disabled")
        
        return self


class LLMCallLoggingConfig(BaseModel):
    """Configuration for LLM call logging."""
    
    enabled: bool = True
    include_images: bool = False
    stats: List[str] = Field(
        default_factory=lambda: ["latency", "tokens", "model", "timestamp"]
    )


class LogLevelsConfig(BaseModel):
    """Log level configuration."""
    
    console: str = "INFO"
    file: str = "DEBUG"
    
    @field_validator('console', 'file')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper


class LoggingPipelineConfig(BaseModel):
    """Configuration for a specific pipeline's logging."""
    
    output_dir: str
    levels: LogLevelsConfig = Field(default_factory=LogLevelsConfig)
    llm_call_logging: LLMCallLoggingConfig = Field(
        default_factory=LLMCallLoggingConfig
    )


class LoggingConfig(BaseModel):
    """Logging configuration for all pipelines."""
    
    labeling: LoggingPipelineConfig
    vqa: LoggingPipelineConfig


class PathsConfig(BaseModel):
    """File path configuration."""
    
    input_structure: str = "input_structure.json"
    templates: str = "src/prompts/templates"
    data_root: str = "data"


class ZoomToolConfig(BaseModel):
    """Configuration for zoom/patch extraction tool."""
    
    enabled: bool = True
    max_zooms_per_image: int = Field(default=5, ge=1, le=10)
    save_patches: bool = True
    patch_format: str = Field(default="png", pattern="^(png|jpg|jpeg)$")


class LabelingResearchConfig(BaseModel):
    """Research settings for labeling pipeline."""
    
    max_retries: int = Field(default=3, ge=1, le=10)
    conversation_mode: bool = True
    zoom_tool: ZoomToolConfig = Field(default_factory=ZoomToolConfig)


class VQAResearchConfig(BaseModel):
    """Research settings for VQA pipeline."""
    
    min_questions: int = Field(default=15, ge=1)
    max_questions: int = Field(default=20, ge=1)
    categories: List[str] = Field(
        default_factory=lambda: [
            "sustainability",
            "infrastructure",
            "economic",
            "maintenance",
            "development"
        ]
    )
    fresh_conversation: bool = True
    
    @model_validator(mode='after')
    def validate_question_range(self) -> 'VQAResearchConfig':
        """Ensure min_questions <= max_questions."""
        if self.min_questions > self.max_questions:
            raise ValueError("min_questions must be <= max_questions")
        return self


class ResearchConfig(BaseModel):
    """Research-specific settings."""
    
    labeling: LabelingResearchConfig = Field(
        default_factory=LabelingResearchConfig
    )
    vqa: VQAResearchConfig = Field(default_factory=VQAResearchConfig)


class Config(BaseModel):
    """Main configuration model."""
    
    llm: LLMConfig
    logging: LoggingConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    
    # Project root for resolving relative paths
    _project_root: Optional[Path] = None
    
    def resolve_path(self, path: str) -> Path:
        """Resolve a path relative to project root.
        
        Args:
            path: Path to resolve (can be absolute or relative)
            
        Returns:
            Absolute Path object
        """
        p = Path(path)
        if p.is_absolute():
            return p
        
        if self._project_root:
            return self._project_root / p
        return p.resolve()
    
    def set_project_root(self, root: Path) -> None:
        """Set the project root directory.
        
        Args:
            root: Project root directory path
        """
        self._project_root = root


def interpolate_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively interpolate environment variables in config.
    
    Replaces ${VAR_NAME} patterns with environment variable values.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configuration with interpolated values
    """
    env_var_pattern = re.compile(r'\$\{([^}]+)\}')
    
    def interpolate_value(value: Any) -> Any:
        if isinstance(value, str):
            # Find all ${VAR} patterns
            matches = env_var_pattern.findall(value)
            for var_name in matches:
                env_value = os.getenv(var_name, '')
                value = value.replace(f'${{{var_name}}}', env_value)
            return value
        elif isinstance(value, dict):
            return {k: interpolate_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [interpolate_value(item) for item in value]
        return value
    
    return interpolate_value(config_dict)


def load_config(config_path: str = "config/settings.yaml") -> Config:
    """Load and validate configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated Config object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please create a config/settings.yaml file."
        )
    
    # Load YAML
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    if not config_dict:
        raise ValueError("Configuration file is empty")
    
    # Interpolate environment variables
    config_dict = interpolate_env_vars(config_dict)
    
    # Parse providers into proper config objects
    if 'llm' in config_dict and 'providers' in config_dict['llm']:
        providers = config_dict['llm']['providers']
        
        if 'groq' in providers:
            providers['groq'] = GroqProviderConfig(**providers['groq'])
        if 'ollama' in providers:
            providers['ollama'] = OllamaProviderConfig(**providers['ollama'])
    
    # Validate with Pydantic
    try:
        config = Config(**config_dict)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {e}")
    
    # Set project root (parent of config directory)
    project_root = config_file.parent.parent
    config.set_project_root(project_root)
    
    return config


def get_provider_config(config: Config, provider_name: Optional[str] = None) -> Any:
    """Get configuration for a specific provider.
    
    Args:
        config: Main configuration object
        provider_name: Name of provider, or None to use default
        
    Returns:
        Provider configuration object
        
    Raises:
        ValueError: If provider not found or not enabled
    """
    if provider_name is None:
        provider_name = config.llm.default_provider
    
    if provider_name not in config.llm.providers:
        raise ValueError(f"Provider '{provider_name}' not found in configuration")
    
    provider_config = config.llm.providers[provider_name]
    
    if hasattr(provider_config, 'enabled') and not provider_config.enabled:
        raise ValueError(f"Provider '{provider_name}' is disabled in configuration")
    
    return provider_config

