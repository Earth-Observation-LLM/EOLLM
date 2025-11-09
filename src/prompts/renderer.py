"""Jinja2-based prompt rendering system.

This module provides a sophisticated templating system for prompts using Jinja2,
with custom filters and comprehensive error handling.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import jinja2


class PromptRenderer:
    """Jinja2-based prompt rendering engine.
    
    This class manages prompt templates and provides rendering capabilities
    with custom filters for text processing. It enables separation of
    prompts from code and allows easy customization of prompt content.
    
    Attributes:
        template_dir: Directory containing template files
        env: Jinja2 environment instance
    """
    
    def __init__(self, template_dir: str):
        """Initialize the prompt renderer.
        
        Args:
            template_dir: Path to directory containing Jinja2 templates
            
        Raises:
            FileNotFoundError: If template directory doesn't exist
        """
        self.template_dir = Path(template_dir)
        
        if not self.template_dir.exists():
            raise FileNotFoundError(
                f"Template directory not found: {template_dir}"
            )
        
        if not self.template_dir.is_dir():
            raise ValueError(
                f"Template path is not a directory: {template_dir}"
            )
        
        # Initialize Jinja2 environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
            keep_trailing_newline=False
        )
        
        # Register custom filters
        self._register_filters()
    
    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""
        
        # Truncate text intelligently (at word boundaries)
        def truncate_smart(text: str, length: int = 200) -> str:
            """Truncate text at word boundary."""
            if len(text) <= length:
                return text
            
            # Find last space before length
            truncated = text[:length]
            last_space = truncated.rfind(' ')
            
            if last_space > 0:
                return truncated[:last_space] + '...'
            return truncated + '...'
        
        # Prioritize items by weight
        def prioritize(items: list, key: Optional[str] = None) -> list:
            """Sort items by priority/weight."""
            if not items:
                return items
            
            if key:
                return sorted(
                    items,
                    key=lambda x: x.get(key, 0) if isinstance(x, dict) else 0,
                    reverse=True
                )
            return sorted(items, reverse=True)
        
        # Join with oxford comma
        def oxford_join(items: list, conjunction: str = 'and') -> str:
            """Join list with oxford comma."""
            if not items:
                return ''
            if len(items) == 1:
                return str(items[0])
            if len(items) == 2:
                return f"{items[0]} {conjunction} {items[1]}"
            
            return ', '.join(str(x) for x in items[:-1]) + f", {conjunction} {items[-1]}"
        
        # Capitalize first letter of each sentence
        def sentence_case(text: str) -> str:
            """Apply sentence case to text."""
            sentences = text.split('. ')
            return '. '.join(s.capitalize() for s in sentences)
        
        # Number list items
        def enumerate_items(items: list, start: int = 1) -> list:
            """Add numbers to list items."""
            return [
                {'number': i + start, 'item': item}
                for i, item in enumerate(items)
            ]
        
        # Register all filters
        self.env.filters['truncate_smart'] = truncate_smart
        self.env.filters['prioritize'] = prioritize
        self.env.filters['oxford_join'] = oxford_join
        self.env.filters['sentence_case'] = sentence_case
        self.env.filters['enumerate_items'] = enumerate_items
    
    def render(self, template_name: str, **context) -> str:
        """Render a template with the given context.
        
        Args:
            template_name: Name of template file (relative to template_dir)
            **context: Variables to pass to template
            
        Returns:
            Rendered template string
            
        Raises:
            jinja2.TemplateNotFound: If template file doesn't exist
            jinja2.TemplateError: If template rendering fails
            
        Example:
            >>> renderer = PromptRenderer('templates')
            >>> prompt = renderer.render(
            ...     'labeling/satellite_thinking.j2',
            ...     image_type='satellite',
            ...     analysis_dimensions=[...]
            ... )
        """
        try:
            template = self.env.get_template(template_name)
            rendered = template.render(**context)
            return rendered.strip()
        except jinja2.TemplateNotFound as e:
            raise jinja2.TemplateNotFound(
                f"Template not found: {template_name}. "
                f"Available templates in {self.template_dir}"
            ) from e
        except jinja2.TemplateError as e:
            raise jinja2.TemplateError(
                f"Error rendering template {template_name}: {e}"
            ) from e
    
    def render_string(self, template_string: str, **context) -> str:
        """Render a template from a string.
        
        Useful for dynamic template generation or testing.
        
        Args:
            template_string: Template content as string
            **context: Variables to pass to template
            
        Returns:
            Rendered string
            
        Example:
            >>> renderer = PromptRenderer('templates')
            >>> prompt = renderer.render_string(
            ...     'Hello {{ name }}!',
            ...     name='World'
            ... )
        """
        try:
            template = self.env.from_string(template_string)
            return template.render(**context).strip()
        except jinja2.TemplateError as e:
            raise jinja2.TemplateError(
                f"Error rendering template string: {e}"
            ) from e
    
    def list_templates(self, pattern: str = "*.j2") -> list[str]:
        """List available templates in template directory.
        
        Args:
            pattern: Glob pattern for template files
            
        Returns:
            List of template file paths (relative to template_dir)
        """
        templates = []
        for template_path in self.template_dir.rglob(pattern):
            rel_path = template_path.relative_to(self.template_dir)
            templates.append(str(rel_path))
        return sorted(templates)
    
    def template_exists(self, template_name: str) -> bool:
        """Check if a template exists.
        
        Args:
            template_name: Template name to check
            
        Returns:
            True if template exists, False otherwise
        """
        template_path = self.template_dir / template_name
        return template_path.exists()
    
    def get_default_context(self) -> Dict[str, Any]:
        """Get default context variables for templates.
        
        These variables are always available in templates.
        
        Returns:
            Dictionary of default context variables
        """
        return {
            'newline': '\n',
            'tab': '\t',
            'empty': ''
        }
    
    def render_with_defaults(
        self,
        template_name: str,
        **context
    ) -> str:
        """Render template with default context merged.
        
        Args:
            template_name: Template name
            **context: Additional context variables
            
        Returns:
            Rendered template string
        """
        full_context = self.get_default_context()
        full_context.update(context)
        return self.render(template_name, **full_context)
    
    def validate_template(self, template_name: str) -> tuple[bool, Optional[str]]:
        """Validate that a template can be loaded and parsed.
        
        Args:
            template_name: Template name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
            error_message is None if template is valid
        """
        try:
            template = self.env.get_template(template_name)
            # Try rendering with empty context to check for syntax errors
            template.render()
            return True, None
        except jinja2.TemplateNotFound:
            return False, f"Template not found: {template_name}"
        except jinja2.TemplateError as e:
            return False, f"Template error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"
    
    def __repr__(self) -> str:
        """String representation of renderer."""
        return f"PromptRenderer(template_dir='{self.template_dir}')"

