"""Image annotation logic for labeling pipeline.

This module implements the 2-phase annotation process:
Phase 1: Satellite image annotation (thinking → annotation)
Phase 2: Street view annotation (thinking → annotation, continuing conversation)
"""

from typing import Dict, Any, List

from src.core.image_utils import encode_image_to_base64
from src.prompts.renderer import PromptRenderer
from src.llm.client import LLMClient


class ImageAnnotator:
    """Handles annotation of satellite and street view images.
    
    Implements the 2-step reasoning process for each image:
    1. Thinking step: Detailed analysis
    2. Annotation step: Concise summary
    
    Maintains conversation history to provide context from satellite
    to street view annotations.
    
    Attributes:
        llm_client: LLM client for API calls
        prompt_renderer: Jinja2 prompt renderer
        logger: Logger instance
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_renderer: PromptRenderer,
        logger: Any
    ):
        """Initialize image annotator.
        
        Args:
            llm_client: Configured LLM client
            prompt_renderer: Prompt renderer with templates
            logger: Logger instance for tracking
        """
        self.llm = llm_client
        self.prompts = prompt_renderer
        self.logger = logger
    
    def annotate_satellite(
        self,
        image_path: str,
        conversation: List[Dict],
        pair_id: str
    ) -> Dict[str, Any]:
        """Annotate satellite image with 2-step reasoning.
        
        Args:
            image_path: Path to satellite image
            conversation: Conversation history (modified in place)
            pair_id: ID of the pair being processed
            
        Returns:
            Dictionary with thinking, annotation, and stats
        """
        self.logger.log_step("Encoding satellite image")
        
        # Encode image
        image_base64 = encode_image_to_base64(image_path)
        
        # Step 1: Thinking prompt
        self.logger.log_step("Rendering satellite thinking prompt")
        thinking_prompt = self.prompts.render('labeling/satellite_thinking.j2')
        
        # Build message with image
        thinking_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": thinking_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
        
        conversation.append(thinking_msg)
        
        # Get thinking response
        self.logger.log_step("Requesting satellite thinking from LLM")
        thinking_response, thinking_stats = self.llm.send_message(
            conversation,
            phase='satellite_thinking',
            pair_id=pair_id
        )
        
        # Add to conversation
        conversation.append({
            "role": "assistant",
            "content": thinking_response
        })
        
        # Step 2: Annotation prompt
        self.logger.log_step("Rendering satellite annotation prompt")
        annotation_prompt = self.prompts.render('labeling/satellite_annotation.j2')
        
        conversation.append({
            "role": "user",
            "content": annotation_prompt
        })
        
        # Get annotation response
        self.logger.log_step("Requesting satellite annotation from LLM")
        annotation_response, annotation_stats = self.llm.send_message(
            conversation,
            phase='satellite_annotation',
            pair_id=pair_id
        )
        
        # Add to conversation
        conversation.append({
            "role": "assistant",
            "content": annotation_response
        })
        
        # Log the annotation
        self.logger.log_annotation(
            'satellite',
            thinking_response,
            annotation_response
        )
        
        # Combine stats
        combined_stats = {
            'thinking': thinking_stats,
            'annotation': annotation_stats,
            'total_latency_ms': (
                thinking_stats.get('latency_ms', 0) +
                annotation_stats.get('latency_ms', 0)
            )
        }
        
        return {
            'image_path': image_path,
            'thinking': thinking_response,
            'annotation': annotation_response,
            'stats': combined_stats
        }
    
    def annotate_street_view(
        self,
        image_path: str,
        conversation: List[Dict],
        pair_id: str
    ) -> Dict[str, Any]:
        """Annotate street view image with 2-step reasoning.
        
        This continues the conversation from satellite annotation,
        providing context for street-level analysis.
        
        Args:
            image_path: Path to street view image
            conversation: Conversation history (modified in place)
            pair_id: ID of the pair being processed
            
        Returns:
            Dictionary with thinking, annotation, and stats
        """
        self.logger.log_step("Encoding street view image")
        
        # Encode image
        image_base64 = encode_image_to_base64(image_path)
        
        # Step 1: Thinking prompt
        self.logger.log_step("Rendering street view thinking prompt")
        thinking_prompt = self.prompts.render('labeling/street_thinking.j2')
        
        # Build message with image
        thinking_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": thinking_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}"
                    }
                }
            ]
        }
        
        conversation.append(thinking_msg)
        
        # Get thinking response
        self.logger.log_step("Requesting street view thinking from LLM")
        thinking_response, thinking_stats = self.llm.send_message(
            conversation,
            phase='street_thinking',
            pair_id=pair_id
        )
        
        # Add to conversation
        conversation.append({
            "role": "assistant",
            "content": thinking_response
        })
        
        # Step 2: Annotation prompt
        self.logger.log_step("Rendering street view annotation prompt")
        annotation_prompt = self.prompts.render('labeling/street_annotation.j2')
        
        conversation.append({
            "role": "user",
            "content": annotation_prompt
        })
        
        # Get annotation response
        self.logger.log_step("Requesting street view annotation from LLM")
        annotation_response, annotation_stats = self.llm.send_message(
            conversation,
            phase='street_annotation',
            pair_id=pair_id
        )
        
        # Add to conversation
        conversation.append({
            "role": "assistant",
            "content": annotation_response
        })
        
        # Log the annotation
        self.logger.log_annotation(
            'street_view',
            thinking_response,
            annotation_response
        )
        
        # Combine stats
        combined_stats = {
            'thinking': thinking_stats,
            'annotation': annotation_stats,
            'total_latency_ms': (
                thinking_stats.get('latency_ms', 0) +
                annotation_stats.get('latency_ms', 0)
            )
        }
        
        return {
            'image_path': image_path,
            'thinking': thinking_response,
            'annotation': annotation_response,
            'stats': combined_stats
        }

