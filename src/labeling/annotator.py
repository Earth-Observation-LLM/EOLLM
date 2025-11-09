"""Image annotation logic for labeling pipeline.

This module implements the 2-phase annotation process:
Phase 1: Satellite image annotation (thinking → annotation)
Phase 2: Street view annotation (thinking → annotation, continuing conversation)
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from src.core.image_utils import encode_image_to_base64
from src.core.image_patches import extract_patch, save_patch_image, get_patch_info
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
    
    def _get_zoom_tool_definition(self, image_type: str) -> Dict:
        """Get the zoom tool definition for the current image type.
        
        Args:
            image_type: Either 'satellite' or 'street_view'
            
        Returns:
            Tool definition in OpenAI format
        """
        return {
            "type": "function",
            "function": {
                "name": "request_zoom_patch",
                "description": f"Request a zoomed-in patch of the {image_type} image for detailed analysis of a specific region",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_type": {
                            "type": "string",
                            "enum": ["satellite", "street_view"],
                            "description": "Which image to zoom into"
                        },
                        "patch": {
                            "type": "string",
                            "enum": ["northwest", "northeast", "southwest", "southeast", "center"],
                            "description": "Which region: northwest/northeast/southwest/southeast (quadrants) or center (middle 50%)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for requesting this zoom (for logging)"
                        }
                    },
                    "required": ["image_type", "patch"]
                }
            }
        }
    
    def _handle_tool_calls(
        self,
        tool_calls: List,
        current_image_path: str,
        conversation: List[Dict],
        pair_id: str,
        zoom_count: int,
        patches_dir: Path,
        image_type: str
    ) -> Tuple[Optional[str], int]:
        """Handle zoom tool calls from the LLM.
        
        Args:
            tool_calls: List of tool calls from LLM response
            current_image_path: Path to the current image being analyzed
            conversation: Conversation history (modified in place)
            pair_id: ID of the pair being processed
            zoom_count: Current zoom count
            patches_dir: Directory to save patches
            image_type: Either 'satellite' or 'street_view'
            
        Returns:
            Tuple of (last_response_text, updated_zoom_count)
        """
        last_response = None
        
        for tool_call in tool_calls:
            if zoom_count >= 5:
                # Max zooms reached
                break
            
            # Extract tool call info
            tool_call_id = tool_call.id if hasattr(tool_call, 'id') else str(zoom_count)
            function_name = tool_call.function.name if hasattr(tool_call.function, 'name') else 'unknown'
            
            # Parse arguments
            try:
                if hasattr(tool_call.function, 'arguments'):
                    arguments = json.loads(tool_call.function.arguments)
                else:
                    arguments = {}
            except json.JSONDecodeError:
                arguments = {}
            
            requested_patch = arguments.get('patch', 'center')
            reason = arguments.get('reason', 'No reason provided')
            
            self.logger.info(
                f"Tool call: {function_name} for {image_type} - "
                f"patch={requested_patch}, reason={reason}"
            )
            
            try:
                # Extract patch
                patch_bytes = extract_patch(current_image_path, requested_patch)
                
                # Save patch
                zoom_count += 1
                patch_filename = f"{image_type}_{requested_patch}_{zoom_count}"
                patch_path = patches_dir / f"{patch_filename}.png"
                saved_path = save_patch_image(patch_bytes, patch_path)
                
                # Get patch info for logging
                patch_info = get_patch_info(current_image_path, requested_patch)
                
                self.logger.info(
                    f"Patch saved: {saved_path} "
                    f"(coordinates: {patch_info['coordinates']})"
                )
                
                # Encode patch for sending to LLM
                import base64
                patch_base64 = base64.b64encode(patch_bytes).decode('utf-8')
                
                # Add tool response to conversation
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Here is the {requested_patch} patch of the {image_type} image you requested."
                })
                
                # Add the patch image to conversation
                conversation.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Zoomed patch ({requested_patch} region):"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{patch_base64}"
                            }
                        }
                    ]
                })
                
                # Log patch details to stats (will be captured in LLM call log)
                # This is done automatically by the logger
                
            except Exception as e:
                self.logger.error(f"Failed to extract/send patch: {e}")
                # Add error response
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Error: Could not extract {requested_patch} patch. Continuing without it."
                })
        
        # If tool calls were made, get the next LLM response
        if zoom_count > 0:
            try:
                # Get response after providing patch
                response, stats, _ = self.llm.send_message(
                    conversation,
                    phase=f"{image_type}_zoom_analysis",
                    pair_id=pair_id,
                    tools=None  # Don't offer tool again immediately
                )
                last_response = response
            except Exception as e:
                self.logger.error(f"Error getting response after tool call: {e}")
                last_response = None
        
        return last_response, zoom_count
    
    def annotate_satellite(
        self,
        image_path: str,
        conversation: List[Dict],
        pair_id: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Annotate satellite image with 2-step reasoning and zoom tool support.
        
        Args:
            image_path: Path to satellite image
            conversation: Conversation history (modified in place)
            pair_id: ID of the pair being processed
            output_dir: Optional output directory for patches
            
        Returns:
            Dictionary with thinking, annotation, and stats
        """
        self.logger.log_step("Encoding satellite image")
        
        # Create patches directory if output_dir provided
        patches_dir = None
        if output_dir:
            patches_dir = output_dir / "patches" / pair_id
            patches_dir.mkdir(parents=True, exist_ok=True)
        
        # Encode image
        image_base64 = encode_image_to_base64(image_path)
        
        # Step 1: Thinking prompt with zoom tool
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
        
        # Define zoom tool
        zoom_tool = self._get_zoom_tool_definition("satellite")
        
        # Get thinking response with tool support
        self.logger.log_step("Requesting satellite thinking from LLM (with zoom tool)")
        zoom_count = 0
        max_zooms = 5
        
        thinking_response, thinking_stats, tool_calls = self.llm.send_message(
            conversation,
            phase='satellite_thinking',
            pair_id=pair_id,
            tools=[zoom_tool]
        )
        
        # Handle tool calls (zoom requests)
        while tool_calls and zoom_count < max_zooms and patches_dir:
            self.logger.log_step(f"Handling {len(tool_calls)} zoom request(s)")
            
            # Add assistant message with tool calls to conversation
            conversation.append({
                "role": "assistant",
                "content": thinking_response,
                "tool_calls": tool_calls
            })
            
            # Process tool calls
            zoom_response, zoom_count = self._handle_tool_calls(
                tool_calls,
                image_path,
                conversation,
                pair_id,
                zoom_count,
                patches_dir,
                "satellite"
            )
            
            # Get next response (with tool if under limit)
            tools = [zoom_tool] if zoom_count < max_zooms else None
            thinking_response, thinking_stats, tool_calls = self.llm.send_message(
                conversation,
                phase='satellite_thinking',
                pair_id=pair_id,
                tools=tools
            )
        
        # Add final thinking response to conversation
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
        
        # Get annotation response (no tools for annotation step)
        self.logger.log_step("Requesting satellite annotation from LLM")
        annotation_response, annotation_stats, _ = self.llm.send_message(
            conversation,
            phase='satellite_annotation',
            pair_id=pair_id,
            tools=None
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
        pair_id: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Annotate street view image with 2-step reasoning and zoom tool support.
        
        This continues the conversation from satellite annotation,
        providing context for street-level analysis.
        
        Args:
            image_path: Path to street view image
            conversation: Conversation history (modified in place)
            pair_id: ID of the pair being processed
            output_dir: Optional output directory for patches
            
        Returns:
            Dictionary with thinking, annotation, and stats
        """
        self.logger.log_step("Encoding street view image")
        
        # Create patches directory if output_dir provided
        patches_dir = None
        if output_dir:
            patches_dir = output_dir / "patches" / pair_id
            patches_dir.mkdir(parents=True, exist_ok=True)
        
        # Encode image
        image_base64 = encode_image_to_base64(image_path)
        
        # Step 1: Thinking prompt with zoom tool
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
        
        # Define zoom tool
        zoom_tool = self._get_zoom_tool_definition("street_view")
        
        # Get thinking response with tool support
        self.logger.log_step("Requesting street view thinking from LLM (with zoom tool)")
        zoom_count = 0
        max_zooms = 5
        
        thinking_response, thinking_stats, tool_calls = self.llm.send_message(
            conversation,
            phase='street_thinking',
            pair_id=pair_id,
            tools=[zoom_tool]
        )
        
        # Handle tool calls (zoom requests)
        while tool_calls and zoom_count < max_zooms and patches_dir:
            self.logger.log_step(f"Handling {len(tool_calls)} zoom request(s)")
            
            # Add assistant message with tool calls to conversation
            conversation.append({
                "role": "assistant",
                "content": thinking_response,
                "tool_calls": tool_calls
            })
            
            # Process tool calls
            zoom_response, zoom_count = self._handle_tool_calls(
                tool_calls,
                image_path,
                conversation,
                pair_id,
                zoom_count,
                patches_dir,
                "street_view"
            )
            
            # Get next response (with tool if under limit)
            tools = [zoom_tool] if zoom_count < max_zooms else None
            thinking_response, thinking_stats, tool_calls = self.llm.send_message(
                conversation,
                phase='street_thinking',
                pair_id=pair_id,
                tools=tools
            )
        
        # Add final thinking response to conversation
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
        
        # Get annotation response (no tools for annotation step)
        self.logger.log_step("Requesting street view annotation from LLM")
        annotation_response, annotation_stats, _ = self.llm.send_message(
            conversation,
            phase='street_annotation',
            pair_id=pair_id,
            tools=None
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

