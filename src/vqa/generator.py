"""VQA generation pipeline.

This module generates Visual Question Answering examples from annotated
satellite and street view images using a fresh conversation context.
"""

import json
from typing import Dict, Any, List, Optional

from src.core.image_utils import encode_image_to_base64
from src.core.config import Config
from src.prompts.renderer import PromptRenderer
from src.llm.client import LLMClient
from src.logging.vqa_logger import VQALogger
from .validator import VQAValidator


class VQAGenerator:
    """Research-grade VQA generation pipeline.
    
    Generates VQA examples from satellite and street view images
    with their annotations, using a fresh conversation (not continuing
    from labeling conversation).
    
    Attributes:
        config: Configuration object
        llm_client: LLM client for API calls
        prompt_renderer: Prompt renderer
        logger: VQA logger
        validator: VQA validator for quality checks
    """
    
    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        prompt_renderer: PromptRenderer,
        logger: VQALogger
    ):
        """Initialize VQA generator.
        
        Args:
            config: Configuration object
            llm_client: Configured LLM client
            prompt_renderer: Prompt renderer with templates
            logger: VQA logger instance
        """
        self.config = config
        self.llm = llm_client
        self.prompts = prompt_renderer
        self.logger = logger
        
        # Initialize validator
        vqa_config = config.research.vqa
        self.validator = VQAValidator(
            min_questions=vqa_config.min_questions,
            max_questions=vqa_config.max_questions,
            categories=vqa_config.categories
        )
    
    def generate(
        self,
        satellite_path: str,
        street_path: str,
        satellite_annotation: str,
        street_annotation: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate VQA examples in fresh conversation.
        
        Args:
            satellite_path: Path to satellite image
            street_path: Path to street view image
            satellite_annotation: Annotation of satellite image
            street_annotation: Annotation of street view image
            session_id: Optional session identifier for logging
            
        Returns:
            Dictionary with VQA examples, validation report, and stats
        """
        if session_id is None:
            from datetime import datetime
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log session start
        source_info = {
            'satellite': satellite_path,
            'street_view': street_path,
            'session_id': session_id
        }
        self.logger.start_vqa_generation(session_id, source_info)
        
        try:
            # Encode images
            self.logger.log_step("Encoding images")
            satellite_base64 = encode_image_to_base64(satellite_path)
            street_base64 = encode_image_to_base64(street_path)
            
            # Render VQA prompt with annotations as context
            self.logger.log_step("Rendering VQA generation prompt")
            prompt = self.prompts.render(
                'vqa/generation.j2',
                satellite_annotation=satellite_annotation,
                street_annotation=street_annotation,
                categories=self.config.research.vqa.categories,
                min_questions=self.config.research.vqa.min_questions,
                max_questions=self.config.research.vqa.max_questions
            )
            
            # Build message with both images
            messages = self._build_vqa_message(
                prompt,
                satellite_base64,
                street_base64
            )
            
            # Get response
            self.logger.log_step("Requesting VQA generation from LLM")
            response, stats = self.llm.send_message(
                messages,
                phase='vqa_generation',
                pair_id=session_id
            )
            
            # Parse and validate
            self.logger.log_step("Parsing JSON response")
            vqa_data = self._parse_json_response(response)
            
            self.logger.log_step("Validating VQA examples")
            validation_report = self.validator.validate(vqa_data)
            
            # Log validation report
            self.logger.log_validation_report(validation_report)
            
            # Log examples preview
            if 'vqa_examples' in vqa_data:
                examples = vqa_data['vqa_examples']
                self.logger.log_vqa_examples(examples, preview_count=3)
                self.logger.log_category_distribution(examples)
            
            # Log session end
            num_questions = len(vqa_data.get('vqa_examples', []))
            self.logger.end_vqa_generation(
                session_id,
                num_questions,
                success=True
            )
            
            result = {
                'session_id': session_id,
                'vqa_examples': vqa_data.get('vqa_examples', []),
                'validation': validation_report,
                'stats': stats,
                'raw_response': response,
                'source_images': {
                    'satellite': satellite_path,
                    'street_view': street_path
                },
                'annotations': {
                    'satellite': satellite_annotation,
                    'street': street_annotation
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.end_vqa_generation(
                session_id,
                0,
                success=False,
                error=str(e)
            )
            raise
    
    def _build_vqa_message(
        self,
        prompt: str,
        satellite_base64: str,
        street_base64: str
    ) -> List[Dict]:
        """Build message with prompt and both images.
        
        Args:
            prompt: Rendered VQA generation prompt
            satellite_base64: Base64-encoded satellite image
            street_base64: Base64-encoded street view image
            
        Returns:
            Message list for LLM
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{satellite_base64}"
                        }
                    },
                    {"type": "text", "text": "[Satellite View ↑]"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{street_base64}"
                        }
                    },
                    {"type": "text", "text": "[Street View ↑]"}
                ]
            }
        ]
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response.
        
        Handles markdown code blocks and extraction.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            ValueError: If JSON parsing fails
        """
        try:
            # Try direct parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try extracting from markdown code blocks
        json_str = response
        
        if "```json" in response:
            # Extract from ```json ... ```
            parts = response.split("```json")
            if len(parts) > 1:
                json_str = parts[1].split("```")[0].strip()
        elif "```" in response:
            # Extract from ``` ... ```
            parts = response.split("```")
            if len(parts) > 2:
                json_str = parts[1].strip()
        
        # Try parsing extracted string
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse JSON from LLM response: {e}\n"
                f"Response preview: {response[:200]}..."
            )
    
    def generate_from_labeling_result(
        self,
        labeling_result: Dict[str, Any],
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate VQA from a labeling pipeline result.
        
        Convenience method that extracts annotations from labeling result.
        
        Args:
            labeling_result: Result dictionary from labeling pipeline
            session_id: Optional session identifier
            
        Returns:
            VQA generation result
        """
        if session_id is None:
            session_id = labeling_result.get('pair_id', 'unknown')
        
        # Extract paths and annotations
        satellite_path = labeling_result['satellite']['image_path']
        street_path = labeling_result['street_view']['image_path']
        satellite_annotation = labeling_result['satellite']['annotation']
        street_annotation = labeling_result['street_view']['annotation']
        
        return self.generate(
            satellite_path,
            street_path,
            satellite_annotation,
            street_annotation,
            session_id
        )
    
    def batch_generate(
        self,
        labeling_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate VQA for multiple labeling results.
        
        Args:
            labeling_results: List of labeling result dictionaries
            
        Returns:
            List of VQA generation results
        """
        results = []
        
        for i, labeling_result in enumerate(labeling_results, 1):
            self.logger.logger.info(
                f"\nGenerating VQA {i}/{len(labeling_results)}"
            )
            
            try:
                result = self.generate_from_labeling_result(labeling_result)
                results.append(result)
            except Exception as e:
                self.logger.logger.error(
                    f"Failed to generate VQA for pair {labeling_result.get('pair_id')}: {e}"
                )
                results.append({
                    'error': str(e),
                    'pair_id': labeling_result.get('pair_id')
                })
        
        return results

