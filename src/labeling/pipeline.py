"""Labeling pipeline orchestrator.

This module orchestrates the complete labeling pipeline, coordinating
input handling, annotation, and result saving.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from src.core.config import Config
from src.llm.client import LLMClient
from src.logging.labeling_logger import LabelingLogger
from src.prompts.renderer import PromptRenderer
from .input_handler import InputHandler, ImagePair
from .annotator import ImageAnnotator


class LabelingPipeline:
    """Orchestrates the 2-phase image labeling process.
    
    This pipeline coordinates:
    - Loading image pairs from input structure
    - Annotating satellite images (Phase 1)
    - Annotating street view images (Phase 2)
    - Saving results with collision prevention
    
    The pipeline maintains conversation context between satellite
    and street view annotations for improved consistency.
    
    Attributes:
        config: Configuration object
        llm_client: LLM client for API calls
        logger: Labeling logger instance
        prompt_renderer: Prompt renderer
        annotator: Image annotator
    """
    
    def __init__(
        self,
        config: Config,
        llm_client: LLMClient,
        logger: LabelingLogger,
        prompt_renderer: PromptRenderer
    ):
        """Initialize labeling pipeline.
        
        Args:
            config: Configuration object
            llm_client: Configured LLM client
            logger: Labeling logger
            prompt_renderer: Prompt renderer with templates
        """
        self.config = config
        self.llm = llm_client
        self.logger = logger
        self.prompts = prompt_renderer
        self.annotator = ImageAnnotator(llm_client, prompt_renderer, logger)
    
    def run(self, input_handler: InputHandler) -> Dict[str, Any]:
        """Process all pairs from input handler.
        
        Args:
            input_handler: Input handler with image pairs
            
        Returns:
            Dictionary with overall results and statistics
        """
        # Log pipeline start
        self.logger.log_pipeline_start({
            'provider': self.llm.provider_name,
            'model': self.llm.provider.model,
            'total_pairs': len(input_handler),
            'input_file': str(input_handler.input_file)
        })
        
        results = []
        failed_pairs = []
        
        # Process each pair
        for pair in input_handler.iter_pairs():
            try:
                result = self.process_pair(pair, input_handler)
                results.append(result)
            except Exception as e:
                self.logger.logger.error(
                    f"Failed to process pair {pair.id}: {e}"
                )
                failed_pairs.append({
                    'pair_id': pair.id,
                    'error': str(e)
                })
        
        # Log statistics
        self.logger.log_statistics_summary()
        
        # Log pipeline end
        success = len(failed_pairs) == 0
        error_msg = None if success else f"{len(failed_pairs)} pair(s) failed"
        
        self.logger.log_pipeline_end(success=success, error=error_msg)
        
        return {
            'results': results,
            'failed_pairs': failed_pairs,
            'statistics': self.logger.get_statistics(),
            'run_summary': self.logger.get_run_summary()
        }
    
    def process_pair(
        self,
        pair: ImagePair,
        input_handler: InputHandler
    ) -> Dict[str, Any]:
        """Process a single image pair.
        
        Args:
            pair: Image pair to process
            input_handler: Input handler for output path resolution
            
        Returns:
            Result dictionary with annotations
        """
        # Log pair start
        pair_info = {
            'satellite': pair.satellite,
            'street_view': pair.street_view,
            'metadata': pair.metadata
        }
        self.logger.start_pair(pair.id, pair_info)
        
        # Initialize conversation history
        conversation = []
        
        try:
            # Phase 1: Satellite annotation
            self.logger.log_satellite_phase()
            satellite_result = self.annotator.annotate_satellite(
                pair.satellite,
                conversation,
                pair.id
            )
            
            # Phase 2: Street view annotation
            self.logger.log_street_view_phase()
            street_result = self.annotator.annotate_street_view(
                pair.street_view,
                conversation,
                pair.id
            )
            
            # Build complete result
            result = {
                'pair_id': pair.id,
                'satellite': satellite_result,
                'street_view': street_result,
                'metadata': pair.metadata,
                'conversation_length': len(conversation)
            }
            
            # Save result to file
            output_path = self._save_result(pair, result, input_handler)
            result['output_file'] = str(output_path)
            
            # Log pair end
            self.logger.end_pair(pair.id, result, success=True)
            
            return result
            
        except Exception as e:
            # Log failure
            self.logger.logger.error(f"Error processing pair {pair.id}: {e}")
            self.logger.end_pair(
                pair.id,
                {'error': str(e)},
                success=False
            )
            raise
    
    def _save_result(
        self,
        pair: ImagePair,
        result: Dict[str, Any],
        input_handler: InputHandler
    ) -> Path:
        """Save labeling result to file.
        
        Args:
            pair: Image pair
            result: Complete labeling result
            input_handler: Input handler for path resolution
            
        Returns:
            Path to saved file
        """
        # Get unique output path
        output_path = input_handler.get_output_path(pair, ensure_unique=True)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save result using logger
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        self.logger.logger.info(f"Saved result to: {output_path}")
        
        return output_path
    
    def process_single_pair(
        self,
        pair_id: str,
        input_handler: InputHandler
    ) -> Dict[str, Any]:
        """Process a single pair by ID.
        
        Args:
            pair_id: ID of pair to process
            input_handler: Input handler
            
        Returns:
            Result dictionary
        """
        pair = input_handler.get_pair(pair_id)
        return self.process_pair(pair, input_handler)

