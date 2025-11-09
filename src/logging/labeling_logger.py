"""Labeling pipeline logger.

This module provides specialized logging for the image labeling pipeline,
including pair-level tracking and annotation logging.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from .base_logger import BaseLogger


class LabelingLogger(BaseLogger):
    """Logger for image labeling pipeline.
    
    Extends BaseLogger with labeling-specific functionality:
    - Pair-level logging
    - Annotation tracking
    - Phase-specific metrics (satellite vs street view)
    """
    
    def __init__(
        self,
        output_dir: str = "logs/labeling",
        console_level: str = "INFO",
        file_level: str = "DEBUG"
    ):
        """Initialize labeling logger.
        
        Args:
            output_dir: Directory for labeling logs
            console_level: Console log level
            file_level: File log level
        """
        super().__init__(output_dir, console_level, file_level)
        
        # Track current pair being processed
        self.current_pair_id: Optional[str] = None
        self.pair_start_time: Optional[str] = None
        
        # Track pair-level statistics
        self.pairs_processed = 0
        self.pairs_succeeded = 0
        self.pairs_failed = 0
    
    def get_pipeline_type(self) -> str:
        """Get pipeline type identifier.
        
        Returns:
            'labeling'
        """
        return "labeling"
    
    def start_pair(self, pair_id: str, pair_info: Optional[Dict[str, Any]] = None) -> None:
        """Log the start of processing an image pair.
        
        Args:
            pair_id: Unique identifier for the pair
            pair_info: Optional metadata about the pair
        """
        from datetime import datetime
        
        self.current_pair_id = pair_id
        self.pair_start_time = datetime.now().isoformat()
        self.pairs_processed += 1
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"PROCESSING PAIR: {pair_id}")
        self.logger.info("=" * 80)
        
        if pair_info:
            for key, value in pair_info.items():
                self.logger.info(f"  {key}: {value}")
    
    def end_pair(
        self,
        pair_id: str,
        result: Dict[str, Any],
        success: bool = True
    ) -> None:
        """Log the completion of processing an image pair.
        
        Args:
            pair_id: Pair identifier
            result: Processing result dictionary
            success: Whether processing succeeded
        """
        from datetime import datetime
        
        if success:
            self.pairs_succeeded += 1
            self.logger.info(f"Pair {pair_id} completed successfully")
        else:
            self.pairs_failed += 1
            self.logger.error(f"Pair {pair_id} failed")
        
        # Log summary stats
        if 'satellite' in result and 'stats' in result['satellite']:
            sat_stats = result['satellite']['stats']
            self.logger.debug(
                f"Satellite annotation: {sat_stats.get('latency_ms', 0):.0f}ms"
            )
        
        if 'street_view' in result and 'stats' in result['street_view']:
            street_stats = result['street_view']['stats']
            self.logger.debug(
                f"Street view annotation: {street_stats.get('latency_ms', 0):.0f}ms"
            )
        
        self.logger.info("=" * 80)
        
        self.current_pair_id = None
        self.pair_start_time = None
    
    def log_annotation(
        self,
        annotation_type: str,
        thinking: str,
        annotation: str
    ) -> None:
        """Log an annotation with its thinking process.
        
        Args:
            annotation_type: Type of annotation ('satellite' or 'street_view')
            thinking: Thinking/reasoning text
            annotation: Final annotation text
        """
        self.logger.debug(f"\n[{annotation_type.upper()} THINKING]")
        self.logger.debug("-" * 80)
        self.logger.debug(thinking[:500] + ("..." if len(thinking) > 500 else ""))
        self.logger.debug("-" * 80)
        
        self.logger.info(f"\n[{annotation_type.upper()} ANNOTATION]")
        self.logger.info("-" * 80)
        self.logger.info(annotation)
        self.logger.info("-" * 80)
    
    def log_satellite_phase(self) -> None:
        """Log start of satellite image processing phase."""
        self.log_phase("Satellite Image Annotation", 1)
    
    def log_street_view_phase(self) -> None:
        """Log start of street view image processing phase."""
        self.log_phase("Street View Image Annotation", 2)
    
    def save_pair_result(
        self,
        pair_id: str,
        result_data: Dict[str, Any]
    ) -> Path:
        """Save labeling result for a pair.
        
        Args:
            pair_id: Pair identifier
            result_data: Complete labeling result
            
        Returns:
            Path to saved result file
        """
        return self.save_result(pair_id, result_data, "labeling")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get labeling statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'pairs_processed': self.pairs_processed,
            'pairs_succeeded': self.pairs_succeeded,
            'pairs_failed': self.pairs_failed,
            'success_rate': (
                self.pairs_succeeded / self.pairs_processed
                if self.pairs_processed > 0 else 0
            ),
            'total_llm_calls': self.call_counter
        }
    
    def log_statistics_summary(self) -> None:
        """Log a summary of labeling statistics."""
        stats = self.get_statistics()
        
        self.logger.info("\nLABELING STATISTICS:")
        self.logger.info(f"  Pairs processed: {stats['pairs_processed']}")
        self.logger.info(f"  Pairs succeeded: {stats['pairs_succeeded']}")
        self.logger.info(f"  Pairs failed: {stats['pairs_failed']}")
        self.logger.info(f"  Success rate: {stats['success_rate']:.1%}")
        self.logger.info(f"  Total LLM calls: {stats['total_llm_calls']}")

