"""VQA pipeline logger.

This module provides specialized logging for the VQA generation pipeline,
including question validation tracking and category distribution logging.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_logger import BaseLogger


class VQALogger(BaseLogger):
    """Logger for VQA generation pipeline.
    
    Extends BaseLogger with VQA-specific functionality:
    - VQA generation tracking
    - Question quality metrics
    - Category distribution logging
    """
    
    def __init__(
        self,
        output_dir: str = "logs/vqa",
        console_level: str = "INFO",
        file_level: str = "DEBUG"
    ):
        """Initialize VQA logger.
        
        Args:
            output_dir: Directory for VQA logs
            console_level: Console log level
            file_level: File log level
        """
        super().__init__(output_dir, console_level, file_level)
        
        # Track VQA generation statistics
        self.vqa_sessions = 0
        self.total_questions_generated = 0
        self.failed_generations = 0
    
    def get_pipeline_type(self) -> str:
        """Get pipeline type identifier.
        
        Returns:
            'vqa'
        """
        return "vqa"
    
    def start_vqa_generation(
        self,
        session_id: str,
        source_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log the start of VQA generation.
        
        Args:
            session_id: Unique identifier for this generation session
            source_info: Information about source images/annotations
        """
        self.vqa_sessions += 1
        
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"VQA GENERATION SESSION: {session_id}")
        self.logger.info("=" * 80)
        
        if source_info:
            for key, value in source_info.items():
                self.logger.info(f"  {key}: {value}")
    
    def end_vqa_generation(
        self,
        session_id: str,
        num_questions: int,
        success: bool = True,
        error: Optional[str] = None
    ) -> None:
        """Log the completion of VQA generation.
        
        Args:
            session_id: Session identifier
            num_questions: Number of questions generated
            success: Whether generation succeeded
            error: Error message if failed
        """
        if success:
            self.total_questions_generated += num_questions
            self.logger.info(f"Generated {num_questions} VQA questions")
        else:
            self.failed_generations += 1
            self.logger.error(f"VQA generation failed: {error}")
        
        self.logger.info("=" * 80)
    
    def log_vqa_examples(
        self,
        examples: List[Dict[str, Any]],
        preview_count: int = 3
    ) -> None:
        """Log VQA examples (preview).
        
        Args:
            examples: List of VQA example dictionaries
            preview_count: Number of examples to preview in logs
        """
        self.logger.info(f"\nGenerated {len(examples)} VQA examples")
        self.logger.info("Example questions (first {}):"
                        .format(min(preview_count, len(examples))))
        
        for i, example in enumerate(examples[:preview_count], 1):
            self.logger.info(f"\n  Question {i}:")
            self.logger.info(f"    Q: {example.get('question', 'N/A')}")
            self.logger.info(f"    A: {example.get('answer', 'N/A')}")
            self.logger.info(f"    Category: {example.get('category', 'N/A')}")
    
    def log_validation_report(self, validation_report: Dict[str, Any]) -> None:
        """Log VQA validation report.
        
        Args:
            validation_report: Validation report from VQAValidator
        """
        self.logger.info("\nVQA VALIDATION REPORT:")
        
        is_valid = validation_report.get('is_valid', False)
        if is_valid:
            self.logger.info("  Status: VALID ✓")
        else:
            self.logger.warning("  Status: ISSUES FOUND")
        
        # Log warnings
        warnings = validation_report.get('warnings', [])
        if warnings:
            self.logger.warning(f"  Warnings ({len(warnings)}):")
            for warning in warnings:
                self.logger.warning(f"    - {warning}")
        
        # Log errors
        errors = validation_report.get('errors', [])
        if errors:
            self.logger.error(f"  Errors ({len(errors)}):")
            for error in errors:
                self.logger.error(f"    - {error}")
        
        # Log statistics
        stats = validation_report.get('statistics', {})
        if stats:
            self.logger.info("  Statistics:")
            for key, value in stats.items():
                self.logger.info(f"    {key}: {value}")
        
        # Log category distribution
        category_dist = validation_report.get('category_distribution', {})
        if category_dist:
            self.logger.info("  Category distribution:")
            for category, count in category_dist.items():
                self.logger.info(f"    {category}: {count}")
    
    def log_category_distribution(
        self,
        examples: List[Dict[str, Any]]
    ) -> None:
        """Log category distribution of generated questions.
        
        Args:
            examples: List of VQA examples
        """
        # Count by category
        distribution: Dict[str, int] = {}
        for example in examples:
            category = example.get('category', 'unknown')
            distribution[category] = distribution.get(category, 0) + 1
        
        self.logger.info("\nCategory distribution:")
        for category, count in sorted(distribution.items()):
            percentage = (count / len(examples) * 100) if examples else 0
            self.logger.info(f"  {category}: {count} ({percentage:.1f}%)")
    
    def save_vqa_result(
        self,
        session_id: str,
        result_data: Dict[str, Any]
    ) -> Path:
        """Save VQA generation result.
        
        Args:
            session_id: Session identifier
            result_data: Complete VQA result
            
        Returns:
            Path to saved result file
        """
        return self.save_result(session_id, result_data, "vqa")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get VQA generation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'vqa_sessions': self.vqa_sessions,
            'total_questions_generated': self.total_questions_generated,
            'failed_generations': self.failed_generations,
            'success_rate': (
                (self.vqa_sessions - self.failed_generations) / self.vqa_sessions
                if self.vqa_sessions > 0 else 0
            ),
            'avg_questions_per_session': (
                self.total_questions_generated / self.vqa_sessions
                if self.vqa_sessions > 0 else 0
            ),
            'total_llm_calls': self.call_counter
        }
    
    def log_statistics_summary(self) -> None:
        """Log a summary of VQA statistics."""
        stats = self.get_statistics()
        
        self.logger.info("\nVQA GENERATION STATISTICS:")
        self.logger.info(f"  Sessions: {stats['vqa_sessions']}")
        self.logger.info(f"  Total questions: {stats['total_questions_generated']}")
        self.logger.info(f"  Failed generations: {stats['failed_generations']}")
        self.logger.info(f"  Success rate: {stats['success_rate']:.1%}")
        self.logger.info(
            f"  Avg questions/session: {stats['avg_questions_per_session']:.1f}"
        )
        self.logger.info(f"  Total LLM calls: {stats['total_llm_calls']}")

