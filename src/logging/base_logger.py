"""Base logging infrastructure for EOLLM pipelines.

This module provides the foundation for research-grade logging with
comprehensive tracking of LLM calls, execution phases, and results.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """Abstract base class for pipeline loggers.
    
    This class provides common logging functionality for both labeling
    and VQA pipelines, including:
    - Dual-format logging (human-readable + structured JSON)
    - LLM call tracking with detailed statistics
    - Phase-based organization
    - Automatic log rotation
    
    Attributes:
        run_id: Unique identifier for this pipeline run
        output_dir: Base output directory for logs
        run_log_file: Path to human-readable run log
        llm_calls_dir: Directory for per-call LLM logs
        results_dir: Directory for final results
        call_counter: Counter for LLM calls in this run
    """
    
    def __init__(
        self,
        output_dir: str,
        console_level: str = "INFO",
        file_level: str = "DEBUG"
    ):
        """Initialize base logger.
        
        Args:
            output_dir: Base directory for log output
            console_level: Log level for console output
            file_level: Log level for file output
        """
        self.output_dir = Path(output_dir)
        self.run_id = self._generate_run_id()
        self.call_counter = 0
        
        # Create directory structure
        self.runs_dir = self.output_dir / "runs"
        self.llm_calls_dir = self.output_dir / "llm_calls"
        self.results_dir = self.output_dir / "results"
        
        for directory in [self.runs_dir, self.llm_calls_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Setup run log file
        self.run_log_file = self.runs_dir / f"{self.run_id}.log"
        
        # Setup Python logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.run_id}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, console_level.upper()))
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.run_log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Track run metadata
        self.run_metadata: Dict[str, Any] = {
            'run_id': self.run_id,
            'start_time': datetime.now().isoformat(),
            'pipeline_type': self.get_pipeline_type()
        }
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID based on timestamp.
        
        Returns:
            Run ID string in format YYYYMMDD_HHMMSS
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @abstractmethod
    def get_pipeline_type(self) -> str:
        """Get the type of pipeline (labeling/vqa).
        
        Returns:
            Pipeline type identifier
        """
        pass
    
    def log_pipeline_start(self, config_info: Dict[str, Any]) -> None:
        """Log the start of a pipeline run.
        
        Args:
            config_info: Configuration information to log
        """
        self.logger.info("=" * 80)
        self.logger.info(f"{self.get_pipeline_type().upper()} PIPELINE STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Run ID: {self.run_id}")
        
        for key, value in config_info.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("=" * 80)
        self.run_metadata['config'] = config_info
    
    def log_pipeline_end(self, success: bool = True, error: Optional[str] = None) -> None:
        """Log the end of a pipeline run.
        
        Args:
            success: Whether pipeline completed successfully
            error: Error message if pipeline failed
        """
        self.logger.info("=" * 80)
        if success:
            self.logger.info(f"{self.get_pipeline_type().upper()} PIPELINE COMPLETED SUCCESSFULLY")
        else:
            self.logger.error(f"{self.get_pipeline_type().upper()} PIPELINE FAILED")
            if error:
                self.logger.error(f"Error: {error}")
        
        self.logger.info("=" * 80)
        self.logger.info(f"Run log: {self.run_log_file}")
        self.logger.info(f"LLM calls logged: {self.call_counter}")
        self.logger.info("=" * 80)
        
        self.run_metadata['end_time'] = datetime.now().isoformat()
        self.run_metadata['success'] = success
        if error:
            self.run_metadata['error'] = error
        self.run_metadata['total_llm_calls'] = self.call_counter
    
    def log_llm_call(
        self,
        messages: List[Dict],
        response: Optional[str],
        stats: Dict[str, Any],
        phase: Optional[str] = None,
        pair_id: Optional[str] = None
    ) -> None:
        """Log a single LLM API call.
        
        Args:
            messages: Messages sent to LLM
            response: Response from LLM (None if failed)
            stats: Statistics dictionary from provider
            phase: Phase name (e.g., 'satellite_thinking')
            pair_id: ID of the pair being processed
        """
        self.call_counter += 1
        call_id = f"{self.run_id}_{self.call_counter:03d}"
        
        # Extract image paths from messages
        image_paths = self._extract_image_paths(messages)
        
        # Get prompt preview
        prompt_preview = self._get_prompt_preview(messages)
        
        # Log to console/file
        log_msg = f"LLM Call [{self.call_counter:03d}]"
        if phase:
            log_msg += f" [{phase}]"
        if pair_id:
            log_msg += f" [pair: {pair_id}]"
        
        latency = stats.get('latency_ms', 0)
        tokens = stats.get('tokens_used', -1)
        
        log_msg += f": {latency:.0f}ms"
        if tokens > 0:
            log_msg += f", {tokens} tokens"
        
        if stats.get('success', True):
            self.logger.info(log_msg)
        else:
            self.logger.error(f"{log_msg} - FAILED: {stats.get('error')}")
        
        # Save detailed JSON log
        call_log = {
            'call_id': call_id,
            'timestamp': stats.get('timestamp', datetime.now().isoformat()),
            'phase': phase,
            'pair_id': pair_id,
            'request': {
                'provider': stats.get('model', 'unknown').split('/')[0] if '/' in stats.get('model', '') else 'unknown',
                'model': stats.get('model', 'unknown'),
                'message_count': len(messages),
                'has_images': len(image_paths) > 0,
                'image_paths': image_paths,
                'prompt_preview': prompt_preview
            },
            'response': {
                'content': response,
                'content_length': len(response) if response else 0
            },
            'stats': stats
        }
        
        # Add tokens per second if available
        if tokens > 0 and latency > 0:
            call_log['stats']['tokens_per_second'] = tokens / (latency / 1000)
        
        # Save to file
        call_log_file = self.llm_calls_dir / f"{call_id}.json"
        with open(call_log_file, 'w', encoding='utf-8') as f:
            json.dump(call_log, f, indent=2, ensure_ascii=False)
    
    def _extract_image_paths(self, messages: List[Dict]) -> List[str]:
        """Extract image file paths from messages.
        
        Args:
            messages: Message list
            
        Returns:
            List of image paths (empty if none found)
        """
        # This is a placeholder - in practice, image paths would need to be
        # tracked separately as they're not in the base64 data
        return []
    
    def _get_prompt_preview(self, messages: List[Dict], length: int = 200) -> str:
        """Get a preview of the prompt from messages.
        
        Args:
            messages: Message list
            length: Maximum length of preview
            
        Returns:
            Truncated prompt text
        """
        for msg in reversed(messages):
            content = msg.get('content', '')
            
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                # Extract text from content list
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                text = ' '.join(text_parts)
            else:
                continue
            
            if text:
                if len(text) <= length:
                    return text
                return text[:length] + '...'
        
        return '[No text content]'
    
    def log_phase(self, phase_name: str, phase_number: Optional[int] = None) -> None:
        """Log the start of a processing phase.
        
        Args:
            phase_name: Name of the phase
            phase_number: Optional phase number
        """
        self.logger.info("")
        self.logger.info("-" * 80)
        if phase_number is not None:
            self.logger.info(f"PHASE {phase_number}: {phase_name}")
        else:
            self.logger.info(f"PHASE: {phase_name}")
        self.logger.info("-" * 80)
    
    def log_step(self, step_name: str, details: Optional[str] = None) -> None:
        """Log a processing step.
        
        Args:
            step_name: Name of the step
            details: Optional additional details
        """
        if details:
            self.logger.debug(f"{step_name}: {details}")
        else:
            self.logger.debug(step_name)
    
    def save_result(
        self,
        result_id: str,
        result_data: Dict[str, Any],
        result_type: str = "result"
    ) -> Path:
        """Save a result to the results directory.
        
        Args:
            result_id: Unique identifier for this result
            result_data: Result data to save
            result_type: Type of result (for filename)
            
        Returns:
            Path to saved result file
        """
        result_file = self.results_dir / f"{result_id}_{result_type}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Result saved: {result_file}")
        return result_file
    
    def get_run_summary(self) -> Dict[str, Any]:
        """Get summary of the current run.
        
        Returns:
            Dictionary with run summary
        """
        return self.run_metadata.copy()

