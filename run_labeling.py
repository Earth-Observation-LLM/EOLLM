#!/usr/bin/env python3
"""Entry point for the image labeling pipeline.

This script runs the labeling pipeline on image pairs defined in an
input structure JSON file.

Usage:
    python run_labeling.py [--config CONFIG] [--input INPUT] [--pair-id PAIR_ID]

Examples:
    # Process all pairs in input structure
    python run_labeling.py

    # Use custom config and input files
    python run_labeling.py --config config/settings.yaml --input my_inputs.json

    # Process a single pair
    python run_labeling.py --pair-id timko_2015_comparison
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import load_config, get_provider_config
from src.core.image_utils import encode_image_to_base64
from src.llm.client import LLMClient
from src.logging.labeling_logger import LabelingLogger
from src.prompts.renderer import PromptRenderer
from src.labeling.input_handler import InputHandler
from src.labeling.pipeline import LabelingPipeline


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run image labeling pipeline on satellite and street view pairs"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration YAML file (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='input_structure.json',
        help='Path to input structure JSON file (default: input_structure.json)'
    )
    
    parser.add_argument(
        '--pair-id',
        type=str,
        default=None,
        help='Process only a specific pair by ID (optional)'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        default=None,
        help='Override LLM provider from config (e.g., groq, ollama)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for labeling pipeline."""
    args = parse_arguments()
    
    print("="*80)
    print("EOLLM IMAGE LABELING PIPELINE")
    print("="*80)
    print()
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        print("✓ Configuration loaded successfully")
        print()
        
        # Determine provider
        provider_name = args.provider or config.llm.default_provider
        print(f"Using LLM provider: {provider_name}")
        provider_config = get_provider_config(config, provider_name)
        print(f"Model: {provider_config.model}")
        print()
        
        # Initialize logger
        log_config = config.logging.labeling
        logger = LabelingLogger(
            output_dir=str(config.resolve_path(log_config.output_dir)),
            console_level=log_config.levels.console,
            file_level=log_config.levels.file
        )
        print(f"✓ Logger initialized (Run ID: {logger.run_id})")
        print(f"  Logs: {logger.runs_dir}")
        print()
        
        # Initialize LLM client
        llm_client = LLMClient(
            provider_name=provider_name,
            config=provider_config,
            logger=logger
        )
        print(f"✓ LLM client initialized")
        print()
        
        # Initialize prompt renderer
        templates_dir = config.resolve_path(config.paths.templates)
        prompt_renderer = PromptRenderer(str(templates_dir))
        print(f"✓ Prompt renderer initialized")
        print(f"  Templates: {templates_dir}")
        print()
        
        # Load input structure
        input_file = config.resolve_path(args.input)
        print(f"Loading input structure from: {input_file}")
        input_handler = InputHandler(
            str(input_file),
            project_root=config._project_root
        )
        print(f"✓ Input structure loaded")
        print(f"  Total pairs: {len(input_handler)}")
        print()
        
        # Initialize pipeline
        pipeline = LabelingPipeline(
            config=config,
            llm_client=llm_client,
            logger=logger,
            prompt_renderer=prompt_renderer
        )
        print("✓ Pipeline initialized")
        print()
        
        # Run pipeline
        if args.pair_id:
            print(f"Processing single pair: {args.pair_id}")
            print("="*80)
            result = pipeline.process_single_pair(args.pair_id, input_handler)
            results = {
                'results': [result],
                'failed_pairs': [],
                'statistics': logger.get_statistics()
            }
        else:
            print(f"Processing all {len(input_handler)} pair(s)...")
            print("="*80)
            results = pipeline.run(input_handler)
        
        # Print summary
        print()
        print("="*80)
        print("LABELING PIPELINE SUMMARY")
        print("="*80)
        stats = results.get('statistics', {})
        print(f"Pairs processed: {stats.get('pairs_processed', 0)}")
        print(f"Pairs succeeded: {stats.get('pairs_succeeded', 0)}")
        print(f"Pairs failed: {stats.get('pairs_failed', 0)}")
        print(f"Total LLM calls: {stats.get('total_llm_calls', 0)}")
        print(f"Success rate: {stats.get('success_rate', 0):.1%}")
        print()
        print(f"Run log: {logger.run_log_file}")
        print(f"Results directory: {logger.results_dir}")
        print("="*80)
        
        # Exit with appropriate code
        sys.exit(0 if stats.get('pairs_failed', 0) == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

