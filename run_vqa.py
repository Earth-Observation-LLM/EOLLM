#!/usr/bin/env python3
"""Entry point for the VQA generation pipeline.

This script generates Visual Question Answering examples from annotated
satellite and street view images.

Usage:
    python run_vqa.py [--config CONFIG] [OPTIONS]

Modes:
    1. Direct mode: Provide image paths and annotations directly
    2. From labeling: Use labeling result JSON files
    3. Batch mode: Process multiple labeling results

Examples:
    # Generate VQA from labeling result
    python run_vqa.py --from-labeling logs/labeling/results/pair_001_labeling.json

    # Batch process all labeling results
    python run_vqa.py --batch logs/labeling/results/*.json

    # Direct mode with image paths
    python run_vqa.py --satellite sat.png --street street.png \
        --sat-annotation "Commercial area..." --street-annotation "Street view..."
"""

import sys
import json
import argparse
from pathlib import Path
from glob import glob

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.config import load_config, get_provider_config
from src.llm.client import LLMClient
from src.logging.vqa_logger import VQALogger
from src.prompts.renderer import PromptRenderer
from src.vqa.generator import VQAGenerator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate VQA examples from satellite and street view images"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration YAML file (default: config/settings.yaml)'
    )
    
    parser.add_argument(
        '--provider',
        type=str,
        default=None,
        help='Override LLM provider from config (e.g., groq, ollama)'
    )
    
    # Mode 1: From labeling result
    parser.add_argument(
        '--from-labeling',
        type=str,
        default=None,
        help='Path to labeling result JSON file'
    )
    
    # Mode 2: Batch mode
    parser.add_argument(
        '--batch',
        type=str,
        default=None,
        help='Glob pattern for batch processing labeling results (e.g., logs/labeling/results/*.json)'
    )
    
    # Mode 3: Direct mode
    parser.add_argument(
        '--satellite',
        type=str,
        default=None,
        help='Path to satellite image (direct mode)'
    )
    
    parser.add_argument(
        '--street',
        type=str,
        default=None,
        help='Path to street view image (direct mode)'
    )
    
    parser.add_argument(
        '--sat-annotation',
        type=str,
        default=None,
        help='Satellite annotation text (direct mode)'
    )
    
    parser.add_argument(
        '--street-annotation',
        type=str,
        default=None,
        help='Street view annotation text (direct mode)'
    )
    
    parser.add_argument(
        '--session-id',
        type=str,
        default=None,
        help='Custom session ID (optional)'
    )
    
    return parser.parse_args()


def load_labeling_result(file_path: str) -> dict:
    """Load a labeling result from JSON file.
    
    Args:
        file_path: Path to labeling result JSON
        
    Returns:
        Labeling result dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_vqa_result(result: dict, output_dir: Path) -> Path:
    """Save VQA result to file.
    
    Args:
        result: VQA generation result
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    session_id = result.get('session_id', 'unknown')
    output_file = output_dir / f"{session_id}_vqa.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return output_file


def main():
    """Main entry point for VQA pipeline."""
    args = parse_arguments()
    
    print("="*80)
    print("EOLLM VQA GENERATION PIPELINE")
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
        log_config = config.logging.vqa
        logger = VQALogger(
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
        print()
        
        # Initialize VQA generator
        generator = VQAGenerator(
            config=config,
            llm_client=llm_client,
            prompt_renderer=prompt_renderer,
            logger=logger
        )
        print("✓ VQA generator initialized")
        print()
        
        # Determine mode and process
        results = []
        
        if args.batch:
            # Batch mode
            print(f"BATCH MODE: Processing labeling results")
            print(f"Pattern: {args.batch}")
            print("="*80)
            
            files = glob(args.batch)
            if not files:
                print(f"ERROR: No files found matching pattern: {args.batch}")
                sys.exit(1)
            
            print(f"Found {len(files)} labeling result(s)\n")
            
            for i, file_path in enumerate(files, 1):
                print(f"\n[{i}/{len(files)}] Processing: {Path(file_path).name}")
                try:
                    labeling_result = load_labeling_result(file_path)
                    result = generator.generate_from_labeling_result(labeling_result)
                    results.append(result)
                    
                    # Save result
                    output_file = save_vqa_result(result, logger.results_dir)
                    print(f"✓ Saved VQA result to: {output_file}")
                    
                except Exception as e:
                    print(f"✗ Failed: {e}")
                    logger.logger.error(f"Failed to process {file_path}: {e}")
        
        elif args.from_labeling:
            # From labeling result
            print(f"FROM LABELING MODE")
            print(f"Input: {args.from_labeling}")
            print("="*80)
            
            labeling_result = load_labeling_result(args.from_labeling)
            result = generator.generate_from_labeling_result(
                labeling_result,
                session_id=args.session_id
            )
            results.append(result)
            
            # Save result
            output_file = save_vqa_result(result, logger.results_dir)
            print(f"\n✓ VQA result saved to: {output_file}")
        
        elif args.satellite and args.street:
            # Direct mode
            print(f"DIRECT MODE")
            print(f"Satellite: {args.satellite}")
            print(f"Street view: {args.street}")
            print("="*80)
            
            if not args.sat_annotation or not args.street_annotation:
                print("ERROR: Direct mode requires --sat-annotation and --street-annotation")
                sys.exit(1)
            
            result = generator.generate(
                satellite_path=args.satellite,
                street_path=args.street,
                satellite_annotation=args.sat_annotation,
                street_annotation=args.street_annotation,
                session_id=args.session_id
            )
            results.append(result)
            
            # Save result
            output_file = save_vqa_result(result, logger.results_dir)
            print(f"\n✓ VQA result saved to: {output_file}")
        
        else:
            print("ERROR: Must specify one of:")
            print("  --from-labeling FILE")
            print("  --batch PATTERN")
            print("  --satellite IMG --street IMG --sat-annotation TEXT --street-annotation TEXT")
            sys.exit(1)
        
        # Log statistics
        logger.log_statistics_summary()
        
        # Print summary
        print()
        print("="*80)
        print("VQA GENERATION SUMMARY")
        print("="*80)
        stats = logger.get_statistics()
        print(f"Sessions completed: {stats.get('vqa_sessions', 0)}")
        print(f"Total questions generated: {stats.get('total_questions_generated', 0)}")
        print(f"Failed generations: {stats.get('failed_generations', 0)}")
        print(f"Success rate: {stats.get('success_rate', 0):.1%}")
        print(f"Avg questions/session: {stats.get('avg_questions_per_session', 0):.1f}")
        print()
        print(f"Run log: {logger.run_log_file}")
        print(f"Results directory: {logger.results_dir}")
        print("="*80)
        
        # Log pipeline end
        logger.log_pipeline_end(
            success=stats.get('failed_generations', 0) == 0
        )
        
        # Exit with appropriate code
        sys.exit(0 if stats.get('failed_generations', 0) == 0 else 1)
        
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

