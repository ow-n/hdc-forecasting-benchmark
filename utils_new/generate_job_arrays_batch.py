#!/usr/bin/env python3
"""
Job Arrays Batch Generator for Time Series Benchmarking

This script creates separate job arrays for each model, allowing independent
submission, monitoring, and cancellation of different model experiments.

Usage:
    # Test DLinear first (as per your plan)
    python utils_new/generate_job_arrays_batch.py --models DLinear --tasks long_term_forecast short_term_forecast

    # Multiple specific models for comprehensive benchmark
    python utils_new/generate_job_arrays_batch.py --models DLinear TiDE TSMixer SegRNN --tasks long_term_forecast short_term_forecast

    # Preview what would be generated
    python utils_new/generate_job_arrays_batch.py --models DLinear TiDE --tasks long_term_forecast --dry-run
    python utils_new/generate_job_arrays_batch.py --models DLinear --tasks short_term_forecast long_term_forecast --dry-run


Recommended Usage (One command per model for independent monitoring):
    python utils_new/generate_job_arrays_batch.py --models DLinear --tasks short_term_forecast long_term_forecast
    python utils_new/generate_job_arrays_batch.py --models TiDE --tasks short_term_forecast long_term_forecast
    python utils_new/generate_job_arrays_batch.py --models TSMixer --tasks short_term_forecast long_term_forecast
    python utils_new/generate_job_arrays_batch.py --models SegRNN --tasks short_term_forecast long_term_forecast
    python utils_new/generate_job_arrays_batch.py --models TimesNet --tasks short_term_forecast long_term_forecast
    python utils_new/generate_job_arrays_batch.py --models PatchTST --tasks short_term_forecast long_term_forecast
"""

import argparse
import subprocess
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from config import (
    DEFAULT_CONCURRENT_PER_MODEL, JOB_TIMEOUT_SECONDS, MODEL_CONCURRENT_LIMITS,
    DEFAULT_TASKS, SUPPORTED_MODELS, SUPPORTED_TASKS, GENERATOR_SCRIPT,
    EXPERIMENT_CONFIGS
)


class JobArraysBatchGenerator:
    """Generates multiple model job arrays using separate job array scripts."""
    
    def __init__(self):
        self.generator_script = GENERATOR_SCRIPT
        # Use experiment configurations from config.py
        self.experiment_configs = EXPERIMENT_CONFIGS

    def validate_environment(self) -> bool:
        """Check if generator script exists and is executable."""
        if not Path(self.generator_script).exists():
            print(f"ERROR: Generator script not found: {self.generator_script}")
            return False
        
        # Test generator script
        try:
            result = subprocess.run([
                'python', self.generator_script, '--model', 'DLinear', 
                '--task', 'long_term_forecast', '--preview'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"ERROR: Generator script failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("ERROR: Generator script timed out")
            return False
        except Exception as e:
            print(f"ERROR: Failed to test generator script: {e}")
            return False
        
        return True

    def generate_model_job_array(self, model: str, tasks: List[str], 
                                datasets: List[str] = None, base_output_dir: Path = None,
                                dry_run: bool = False) -> Dict[str, Any]:
        """Generate job array for a specific model."""
        
        if base_output_dir is None:
            base_output_dir = Path('./experiments')
        
        # Create model-specific directory
        model_dir = base_output_dir / f"{model.lower()}_experiments"
        
        config = self.experiment_configs.get(model, {})
        max_concurrent = config.get('max_concurrent', DEFAULT_CONCURRENT_PER_MODEL)
        
        # Build command
        cmd = [
            'python', self.generator_script,
            '--model', model,
            '--task'] + tasks + [
            '--output-dir', str(model_dir),
            '--max-concurrent', str(max_concurrent)
        ]
        
        if datasets:
            cmd.extend(['--datasets'] + datasets)
        
        print(f"\n=== Generating {model} Job Array ===")
        print(f"Tasks: {tasks}")
        if datasets:
            print(f"Datasets: {datasets}")
        print(f"Output directory: {model_dir}")
        print(f"Max concurrent jobs: {max_concurrent}")
        print(f"Description: {config.get('description', 'No description')}")
        
        if dry_run:
            print(f"DRY RUN - Would execute: {' '.join(cmd)}")
            return {
                'model': model,
                'status': 'dry_run',
                'command': ' '.join(cmd),
                'output_dir': str(model_dir)
            }
        
        # Execute generator
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=JOB_TIMEOUT_SECONDS)
            
            if result.returncode == 0:
                print(f"SUCCESS: Generated {model} job array")
                print(result.stdout)
                
                return {
                    'model': model,
                    'status': 'success',
                    'output_dir': str(model_dir),
                    'max_concurrent': max_concurrent,
                    'command': ' '.join(cmd),
                    'stdout': result.stdout
                }
            else:
                print(f"ERROR: Failed to generate {model} job array")
                print(f"Error: {result.stderr}")
                
                return {
                    'model': model,
                    'status': 'failed',
                    'error': result.stderr,
                    'command': ' '.join(cmd)
                }
                
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {model} job array generation timed out")
            return {
                'model': model,
                'status': 'timeout',
                'command': ' '.join(cmd)
            }
        except Exception as e:
            print(f"EXCEPTION: Failed to generate {model} job array: {e}")
            return {
                'model': model,
                'status': 'exception',
                'error': str(e),
                'command': ' '.join(cmd)
            }

    def generate_all_job_arrays(self, models: List[str], tasks: List[str],
                              datasets: List[str] = None, output_dir: Path = None,
                              dry_run: bool = False) -> Dict[str, Any]:
        """Generate job arrays for all specified models."""
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f'./experiments_{timestamp}')
        
        print(f"Starting Experiment Orchestration")
        print(f"Models: {models}")
        print(f"Tasks: {tasks}")
        if datasets:
            print(f"Datasets: {datasets}")
        print(f"Base output directory: {output_dir}")
        print(f"Dry run: {dry_run}")
        
        # Create base directory
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate job arrays for each model
        results = {}
        successful_models = []
        failed_models = []
        
        for model in models:
            if model not in self.experiment_configs:
                print(f"WARNING: Unknown model {model}, using default settings")
            
            result = self.generate_model_job_array(
                model=model,
                tasks=tasks,
                datasets=datasets,
                base_output_dir=output_dir,
                dry_run=dry_run
            )
            
            results[model] = result
            
            if result['status'] == 'success':
                successful_models.append(model)
            elif result['status'] != 'dry_run':
                failed_models.append(model)
        
        # Generate orchestration summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models': models,
            'tasks': tasks,
            'datasets': datasets,
            'output_dir': str(output_dir),
            'dry_run': dry_run,
            'results': results,
            'successful_models': successful_models,
            'failed_models': failed_models,
            'total_models': len(models),
            'success_count': len(successful_models),
            'failure_count': len(failed_models)
        }
        
        # Save summary
        if not dry_run:
            summary_file = output_dir / 'orchestration_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nOrchestration summary saved to: {summary_file}")
        
        # Skip script generation - using Job Scheduler UI for deployment
        
        return summary

    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted summary of orchestration results."""
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT ORCHESTRATION SUMMARY")
        print(f"{'='*60}")
        
        print(f"Overview:")
        print(f"   Total models: {summary['total_models']}")
        print(f"   Successful: {summary['success_count']}")
        print(f"   Failed: {summary['failure_count']}")
        print(f"   Tasks: {', '.join(summary['tasks'])}")
        
        if summary['successful_models']:
            print(f"\nSuccessful Models:")
            for model in summary['successful_models']:
                result = summary['results'][model]
                output_dir = result.get('output_dir', 'Unknown')
                max_concurrent = result.get('max_concurrent', 'Unknown')
                print(f"   • {model:12} -> {output_dir} (max concurrent: {max_concurrent})")
        
        if summary['failed_models']:
            print(f"\nFailed Models:")
            for model in summary['failed_models']:
                result = summary['results'][model]
                error = result.get('error', result.get('status', 'Unknown error'))
                print(f"   • {model:12} -> {error}")
        
        if not summary['dry_run'] and summary['successful_models']:
            print(f"\nNext Steps:")
            print(f"   1. Upload job array directories to HPC using Job Scheduler UI")
            print(f"   2. Submit individual job arrays (run_job_array.sh) via UI")
            print(f"   3. Monitor progress through HPC dashboard")
        
        print(f"\nOutput directory: {summary['output_dir']}")
        print(f"Generated: {summary['timestamp']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate separate job arrays for multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate separate job arrays for DLinear and TiDE for long-term forecasting
  python utils_new/generate_job_arrays_batch.py --models DLinear TiDE --tasks long_term_forecast
  
  # Multiple linear models for both tasks (creates separate job arrays per model)
  python utils_new/generate_job_arrays_batch.py --models DLinear TiDE TSMixer SegRNN --tasks long_term_forecast short_term_forecast
  
  # Dry run to see what would be generated
  python utils_new/generate_job_arrays_batch.py --models DLinear TiDE --tasks long_term_forecast --dry-run
        """
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=SUPPORTED_MODELS,
        default=SUPPORTED_MODELS,
        help='Models to generate job arrays for (default: all models)'
    )
    
    parser.add_argument(
        '--tasks',
        nargs='+',
        required=True,
        choices=SUPPORTED_TASKS,
        help='Tasks to include in experiments'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets for long-term forecasting (default: all)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Base output directory (default: auto-generated with timestamp)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without creating files'
    )
    
    args = parser.parse_args()
    
    # Initialize job arrays batch generator
    orchestrator = JobArraysBatchGenerator()
    
    # Validate environment
    if not orchestrator.validate_environment():
        print("❌ Environment validation failed")
        return 1
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Generate all job arrays
    summary = orchestrator.generate_all_job_arrays(
        models=args.models,
        tasks=args.tasks,
        datasets=args.datasets,
        output_dir=output_dir,
        dry_run=args.dry_run
    )
    
    # Print summary
    orchestrator.print_summary(summary)
    
    # Return appropriate exit code
    if summary['failure_count'] > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit(main())