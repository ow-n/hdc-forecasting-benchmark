#!/usr/bin/env python3
"""
Batch Job Array Launcher for Time Series Benchmarking

Sequentially generates job arrays for all specified models:
- DLinear, TiDE, TSMixer, SegRNN, TimesNet, PatchTST

Usage:
    python utils_new/generate_all_job_arrays.py
    python utils_new/generate_all_job_arrays.py --models DLinear TiDE
    python utils_new/generate_all_job_arrays.py --datasets ETTh1 ETTh2
    python utils_new/generate_all_job_arrays.py --dry-run
"""

import argparse
import subprocess
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from config import (
    SUPPORTED_MODELS, DEFAULT_TASKS, SUPPORTED_TASKS,
    JOB_ARRAYS_BATCH_SCRIPT, MODEL_DELAY_SECONDS, MAX_MODEL_TIMEOUT
)


class JobArrayBatchLauncher:
    """Batch launcher for generating job arrays across multiple models."""
    
    def __init__(self):
        self.job_arrays_batch_script = JOB_ARRAYS_BATCH_SCRIPT
        self.results = {}
    
    def validate_environment(self) -> bool:
        """Validate that required scripts exist."""
        if not Path(self.job_arrays_batch_script).exists():
            print(f"ERROR: Job arrays batch script not found: {self.job_arrays_batch_script}")
            return False
        
        # Test job arrays batch script
        try:
            result = subprocess.run([
                'python', self.job_arrays_batch_script, '--models', 'DLinear', 
                '--tasks', 'long_term_forecast', '--dry-run'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"ERROR: Job arrays batch script failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"ERROR: Failed to test orchestrator script: {e}")
            return False
        
        return True
    
    def execute_model(self, model: str, tasks: List[str] = None, 
                     datasets: List[str] = None, base_output_dir: Path = None,
                     dry_run: bool = False) -> Dict[str, Any]:
        """Execute experiments for a single model."""
        
        if tasks is None:
            tasks = DEFAULT_TASKS
        
        print(f"\n{'='*70}")
        print(f"EXECUTING: {model}")
        print(f"{'='*70}")
        print(f"Tasks: {', '.join(tasks)}")
        if datasets:
            print(f"Datasets: {', '.join(datasets)}")
        print()
        
        # Build command for job arrays batch script
        cmd = [
            'python', self.job_arrays_batch_script,
            '--models', model,
            '--tasks'] + tasks
        
        if datasets:
            cmd.extend(['--datasets'] + datasets)
        
        if base_output_dir:
            model_output_dir = base_output_dir / f'{model}_experiments'
            cmd.extend(['--output-dir', str(model_output_dir)])
        
        if dry_run:
            cmd.append('--dry-run')
        
        print(f"Command: {' '.join(cmd)}")
        
        if dry_run:
            return {
                'model': model,
                'status': 'dry_run',
                'command': ' '.join(cmd)
            }
        
        # Execute orchestrator
        start_time = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=MAX_MODEL_TIMEOUT)
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"SUCCESS: {model} completed in {execution_time:.1f}s")
                print(result.stdout)
                
                return {
                    'model': model,
                    'status': 'success',
                    'command': ' '.join(cmd),
                    'execution_time': execution_time,
                    'stdout': result.stdout
                }
            else:
                print(f"ERROR: {model} failed after {execution_time:.1f}s")
                print(f"Error: {result.stderr}")
                
                return {
                    'model': model,
                    'status': 'failed',
                    'command': ' '.join(cmd),
                    'execution_time': execution_time,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            print(f"TIMEOUT: {model} timed out after {execution_time:.1f}s")
            return {
                'model': model,
                'status': 'timeout',
                'command': ' '.join(cmd),
                'execution_time': execution_time
            }
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"EXCEPTION: {model} failed: {e}")
            return {
                'model': model,
                'status': 'exception',
                'command': ' '.join(cmd),
                'execution_time': execution_time,
                'error': str(e)
            }
    
    def execute_all_models(self, models: List[str] = None, tasks: List[str] = None,
                          datasets: List[str] = None, output_dir: Path = None,
                          dry_run: bool = False) -> Dict[str, Any]:
        """Execute experiments for multiple models sequentially."""
        
        if models is None:
            models = SUPPORTED_MODELS
        
        if tasks is None:
            tasks = DEFAULT_TASKS
        
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(f'./master_experiments_{timestamp}')
        
        print(f"MASTER EXPERIMENT CONTROLLER")
        print(f"Models: {', '.join(models)}")
        print(f"Tasks: {', '.join(tasks)}")
        if datasets:
            print(f"Datasets: {', '.join(datasets)}")
        print(f"Output directory: {output_dir}")
        print(f"Dry run: {dry_run}")
        
        # Create base directory
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Execute models sequentially
        model_results = {}
        successful_models = []
        failed_models = []
        total_execution_time = 0
        
        for i, model in enumerate(models):
            # Add delay between models for monitoring
            if i > 0 and not dry_run:
                print(f"\nWaiting {MODEL_DELAY_SECONDS}s before next model...")
                time.sleep(MODEL_DELAY_SECONDS)
            
            model_result = self.execute_model(
                model=model,
                tasks=tasks,
                datasets=datasets,
                base_output_dir=output_dir,
                dry_run=dry_run
            )
            
            model_results[model] = model_result
            
            if model_result['status'] == 'success':
                successful_models.append(model)
            elif model_result['status'] != 'dry_run':
                failed_models.append(model)
                print(f"\nWARNING: {model} failed. Continue with remaining models? [y/N]")
                if not dry_run:
                    # In production, you might want to prompt user or implement retry logic
                    pass
            
            if 'execution_time' in model_result:
                total_execution_time += model_result['execution_time']
        
        # Generate master summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'models_requested': models,
            'tasks': tasks,
            'datasets': datasets,
            'output_dir': str(output_dir),
            'dry_run': dry_run,
            'model_results': model_results,
            'successful_models': successful_models,
            'failed_models': failed_models,
            'total_models': len(models),
            'success_count': len(successful_models),
            'failure_count': len(failed_models),
            'total_execution_time': total_execution_time
        }
        
        # Save master summary
        if not dry_run:
            summary_file = output_dir / 'master_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nMaster summary saved to: {summary_file}")
        
        return summary
    
    def print_summary(self, summary: Dict[str, Any]) -> None:
        """Print formatted summary of master controller results."""
        
        print(f"\n{'='*70}")
        print(f"MASTER CONTROLLER SUMMARY")
        print(f"{'='*70}")
        
        print(f"Overview:")
        print(f"   Total models: {summary['total_models']}")
        print(f"   Successful: {summary['success_count']}")
        print(f"   Failed: {summary['failure_count']}")
        print(f"   Tasks: {', '.join(summary['tasks'])}")
        if summary.get('total_execution_time'):
            print(f"   Total execution time: {summary['total_execution_time']:.1f}s")
        
        if summary['successful_models']:
            print(f"\nSuccessful Models:")
            for model in summary['successful_models']:
                model_result = summary['model_results'][model]
                exec_time = model_result.get('execution_time', 0)
                print(f"   • {model} ({exec_time:.1f}s)")
        
        if summary['failed_models']:
            print(f"\nFailed Models:")
            for model in summary['failed_models']:
                model_result = summary['model_results'][model]
                error = model_result.get('error', model_result.get('status', 'Unknown'))
                print(f"   • {model} -> {error}")
        
        if not summary['dry_run'] and summary['successful_models']:
            print(f"\nNext Steps:")
            print(f"   1. Upload model experiments to HPC individually")
            print(f"   2. Monitor model jobs independently")
            print(f"   3. Analyze results when complete")
        
        print(f"\nOutput directory: {summary['output_dir']}")
        print(f"Generated: {summary['timestamp']}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch launcher for generating job arrays across all models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate job arrays for all models sequentially
  python utils_new/generate_all_job_arrays.py
  
  # Generate job arrays for specific models only
  python utils_new/generate_all_job_arrays.py --models DLinear TiDE
  
  # Dry run to see what would be executed
  python utils_new/generate_all_job_arrays.py --dry-run
  
  # Custom datasets for long-term forecasting
  python utils_new/generate_all_job_arrays.py --models DLinear --datasets ETTh1 ETTh2

This generates separate generate_job_arrays_batch.py calls:
  python utils_new/generate_job_arrays_batch.py --models DLinear --tasks short_term_forecast long_term_forecast
  python utils_new/generate_job_arrays_batch.py --models TiDE --tasks short_term_forecast long_term_forecast
  python utils_new/generate_job_arrays_batch.py --models TSMixer --tasks short_term_forecast long_term_forecast
  python utils_new/generate_job_arrays_batch.py --models SegRNN --tasks short_term_forecast long_term_forecast
  python utils_new/generate_job_arrays_batch.py --models TimesNet --tasks short_term_forecast long_term_forecast
  python utils_new/generate_job_arrays_batch.py --models PatchTST --tasks short_term_forecast long_term_forecast
        """
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=SUPPORTED_MODELS,
        default=SUPPORTED_MODELS,
        help='Models to run experiments for (default: all models)'
    )
    
    parser.add_argument(
        '--tasks',
        nargs='+',
        choices=SUPPORTED_TASKS,
        default=DEFAULT_TASKS,
        help=f'Tasks to include (default: {DEFAULT_TASKS})'
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
        help='Show what would be executed without running commands'
    )
    
    args = parser.parse_args()
    
    # Initialize job array batch launcher
    controller = JobArrayBatchLauncher()
    
    # Validate environment
    if not controller.validate_environment():
        print("Environment validation failed")
        return 1
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Execute models
    summary = controller.execute_all_models(
        models=args.models,
        tasks=args.tasks,
        datasets=args.datasets,
        output_dir=output_dir,
        dry_run=args.dry_run
    )
    
    # Print summary
    controller.print_summary(summary)
    
    # Return appropriate exit code
    if summary['failure_count'] > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    exit(main())