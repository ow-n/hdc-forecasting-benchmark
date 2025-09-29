#!/usr/bin/env python3
"""
HPC Job Monitor and Results Collector for Time Series Experiments

This utility helps monitor SLURM job arrays and collect results from completed experiments.
Compatible with the new job array architecture from orchestrate_experiments.py.

Usage:
    python hpc_monitor.py --monitor          # Monitor running job arrays
    python hpc_monitor.py --collect          # Collect and summarize results
    python hpc_monitor.py --status           # Show job status summary
    python hpc_monitor.py --resubmit-failed  # Resubmit failed experiments
    python hpc_monitor.py --orchestration-dir ./experiments_20240929_143022  # Monitor specific orchestration
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import glob
import re


class HPCMonitor:
    """Monitor and manage HPC SLURM job arrays for time series experiments."""
    
    def __init__(self, results_dir: str = "./results", checkpoints_dir: str = "./checkpoints", 
                 orchestration_dir: str = None):
        self.results_dir = Path(results_dir)
        self.checkpoints_dir = Path(checkpoints_dir)
        self.orchestration_dir = Path(orchestration_dir) if orchestration_dir else None
        
        # Updated job patterns for job arrays
        self.job_name_pattern = "ts_experiments"
        self.array_job_pattern = r"ts_experiments_\d+"
        
        # Supported models
        self.models = {
            'linear': ['DLinear', 'TiDE', 'TSMixer', 'SegRNN'],
            'priority': ['TimesNet', 'PatchTST'],
            'all': ['DLinear', 'TiDE', 'TSMixer', 'SegRNN', 'TimesNet', 'PatchTST']
        }
        
        # Dataset configurations
        self.long_term_datasets = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'ECL', 'Exchange', 'Traffic', 'Weather', 'ILI']
        self.short_term_datasets = ['M4']
        self.pred_lengths = [96, 192, 336, 720]
        
        # Load orchestration summary if available
        self.orchestration_summary = self._load_orchestration_summary()
    
    def _load_orchestration_summary(self) -> Optional[Dict]:
        """Load orchestration summary if available."""
        if not self.orchestration_dir:
            return None
            
        summary_file = self.orchestration_dir / 'orchestration_summary.json'
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load orchestration summary: {e}")
        return None
    
    def get_expected_experiments(self) -> int:
        """Calculate expected number of experiments based on orchestration summary or defaults."""
        if self.orchestration_summary:
            # Get from orchestration summary
            models = self.orchestration_summary.get('models', [])
            tasks = self.orchestration_summary.get('tasks', [])
            datasets = self.orchestration_summary.get('datasets')
            
            total = 0
            for model in models:
                if 'long_term_forecast' in tasks:
                    dataset_count = len(datasets) if datasets else len(self.long_term_datasets)
                    total += dataset_count * len(self.pred_lengths)
                if 'short_term_forecast' in tasks:
                    total += 6  # M4 has 6 seasonal patterns
            return total
        else:
            # Default calculation for single model
            return len(self.long_term_datasets) * len(self.pred_lengths)
        
    def get_slurm_jobs(self) -> List[Dict]:
        """Get current SLURM job status for time series experiments."""
        try:
            # Get current jobs - look for job arrays
            result = subprocess.run(['squeue', '-u', os.getenv('USER', 'unknown'), '--format=%i,%j,%T,%M,%l,%R'], 
                                  capture_output=True, text=True, check=True)
            
            jobs = []
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 6 and self.job_name_pattern in parts[1]:
                        job_entry = {
                            'job_id': parts[0],
                            'job_name': parts[1],
                            'state': parts[2],
                            'time': parts[3],
                            'time_limit': parts[4],
                            'reason': parts[5] if len(parts) > 5 else ''
                        }
                        
                        # Determine model from job name or use orchestration info
                        job_entry['model'] = self._extract_model_from_job_name(parts[1])
                        jobs.append(job_entry)
            
            return jobs
        except subprocess.CalledProcessError as e:
            print(f"Error getting SLURM jobs: {e}")
            return []
        except FileNotFoundError:
            print("SLURM not available (squeue command not found)")
            return []
    
    def _extract_model_from_job_name(self, job_name: str) -> str:
        """Extract model name from job name if possible."""
        # For job arrays, job name is typically "ts_experiments"
        # Model info might be in orchestration summary
        if self.orchestration_summary:
            models = self.orchestration_summary.get('successful_models', [])
            if len(models) == 1:
                return models[0]
            elif len(models) > 1:
                return f"Multiple ({','.join(models)})"
        return "Unknown"
    
    def get_job_array_summary(self) -> Dict[str, Dict]:
        """Get detailed job array status breakdown."""
        try:
            # Get detailed job array info
            result = subprocess.run([
                'squeue', '-u', os.getenv('USER', 'unknown'), 
                '--name=' + self.job_name_pattern,
                '--format=%A,%t', '--noheader'
            ], capture_output=True, text=True, check=True)
            
            array_stats = {}
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        job_id = parts[0]
                        state = parts[1]
                        
                        # Extract array ID (parent job ID)
                        array_id = job_id.split('_')[0] if '_' in job_id else job_id
                        
                        if array_id not in array_stats:
                            array_stats[array_id] = {
                                'running': 0, 'pending': 0, 'completed': 0, 'failed': 0
                            }
                        
                        if state in ['RUNNING', 'R']:
                            array_stats[array_id]['running'] += 1
                        elif state in ['PENDING', 'PD']:
                            array_stats[array_id]['pending'] += 1
                        elif state in ['COMPLETED', 'CD']:
                            array_stats[array_id]['completed'] += 1
                        elif state in ['FAILED', 'F', 'CANCELLED', 'CA']:
                            array_stats[array_id]['failed'] += 1
            
            return array_stats
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {}

    def get_completed_jobs(self, days_back: int = 7) -> List[Dict]:
        """Get completed job information from sacct."""
        try:
            # Get completed jobs from last week
            result = subprocess.run([
                'sacct', '-u', os.getenv('USER', 'unknown'), 
                f'--starttime=now-{days_back}days',
                '--format=JobID,JobName,State,ExitCode,MaxRSS,Elapsed,End',
                '--parsable2', '--noheader'
            ], capture_output=True, text=True, check=True)
            
            jobs = []
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    parts = line.split('|')
                    if len(parts) >= 7 and 'DLinear_' in parts[1]:
                        jobs.append({
                            'job_id': parts[0],
                            'job_name': parts[1],
                            'state': parts[2],
                            'exit_code': parts[3],
                            'max_rss': parts[4],
                            'elapsed': parts[5],
                            'end_time': parts[6]
                        })
            
            return jobs
        except subprocess.CalledProcessError as e:
            print(f"Error getting completed jobs: {e}")
            return []
        except FileNotFoundError:
            print("SLURM not available (sacct command not found)")
            return []

    def monitor_jobs(self):
        """Display current job status and progress."""
        print("=== DLinear Experiments Job Monitor ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Current jobs with job array support
        current_jobs = self.get_slurm_jobs()
        array_summary = self.get_job_array_summary()
        
        if current_jobs:
            print(f"📊 Current Job Arrays ({len(current_jobs)} active):")
            print(f"{'Job ID':<15} {'Model':<12} {'State':<12} {'Runtime':<10} {'Status/Reason'}")
            print("-" * 75)
            
            for job in current_jobs:
                model = job.get('model', 'Unknown')
                print(f"{job['job_id']:<15} {model:<12} {job['state']:<12} {job['time']:<10} {job['reason']}")
        else:
            print("📭 No time series job arrays currently running or queued")
        
        # Show job array details if available
        if array_summary:
            print(f"\n📋 Job Array Details:")
            for array_id, details in array_summary.items():
                print(f"  Array {array_id}: {details['running']} running, {details['pending']} pending, {details['completed']} completed")
        
        print()
        
        print()
        
        # Completed jobs
        completed_jobs = self.get_completed_jobs()
        if completed_jobs:
            successful = len([j for j in completed_jobs if j['state'] == 'COMPLETED'])
            failed = len([j for j in completed_jobs if j['state'] in ['FAILED', 'CANCELLED', 'TIMEOUT']])
            
            print(f"✅ Completed Jobs: {successful}")
            print(f"❌ Failed Jobs: {failed}")
            
            if failed > 0:
                print("\nFailed Jobs:")
                for job in completed_jobs:
                    if job['state'] in ['FAILED', 'CANCELLED', 'TIMEOUT']:
                        dataset = job['job_name'].replace('DLinear_', '').replace('_S', '')
                        print(f"  {job['job_id']:<10} {dataset:<10} {job['state']:<12} {job['exit_code']}")
        
        # Results summary
        print()
        self.show_results_summary()

    def show_results_summary(self):
        """Show summary of collected results."""
        print("=== Results Summary ===")
        
        if not self.results_dir.exists():
            print(f"Results directory not found: {self.results_dir}")
            return
        
        # Updated patterns for all models, not just DLinear
        all_models = self.models['all']
        all_result_patterns = []
        
        for model in all_models:
            # Long-term forecast pattern
            all_result_patterns.extend(self.results_dir.glob(f"long_term_forecast_*_{model}_*_ftS_*"))
            # Short-term forecast pattern  
            all_result_patterns.extend(self.results_dir.glob(f"short_term_forecast_*_{model}_*_ftS_*"))
        
        result_dirs = list(set(all_result_patterns))  # Remove duplicates
        
        expected_total = self.get_expected_experiments()
        print(f"📁 Total result directories: {len(result_dirs)}")
        print(f"🎯 Expected experiments: {expected_total}")
        
        if len(result_dirs) > 0:
            # Group by model and dataset
            model_results = {}
            dataset_results = {}
            
            for result_dir in result_dirs:
                # Updated regex to handle all models
                match = re.search(r'(long|short)_term_forecast_([^_]+)_\d*_?(\d+)?_([^_]+)_([^_]+)_ft([SM])', result_dir.name)
                if match:
                    task_type = match.group(1)
                    dataset = match.group(2)  
                    model = match.group(4)
                    
                    # Track by model
                    if model not in model_results:
                        model_results[model] = 0
                    model_results[model] += 1
                    
                    # Track by dataset
                    if dataset not in dataset_results:
                        dataset_results[dataset] = 0
                    dataset_results[dataset] += 1
            
            if model_results:
                print(f"\nResults by model:")
                for model in all_models:
                    count = model_results.get(model, 0)
                    status = "✅" if count > 0 else "❌"
                    print(f"  {status} {model:<12}: {count} experiments")
            
            if dataset_results:
                print(f"\nResults by dataset:")
                for dataset in self.long_term_datasets + self.short_term_datasets:
                    count = dataset_results.get(dataset, 0)
                    status = "✅" if count > 0 else "❌"
                    print(f"  {status} {dataset:<12}: {count} experiments")
        
        print()

    def collect_results(self) -> pd.DataFrame:
        """Collect and aggregate results from all experiments."""
        print("=== Collecting Results ===")
        
        if not self.results_dir.exists():
            print(f"Results directory not found: {self.results_dir}")
            return pd.DataFrame()
        
        results_data = []
        
        # Collect results from all models
        all_models = self.models['all']
        all_result_dirs = []
        
        for model in all_models:
            # Long-term forecast results
            all_result_dirs.extend(self.results_dir.glob(f"long_term_forecast_*_{model}_*_ftS_*"))
            # Short-term forecast results
            all_result_dirs.extend(self.results_dir.glob(f"short_term_forecast_*_{model}_*_ftS_*"))
        
        result_dirs = list(set(all_result_dirs))  # Remove duplicates
        print(f"Found {len(result_dirs)} result directories across all models")
        
        for result_dir in result_dirs:
            try:
                # Parse directory name to extract experiment info
                dir_name = result_dir.name
                
                # Updated regex to handle all models and tasks
                # Long-term: long_term_forecast_ETTh1_96_192_DLinear_ETTh1_ftS_...
                # Short-term: short_term_forecast_M4_DLinear_M4_ftS_...
                long_match = re.search(r'long_term_forecast_([^_]+)_(\d+)_(\d+)_([^_]+)_([^_]+)_ft([SM])', dir_name)
                short_match = re.search(r'short_term_forecast_([^_]+)_([^_]+)_([^_]+)_ft([SM])', dir_name)
                
                if long_match:
                    dataset = long_match.group(1)
                    seq_len = int(long_match.group(2))
                    pred_len = int(long_match.group(3))
                    model = long_match.group(4)
                    data_name = long_match.group(5)
                    features = long_match.group(6)
                    task_type = 'long_term_forecast'
                elif short_match:
                    dataset = short_match.group(1)
                    model = short_match.group(2)
                    data_name = short_match.group(3)
                    features = short_match.group(4)
                    task_type = 'short_term_forecast'
                    seq_len = pred_len = None  # Not applicable for M4
                else:
                    print(f"⚠️ Could not parse directory name: {dir_name}")
                    continue
                
                # Load metrics
                metrics_file = result_dir / "metrics.npy"
                if metrics_file.exists():
                    metrics = np.load(metrics_file)
                    
                    # Assuming metrics format: [mae, mse, rmse, mape, mspe]
                    result_entry = {
                        'task_type': task_type,
                        'model': model,
                        'dataset': dataset,
                        'features': features,
                        'seq_len': seq_len,
                        'pred_len': pred_len,
                        'mae': float(metrics[0]) if len(metrics) > 0 else None,
                        'mse': float(metrics[1]) if len(metrics) > 1 else None,
                        'rmse': float(metrics[2]) if len(metrics) > 2 else None,
                        'mape': float(metrics[3]) if len(metrics) > 3 else None,
                        'mspe': float(metrics[4]) if len(metrics) > 4 else None,
                        'result_dir': str(result_dir)
                    }
                    
                    results_data.append(result_entry)
                    pred_info = f"pred_len={pred_len}" if pred_len else "short_term"
                    print(f"✅ Collected: {model} on {dataset} ({pred_info})")
                else:
                    print(f"⚠️ Metrics file not found: {metrics_file}")
                    
            except Exception as e:
                print(f"❌ Error processing {result_dir}: {e}")
        
        if results_data:
            df = pd.DataFrame(results_data)
            
            # Save aggregated results
            output_file = "dlinear_results_summary.csv"
            df.to_csv(output_file, index=False)
            print(f"\n📊 Results saved to: {output_file}")
            
            # Show summary statistics
            print("\n=== Results Summary ===")
            print(f"Total experiments collected: {len(df)}")
            print(f"Datasets: {sorted(df['dataset'].unique())}")
            print(f"Prediction lengths: {sorted(df['pred_len'].unique())}")
            
            # Best results per dataset
            if 'mae' in df.columns:
                print("\n🏆 Best MAE by Dataset:")
                best_mae = df.groupby('dataset')['mae'].min().sort_values()
                for dataset, mae in best_mae.items():
                    best_row = df[(df['dataset'] == dataset) & (df['mae'] == mae)].iloc[0]
                    print(f"  {dataset:<10}: MAE={mae:.4f} (pred_len={best_row['pred_len']})")
            
            return df
        else:
            print("No results collected")
            return pd.DataFrame()

    def resubmit_failed_jobs(self):
        """Identify and resubmit failed experiments."""
        print("=== Resubmitting Failed Jobs ===")
        
        # Get completed jobs
        completed_jobs = self.get_completed_jobs()
        failed_jobs = [j for j in completed_jobs if j['state'] in ['FAILED', 'CANCELLED', 'TIMEOUT']]
        
        if not failed_jobs:
            print("✅ No failed jobs found")
            return
        
        # Extract dataset names from failed jobs
        failed_datasets = set()
        for job in failed_jobs:
            dataset = job['job_name'].replace('DLinear_', '').replace('_S', '')
            failed_datasets.add(dataset)
        
        print(f"Found {len(failed_jobs)} failed jobs for datasets: {failed_datasets}")
        
        # Find corresponding script files
        resubmitted = 0
        for dataset in failed_datasets:
            script_file = f"DLinear_{dataset}_univariate.sh"
            if os.path.exists(script_file):
                try:
                    result = subprocess.run(['sbatch', script_file], capture_output=True, text=True, check=True)
                    job_id = result.stdout.strip().split()[-1]
                    print(f"✅ Resubmitted {dataset}: Job ID {job_id}")
                    resubmitted += 1
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to resubmit {dataset}: {e}")
            else:
                print(f"⚠️ Script not found: {script_file}")
        
        print(f"\n📊 Resubmitted {resubmitted} jobs")

    def generate_status_report(self) -> str:
        """Generate a comprehensive status report."""
        report = []
        report.append("# Time Series Experiments Status Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if self.orchestration_summary:
            models = self.orchestration_summary.get('successful_models', [])
            tasks = self.orchestration_summary.get('tasks', [])
            report.append(f"## Orchestration Info")
            report.append(f"- Models: {', '.join(models)}")
            report.append(f"- Tasks: {', '.join(tasks)}")
            report.append("")
        
        # Job status
        current_jobs = self.get_slurm_jobs()
        completed_jobs = self.get_completed_jobs()
        array_summary = self.get_job_array_summary()
        
        report.append("## Job Array Status")
        report.append(f"- Active job arrays: {len(current_jobs)}")
        report.append(f"- Completed: {len([j for j in completed_jobs if j['state'] == 'COMPLETED'])}")
        report.append(f"- Failed: {len([j for j in completed_jobs if j['state'] in ['FAILED', 'CANCELLED', 'TIMEOUT']])}")
        
        if array_summary:
            total_running = sum(details['running'] for details in array_summary.values())
            total_pending = sum(details['pending'] for details in array_summary.values())
            report.append(f"- Individual tasks running: {total_running}")
            report.append(f"- Individual tasks pending: {total_pending}")
        
        report.append("")
        
        # Results status with updated patterns
        all_models = self.models['all']
        all_result_dirs = []
        
        for model in all_models:
            all_result_dirs.extend(self.results_dir.glob(f"*_forecast_*_{model}_*_ftS_*"))
        
        result_dirs = list(set(all_result_dirs))
        expected_total = self.get_expected_experiments()
        
        report.append("## Results Status")
        report.append(f"- Total result directories: {len(result_dirs)}")
        report.append(f"- Expected experiments: {expected_total}")
        if expected_total > 0:
            report.append(f"- Progress: {len(result_dirs)}/{expected_total} ({100*len(result_dirs)/expected_total:.1f}%)")
        report.append("")
        
        return '\n'.join(report)


def main():
    parser = argparse.ArgumentParser(
        description="HPC Monitor for Time Series Experiments (Job Arrays)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hpc_monitor.py --monitor                                    # Show current job status
  python hpc_monitor.py --collect                                    # Collect and summarize results
  python hpc_monitor.py --status                                     # Generate status report
  python hpc_monitor.py --orchestration-dir ./experiments_20240929  # Monitor specific orchestration
  python hpc_monitor.py --resubmit-failed                           # Resubmit failed experiments
        """
    )
    
    parser.add_argument('--monitor', action='store_true', help='Monitor current job status')
    parser.add_argument('--collect', action='store_true', help='Collect and aggregate results')
    parser.add_argument('--status', action='store_true', help='Generate status report')
    parser.add_argument('--resubmit-failed', action='store_true', help='Resubmit failed jobs')
    parser.add_argument('--results-dir', default='./results', help='Results directory path')
    parser.add_argument('--checkpoints-dir', default='./checkpoints', help='Checkpoints directory path')
    parser.add_argument('--orchestration-dir', help='Orchestration directory with summary.json')
    
    args = parser.parse_args()
    
    if not any([args.monitor, args.collect, args.status, args.resubmit_failed]):
        # Default action if no specific action is specified
        args.monitor = True
    
    monitor = HPCMonitor(args.results_dir, args.checkpoints_dir, args.orchestration_dir)
    
    if args.monitor:
        monitor.monitor_jobs()
    
    if args.collect:
        monitor.collect_results()
    
    if args.status:
        report = monitor.generate_status_report()
        print(report)
        
        # Save report to file
        with open('ts_experiments_status_report.md', 'w') as f:
            f.write(report)
        print(f"\n📄 Status report saved to: ts_experiments_status_report.md")
    
    if args.resubmit_failed:
        monitor.resubmit_failed_jobs()


if __name__ == "__main__":
    main()