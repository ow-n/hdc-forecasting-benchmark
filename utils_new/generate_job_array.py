#!/usr/bin/env python3
"""
Job Array + Manifest Generator for Time Series Experiments

This script generates a manifest file (CSV) containing all experiment configurations
and a single SLURM job array script that processes the manifest.

Much more efficient than individual scripts - uses SLURM job arrays for parallel execution.

Usage:
    python utils_new/generate_job_array.py --model DLinear --task long_term_forecast
    python utils_new/generate_job_array.py --model DLinear TimesNet PatchTST --task long_term_forecast short_term_forecast
    python utils_new/generate_job_array.py --help
"""

import os
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from config import (
    LONG_TERM_DATASETS, SHORT_TERM_DATASETS, MODEL_CONFIGS,
    TRAINING_PARAMS, SUPPORTED_MODELS, SUPPORTED_TASKS
)


class JobArrayGenerator:
    """Generator for SLURM job array experiments using manifest files."""
    
    def __init__(self):
        # Use configurations from config.py
        self.long_term_datasets = LONG_TERM_DATASETS
        self.short_term_datasets = SHORT_TERM_DATASETS
        self.model_configs = MODEL_CONFIGS
        self.training_params = TRAINING_PARAMS

    def generate_experiment_manifest(self, models: List[str], tasks: List[str], 
                                   datasets: List[str] = None) -> List[Dict]:
        """Generate list of all experiment configurations."""
        experiments = []
        exp_id = 1
        
        for model in models:
            if model not in self.model_configs:
                print(f"WARNING: Model {model} not supported. Skipping.")
                continue
            
            for task in tasks:
                if task == 'long_term_forecast':
                    target_datasets = datasets if datasets else list(self.long_term_datasets.keys())
                    
                    for dataset_name in target_datasets:
                        if dataset_name not in self.long_term_datasets:
                            print(f"WARNING: Dataset {dataset_name} not supported for long-term. Skipping.")
                            continue
                        
                        dataset_config = self.long_term_datasets[dataset_name]
                        model_config = self.model_configs[model]
                        
                        # Create experiment for each signal and each prediction length
                        for signal in dataset_config['signals']:
                            for pred_len in dataset_config['pred_lengths']:
                                # Use data_name if specified, otherwise use dataset_name
                                data_param = dataset_config.get('data_name', dataset_name)
                                
                                exp = {
                                    'exp_id': exp_id,
                                    'task_name': task,
                                    'model': model,
                                    'dataset': data_param,  # Use data_param instead of dataset_name
                                    'data_path': dataset_config['data_path'],
                                    'root_path': dataset_config['root_path'],
                                    'model_id': f"{dataset_name}_{signal}_{dataset_config['seq_len']}_{pred_len}",
                                    'features': 'S',
                                    'target': signal,
                                    'seq_len': dataset_config['seq_len'],
                                    'label_len': dataset_config['label_len'],
                                    'pred_len': pred_len,
                                    'use_gpu': 0,
                                    'enc_in': 1,
                                    'dec_in': 1,
                                    'c_out': 1,
                                    'seasonal_patterns': '',  # Only for short-term
                                    'loss': 'MSE'  # default for long-term
                                }
                                
                                # Add model-specific parameters
                                exp.update(model_config)
                                
                                # Add training parameters
                                exp.update(self.training_params)
                                
                                experiments.append(exp)
                                exp_id += 1
                
                elif task == 'short_term_forecast':
                    model_config = self.model_configs[model]
                    
                    # One experiment per seasonal pattern with specified seq/pred lengths
                    for seasonal_pattern, pattern_config in self.short_term_datasets['M4']['seasonal_patterns'].items():
                        exp = {
                            'exp_id': exp_id,
                            'task_name': task,
                            'model': model,
                            'dataset': 'm4',
                            'data_path': '',  # Not used for M4
                            'root_path': self.short_term_datasets['M4']['root_path'],
                            'model_id': f"m4_{seasonal_pattern}",
                            'features': 'S',  # Always univariate - treat each season as individual dataset
                            'target': '',
                            'seq_len': pattern_config['seq_len'],
                            'label_len': pattern_config['seq_len'] // 2,  # Half of seq_len
                            'pred_len': pattern_config['pred_len'],
                            'use_gpu': 0,
                            'enc_in': 1,
                            'dec_in': 1,
                            'c_out': 1,
                            'seasonal_patterns': seasonal_pattern,
                            'loss': 'SMAPE',  # Specific for M4
                            'batch_size': 16,  # Override for short-term
                            'learning_rate': 0.001  # Override for short-term
                        }
                        
                        # Add all model-specific parameters for short-term
                        exp.update(model_config)
                        
                        # Override with short-term specific training parameters
                        exp['batch_size'] = 16     # M4 specific
                        exp['learning_rate'] = 0.001  # M4 specific
                        exp['des'] = self.training_params['des']
                        exp['itr'] = self.training_params['itr']
                        
                        experiments.append(exp)
                        exp_id += 1
        
        return experiments

    def save_manifest_csv(self, experiments: List[Dict], output_file: Path) -> str:
        """Save experiment manifest as CSV file."""
        if not experiments:
            raise ValueError("No experiments to save")
        
        # Get all unique keys from all experiments
        all_keys = set()
        for exp in experiments:
            all_keys.update(exp.keys())
        
        # Sort keys for consistent column order
        fieldnames = sorted(all_keys)
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(experiments)
        
        print(f"Generated manifest: {output_file}")
        print(f"Total experiments: {len(experiments)}")
        return str(output_file)

    def generate_job_array_script(self, manifest_file: str, output_dir: Path, 
                                max_concurrent: int = 100) -> str:
        """Generate SLURM job array script that processes the manifest."""
        
        script_path = output_dir / "run_job_array.sh"
        
        script_content = f"""#!/bin/bash
#SBATCH --job-name=ts_experiments
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=1-{len(self.experiments)}%{max_concurrent}
#SBATCH --output=job_%A_%a.out
#SBATCH --error=job_%A_%a.err

# Load necessary modules (modify according to your cluster)
module load python/3.8
module load pytorch/1.12

# Activate your environment (uncomment and modify as needed)
# conda activate your_time_series_env
# source activate your_time_series_env

# Navigate to the Time-Series-Library directory
cd $SLURM_SUBMIT_DIR

echo "=== Job Array Task $SLURM_ARRAY_TASK_ID ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date)"
echo

# Read experiment configuration from manifest
MANIFEST_FILE="{manifest_file}"
EXPERIMENT_LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" $MANIFEST_FILE | tail -n 1)

if [ -z "$EXPERIMENT_LINE" ]; then
    echo "ERROR: Could not read experiment configuration for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Experiment configuration:"
echo "$EXPERIMENT_LINE"
echo

# Parse CSV line (assuming comma-separated values)
IFS=',' read -ra PARAMS <<< "$EXPERIMENT_LINE"

# Function to get parameter value by column name
get_param() {{
    local param_name=$1
    local header_line=$(head -n 1 $MANIFEST_FILE)
    IFS=',' read -ra HEADERS <<< "$header_line"
    
    for i in "${{!HEADERS[@]}}"; do
        if [[ "${{HEADERS[$i]}}" == "$param_name" ]]; then
            echo "${{PARAMS[$i]}}"
            return
        fi
    done
    echo ""
}}

# Function to add parameter if it exists and is not empty
add_param() {{
    local param_name=$1
    local param_value=$(get_param "$param_name")
    if [[ -n "$param_value" && "$param_value" != "0" && "$param_value" != "" ]]; then
        echo "--$param_name $param_value"
    fi
}}

# Extract parameters
TASK_NAME=$(get_param "task_name")
MODEL=$(get_param "model")
DATASET=$(get_param "dataset")
MODEL_ID=$(get_param "model_id")

echo "Running experiment: $MODEL on $DATASET (Task: $TASK_NAME, ID: $MODEL_ID)"

# Build Python command based on task type
if [[ "$TASK_NAME" == "long_term_forecast" ]]; then
    echo "Executing long-term forecasting experiment..."
    
    python -u run.py \\
        --task_name $(get_param "task_name") \\
        --is_training 1 \\
        --root_path "$(get_param "root_path")" \\
        --data_path "$(get_param "data_path")" \\
        --model_id "$(get_param "model_id")" \\
        --model "$(get_param "model")" \\
        --data "$(get_param "dataset")" \\
        --features "$(get_param "features")" \\
        --target "$(get_param "target")" \\
        --seq_len $(get_param "seq_len") \\
        --label_len $(get_param "label_len") \\
        --pred_len $(get_param "pred_len") \\
        --use_gpu $(get_param "use_gpu") \\
        --enc_in $(get_param "enc_in") \\
        --dec_in $(get_param "dec_in") \\
        --c_out $(get_param "c_out") \\
        $(add_param "e_layers") \\
        $(add_param "d_layers") \\
        $(add_param "factor") \\
        $(add_param "d_model") \\
        $(add_param "n_heads") \\
        $(add_param "d_ff") \\
        $(add_param "dropout") \\
        $(add_param "moving_avg") \\
        $(add_param "top_k") \\
        $(add_param "seg_len") \\
        --train_epochs $(get_param "train_epochs") \\
        --batch_size $(get_param "batch_size") \\
        --patience $(get_param "patience") \\
        --learning_rate $(get_param "learning_rate") \\
        --des "$(get_param "des")" \\
        --itr $(get_param "itr")

elif [[ "$TASK_NAME" == "short_term_forecast" ]]; then
    echo "Executing short-term forecasting experiment..."
    
    python -u run.py \\
        --task_name $(get_param "task_name") \\
        --is_training 1 \\
        --root_path "$(get_param "root_path")" \\
        --seasonal_patterns "$(get_param "seasonal_patterns")" \\
        --model_id "$(get_param "model_id")" \\
        --model "$(get_param "model")" \\
        --data "$(get_param "dataset")" \\
        --features "$(get_param "features")" \\
        --use_gpu $(get_param "use_gpu") \\
        --enc_in $(get_param "enc_in") \\
        --dec_in $(get_param "dec_in") \\
        --c_out $(get_param "c_out") \\
        --batch_size $(get_param "batch_size") \\
        --learning_rate $(get_param "learning_rate") \\
        --loss "$(get_param "loss")" \\
        $(add_param "e_layers") \\
        $(add_param "d_layers") \\
        $(add_param "factor") \\
        $(add_param "d_model") \\
        $(add_param "n_heads") \\
        $(add_param "d_ff") \\
        $(add_param "dropout") \\
        $(add_param "moving_avg") \\
        $(add_param "top_k") \\
        $(add_param "seg_len") \\
        --des "$(get_param "des")" \\
        --itr $(get_param "itr")

else
    echo "ERROR: Unknown task type: $TASK_NAME"
    exit 1
fi

# Check execution status
if [ $? -eq 0 ]; then
    echo "SUCCESS: Completed $MODEL $DATASET (Task $SLURM_ARRAY_TASK_ID)"
else
    echo "ERROR: Failed $MODEL $DATASET (Task $SLURM_ARRAY_TASK_ID)"
    exit 1
fi

echo "Task $SLURM_ARRAY_TASK_ID completed at: $(date)"
"""
        
        with open(script_path, 'w', newline='\n') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"Generated job array script: {script_path}")
        return str(script_path)

    def generate_submission_script(self, job_array_script: str, manifest_file: str, 
                                 output_dir: Path) -> str:
        """Generate submission and monitoring scripts."""
        
        submit_script = output_dir / "submit_job_array.sh"
        
        content = f"""#!/bin/bash
# Submit time series experiments job array

echo "=== Time Series Experiments Job Array ==="
echo "Manifest file: {manifest_file}"
echo "Job array script: {Path(job_array_script).name}"
echo "Total experiments: {len(self.experiments)}"
echo

# Submit the job array
JOB_ID=$(sbatch {Path(job_array_script).name} | awk '{{print $4}}')

if [ -n "$JOB_ID" ]; then
    echo "Job array submitted successfully!"
    echo "Job ID: $JOB_ID"
    echo
    echo "Monitor progress:"
    echo "  squeue -j $JOB_ID"
    echo "  squeue -u $USER"
    echo
    echo "Check logs:"
    echo "  ls -la job_${{JOB_ID}}_*.out"
    echo "  tail -f job_${{JOB_ID}}_1.out"
    echo
    echo "Cancel if needed:"
    echo "  scancel $JOB_ID"
else
    echo "ERROR: Failed to submit job array"
    exit 1
fi
"""
        
        with open(submit_script, 'w', newline='\n') as f:
            f.write(content)
        
        os.chmod(submit_script, 0o755)
        
        # Also create monitoring script
        monitor_script = output_dir / "monitor_jobs.sh"
        
        monitor_content = f"""#!/bin/bash
# Monitor job array progress

echo "=== Job Array Monitoring ==="
echo

# Get job array info
JOB_PATTERN="ts_experiments"
JOBS=$(squeue -u $USER --name=$JOB_PATTERN --format="%A %T" --noheader)

if [ -z "$JOBS" ]; then
    echo "No active job arrays found with name: $JOB_PATTERN"
    echo
    echo "Recent completed jobs:"
    sacct -u $USER --name=$JOB_PATTERN -S today --format=JobID,JobName,State,ExitCode -P | head -20
else
    echo "Active jobs:"
    echo "Job ID    State"
    echo "$JOBS"
    echo
    
    # Count by state
    RUNNING=$(echo "$JOBS" | grep -c "RUNNING" || echo "0")
    PENDING=$(echo "$JOBS" | grep -c "PENDING" || echo "0")
    COMPLETING=$(echo "$JOBS" | grep -c "COMPLETING" || echo "0")
    
    echo "Summary:"
    echo "  Running: $RUNNING"
    echo "  Pending: $PENDING" 
    echo "  Completing: $COMPLETING"
    echo "  Total experiments: {len(self.experiments)}"
fi

echo
echo "Recent log files:"
ls -lt job_*.out 2>/dev/null | head -5

echo
echo "Commands:"
echo "  View specific log: tail -f job_<job_id>_<task_id>.out"
echo "  Cancel all jobs: scancel -u $USER --name=$JOB_PATTERN"
echo "  Job details: scontrol show job <job_id>"
"""
        
        with open(monitor_script, 'w', newline='\n') as f:
            f.write(monitor_content)
        
        os.chmod(monitor_script, 0o755)
        
        print(f"Generated submission script: {submit_script}")
        print(f"Generated monitoring script: {monitor_script}")
        
        return str(submit_script)

    def generate_all(self, models: List[str], tasks: List[str], datasets: List[str] = None,
                    output_dir: Path = None, max_concurrent: int = 500):
        """Generate complete job array system."""
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path('.')
        
        print(f"Generating job array system...")
        print(f"Models: {models}")
        print(f"Tasks: {tasks}")
        if datasets:
            print(f"Datasets: {datasets}")
        print(f"Output directory: {output_dir}")
        print()
        
        # Generate experiment manifest
        self.experiments = self.generate_experiment_manifest(models, tasks, datasets)
        
        if not self.experiments:
            print("ERROR: No experiments generated")
            return
        
        # Save manifest CSV
        manifest_file = output_dir / "experiments_manifest.csv"
        self.save_manifest_csv(self.experiments, manifest_file)
        
        # Generate job array script
        job_array_script = self.generate_job_array_script(
            manifest_file.name, output_dir, max_concurrent
        )
        
        # Generate submission scripts
        submit_script = self.generate_submission_script(
            job_array_script, manifest_file.name, output_dir
        )
        
        print(f"""
=== Job Array System Generated ===

Files created:
  {manifest_file.name} - Experiment configurations ({len(self.experiments)} experiments)
  {Path(job_array_script).name} - SLURM job array script
  {Path(submit_script).name} - Submission script
  monitor_jobs.sh - Monitoring script

Usage:
  1. Upload to HPC: rsync -avz {output_dir}/ username@hpc:~/experiments/
  2. Submit jobs: ./submit_job_array.sh
  3. Monitor: ./monitor_jobs.sh

Job Array Features:
  - Single SLURM job with {len(self.experiments)} tasks
  - Max concurrent jobs: {max_concurrent}
  - Automatic load balancing
  - Individual logs: job_<job_id>_<task_id>.out
  - Easy monitoring and cancellation
""")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM job array for time series experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate DLinear job array for long-term forecasting
  python utils_new/generate_job_array.py --model DLinear --task long_term_forecast
  
  # Generate multiple models and tasks
  python utils_new/generate_job_array.py --model DLinear TimesNet PatchTST --task long_term_forecast short_term_forecast
  
  # Specific datasets only
  python utils_new/generate_job_array.py --model DLinear --task long_term_forecast --datasets ETTh1 ETTh2
  
  # Custom concurrent limit
  python utils_new/generate_job_array.py --model DLinear --task long_term_forecast --max-concurrent 20
        """
    )
    
    parser.add_argument(
        '--model', 
        nargs='+',
        required=True,
        choices=SUPPORTED_MODELS,
        help='Models to include in job array'
    )
    
    parser.add_argument(
        '--task',
        nargs='+',
        required=True,
        choices=SUPPORTED_TASKS,
        help='Tasks to include in job array'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Specific datasets for long-term forecasting (default: all)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./job_array_experiments/',
        help='Directory to save job array files (default: ./job_array_experiments/)'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=50,
        help='Maximum concurrent jobs in array (default: 50)'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show experiment count without generating files'
    )
    
    args = parser.parse_args()
    
    generator = JobArrayGenerator()
    
    if args.preview:
        experiments = generator.generate_experiment_manifest(args.model, args.task, args.datasets)
        print(f"Would generate {len(experiments)} experiments")
        
        # Group by model and task
        by_model = {}
        by_task = {}
        for exp in experiments:
            model = exp['model']
            task = exp['task_name']
            by_model[model] = by_model.get(model, 0) + 1
            by_task[task] = by_task.get(task, 0) + 1
        
        print(f"\nBy model: {by_model}")
        print(f"By task: {by_task}")
        return
    
    # Generate the job array system
    generator.generate_all(
        models=args.model,
        tasks=args.task,
        datasets=args.datasets,
        output_dir=Path(args.output_dir),
        max_concurrent=args.max_concurrent
    )


if __name__ == "__main__":
    main()