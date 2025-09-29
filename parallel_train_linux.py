#!/usr/bin/env python3
"""
Linux HPC version of parallel training script for Time-Series-Library

Usage:
# Run ALL 30 scripts (will use 4 parallel workers by default)
> python3 parallel_train_linux.py

# Run specific scripts only
> python3 parallel_train_linux.py DLINEAR_M4 TIMESNET_M4 PATCHTST_ECL

# Customize parallel workers (for HPC with more cores)
> PARALLEL_WORKERS=8 python3 parallel_train_linux.py

# For SLURM environments, run as array job
> sbatch --array=1-30 parallel_slurm.sh

Monitoring:
# Watch real-time logs:
> tail -f parallel_logs/*.log

# Check which scripts are running:
> ps aux | grep python

# Monitor system resources:
> htop
# or:
> watch -n 1 'free -h && echo "CPU:" && mpstat 1 1'
"""

import subprocess
import threading
import time
import sys
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Configuration - can be overridden by environment variables
MAX_PARALLEL_SCRIPTS = int(os.environ.get('PARALLEL_WORKERS', 4))
USE_SLURM = os.environ.get('USE_SLURM', 'false').lower() == 'true'

# Scripts to Train Selected Models - 5 Models, 3 Datasets, 30 Scripts
# These paths point to the Linux-compatible scripts created by fix_scripts_for_hpc.py
# Short-term forecasting: M4 Dataset Scripts 
DLINEAR_M4 = "scripts/short_term_forecast/DLinear_M4.sh"
TIMESNET_M4 = "scripts/short_term_forecast/TimesNet_M4.sh"
PATCHTST_M4 = "scripts/short_term_forecast/PatchTST_M4.sh"
ITRANSFORMER_M4 = "scripts/short_term_forecast/iTransformer_M4.sh"
FEDFORMER_M4 = "scripts/short_term_forecast/FEDformer_M4.sh"

# Long-term forecasting: ETT Dataset Scripts
DLINEAR_ETTH1 = "scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh"
DLINEAR_ETTH2 = "scripts/long_term_forecast/ETT_script/DLinear_ETTh2.sh"
DLINEAR_ETTM1 = "scripts/long_term_forecast/ETT_script/DLinear_ETTm1.sh"
DLINEAR_ETTM2 = "scripts/long_term_forecast/ETT_script/DLinear_ETTm2.sh"

TIMESNET_ETTH1 = "scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh"
TIMESNET_ETTH2 = "scripts/long_term_forecast/ETT_script/TimesNet_ETTh2.sh"
TIMESNET_ETTM1 = "scripts/long_term_forecast/ETT_script/TimesNet_ETTm1.sh"
TIMESNET_ETTM2 = "scripts/long_term_forecast/ETT_script/TimesNet_ETTm2.sh"

PATCHTST_ETTH1 = "scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh"
PATCHTST_ETTH2 = "scripts/long_term_forecast/ETT_script/PatchTST_ETTh2.sh"
PATCHTST_ETTM1 = "scripts/long_term_forecast/ETT_script/PatchTST_ETTm1.sh"
PATCHTST_ETTM2 = "scripts/long_term_forecast/ETT_script/PatchTST_ETTm2.sh"

ITRANSFORMER_ETTH1 = "scripts/long_term_forecast/ETT_script/iTransformer_ETTh1.sh"
ITRANSFORMER_ETTH2 = "scripts/long_term_forecast/ETT_script/iTransformer_ETTh2.sh"
ITRANSFORMER_ETTM1 = "scripts/long_term_forecast/ETT_script/iTransformer_ETTm1.sh"
ITRANSFORMER_ETTM2 = "scripts/long_term_forecast/ETT_script/iTransformer_ETTm2.sh"

FEDFORMER_ETTH1 = "scripts/long_term_forecast/ETT_script/FEDformer_ETTh1.sh"
FEDFORMER_ETTH2 = "scripts/long_term_forecast/ETT_script/FEDformer_ETTh2.sh"
FEDFORMER_ETTM1 = "scripts/long_term_forecast/ETT_script/FEDformer_ETTm1.sh"
FEDFORMER_ETTM2 = "scripts/long_term_forecast/ETT_script/FEDformer_ETTm2.sh"

# Long-term forecasting: ECL Dataset Scripts
DLINEAR_ECL = "scripts/long_term_forecast/ECL_script/DLinear.sh"
TIMESNET_ECL = "scripts/long_term_forecast/ECL_script/TimesNet.sh"
PATCHTST_ECL = "scripts/long_term_forecast/ECL_script/PatchTST.sh"
ITRANSFORMER_ECL = "scripts/long_term_forecast/ECL_script/iTransformer.sh"
FEDFORMER_ECL = "scripts/long_term_forecast/ECL_script/FEDformer.sh"

def check_environment():
    """Check if we're in a proper HPC/Linux environment"""
    # Check for bash
    bash_path = shutil.which('bash')
    if not bash_path:
        print("❌ ERROR: bash not found in PATH. Required for running shell scripts.")
        return False
    
    # Check if scripts exist and are Linux-compatible
    script_dir = Path("scripts")
    if not script_dir.exists():
        print(f"❌ ERROR: scripts directory not found at {script_dir.absolute()}")
        print("Run: python fix_scripts_for_hpc.py to create Linux-compatible scripts")
        return False
    
    # Check if scripts look like HPC versions (have shebang)
    sample_script = script_dir / "short_term_forecast" / "DLinear_M4.sh"
    if sample_script.exists():
        with open(sample_script, 'r') as f:
            first_line = f.readline().strip()
            if not first_line.startswith('#!/bin/bash'):
                print("⚠️  WARNING: Scripts may not be Linux-compatible")
                print("Original Windows scripts detected. Run: python fix_scripts_for_hpc.py")
                print("Continuing anyway, but expect failures...")
    
    # Check Python environment
    try:
        import torch
        print(f"✅ PyTorch found: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
        else:
            print("⚠️  CUDA not available - will run on CPU")
    except ImportError:
        print("⚠️  PyTorch not found - make sure your conda environment is activated")
    
    return True

def make_script_executable(script_path):
    """Ensure script has execute permissions"""
    try:
        os.chmod(script_path, 0o755)
    except OSError as e:
        print(f"⚠️  Warning: Could not set execute permissions on {script_path}: {e}")

def run_script_with_logging(script_info):
    """Run a script with proper logging and error handling for Linux"""
    script_name, script_path = script_info
    
    print(f"🚀 [{script_name}] Starting...")
    start_time = time.time()
    
    # Create log file for this script
    log_dir = Path("parallel_logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{script_name}_{int(start_time)}.log"
    
    try:
        # Ensure script is executable
        make_script_executable(script_path)
        
        with open(log_file, 'w') as f:
            # Write header to log file
            f.write(f"=== {script_name} Training Log ===\n")
            f.write(f"Script: {script_path}\n")
            f.write(f"Started: {time.ctime(start_time)}\n")
            f.write(f"Host: {os.uname().nodename}\n")
            f.write(f"PID: {os.getpid()}\n")
            f.write("="*50 + "\n\n")
            f.flush()
            
            # Debug script before running
            f.write("=== SCRIPT DEBUG ===\n")
            f.write(f"Script exists: {Path(script_path).exists()}\n")
            f.write(f"Script executable: {os.access(script_path, os.X_OK)}\n")
            f.write(f"Script size: {Path(script_path).stat().st_size} bytes\n")
            f.write("Script first 10 lines:\n")
            try:
                with open(script_path, 'r') as script_file:
                    for i, line in enumerate(script_file):
                        if i >= 10:
                            break
                        f.write(f"{i+1:2d}: {line}")
            except Exception as e:
                f.write(f"Error reading script: {e}\n")
            f.write("\n" + "="*50 + "\n\n")
            f.flush()
            
            # Set environment variables for the subprocess
            env = os.environ.copy()
            # Ensure PYTHONPATH includes current directory
            pythonpath = env.get('PYTHONPATH', '')
            current_dir = os.getcwd()
            if current_dir not in pythonpath.split(':'):
                env['PYTHONPATH'] = f"{current_dir}:{pythonpath}" if pythonpath else current_dir
            
            # Try to find bash
            bash_cmd = shutil.which('bash') or '/bin/bash'
            
            f.write(f"=== EXECUTION DEBUG ===\n")
            f.write(f"Bash command: {bash_cmd}\n")
            f.write(f"Working directory: {current_dir}\n")
            f.write(f"PYTHONPATH: {env.get('PYTHONPATH', 'not set')}\n")
            f.write("="*50 + "\n\n")
            f.flush()
            
            # Run the script using bash directly
            result = subprocess.run(
                [bash_cmd, script_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd(),
                env=env
            )
        
        wall_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ [{script_name}] COMPLETED in {wall_time/60:.1f} minutes")
            return True, script_name, wall_time
        else:
            print(f"❌ [{script_name}] FAILED after {wall_time/60:.1f} minutes (exit code: {result.returncode})")
            # Print last few lines of log for debugging
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    print(f"Last 5 lines from {script_name} log:")
                    for line in lines[-5:]:
                        print(f"  {line.rstrip()}")
            except:
                pass
            return False, script_name, wall_time
            
    except Exception as e:
        wall_time = time.time() - start_time
        print(f"💥 [{script_name}] ERROR: {e}")
        # Log the error
        try:
            with open(log_file, 'a') as f:
                f.write(f"\n\nERROR: {e}\n")
        except:
            pass
        return False, script_name, wall_time

def get_system_info():
    """Get system information for logging"""
    info = {}
    try:
        info['hostname'] = os.uname().nodename
        info['cpu_count'] = os.cpu_count()
        
        # Check if we're on SLURM
        if 'SLURM_JOB_ID' in os.environ:
            info['slurm_job_id'] = os.environ['SLURM_JOB_ID']
            info['slurm_cpus'] = os.environ.get('SLURM_CPUS_PER_TASK', 'unknown')
            info['slurm_nodes'] = os.environ.get('SLURM_JOB_NUM_NODES', 'unknown')
        
        # GPU info
        try:
            import torch
            if torch.cuda.is_available():
                info['gpus'] = torch.cuda.device_count()
                info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        except ImportError:
            pass
            
    except Exception as e:
        print(f"⚠️  Could not get system info: {e}")
    
    return info

def run_all_scripts_parallel():
    """Run all training scripts in parallel with max workers limit"""
    
    if not check_environment():
        return
    
    # Create logs directory
    log_dir = Path("parallel_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Define all scripts to run - ORGANIZED BY MODEL (better resource utilization)
    all_scripts = [
        # Phase 1: All DLinear experiments (6 scripts total)
        ("DLINEAR_M4", DLINEAR_M4),           # M4 dataset
        ("DLINEAR_ETTH1", DLINEAR_ETTH1),     # ETT datasets
        ("DLINEAR_ETTH2", DLINEAR_ETTH2),
        ("DLINEAR_ETTM1", DLINEAR_ETTM1),
        ("DLINEAR_ETTM2", DLINEAR_ETTM2),
        ("DLINEAR_ECL", DLINEAR_ECL),         # ECL dataset
        
        # Phase 2: All TimesNet experiments (6 scripts total)
        ("TIMESNET_M4", TIMESNET_M4),
        ("TIMESNET_ETTH1", TIMESNET_ETTH1),
        ("TIMESNET_ETTH2", TIMESNET_ETTH2),
        ("TIMESNET_ETTM1", TIMESNET_ETTM1),
        ("TIMESNET_ETTM2", TIMESNET_ETTM2),
        ("TIMESNET_ECL", TIMESNET_ECL),
        
        # Phase 3: All PatchTST experiments (6 scripts total)
        ("PATCHTST_M4", PATCHTST_M4),
        ("PATCHTST_ETTH1", PATCHTST_ETTH1),
        ("PATCHTST_ETTH2", PATCHTST_ETTH2),
        ("PATCHTST_ETTM1", PATCHTST_ETTM1),
        ("PATCHTST_ETTM2", PATCHTST_ETTM2),
        ("PATCHTST_ECL", PATCHTST_ECL),
        
        # Phase 4: All iTransformer experiments (6 scripts total)
        ("ITRANSFORMER_M4", ITRANSFORMER_M4),
        ("ITRANSFORMER_ETTH1", ITRANSFORMER_ETTH1),
        ("ITRANSFORMER_ETTH2", ITRANSFORMER_ETTH2),
        ("ITRANSFORMER_ETTM1", ITRANSFORMER_ETTM1),
        ("ITRANSFORMER_ETTM2", ITRANSFORMER_ETTM2),
        ("ITRANSFORMER_ECL", ITRANSFORMER_ECL),
        
        # Phase 5: All FEDformer experiments (6 scripts total)
        ("FEDFORMER_M4", FEDFORMER_M4),
        ("FEDFORMER_ETTH1", FEDFORMER_ETTH1),
        ("FEDFORMER_ETTH2", FEDFORMER_ETTH2),
        ("FEDFORMER_ETTM1", FEDFORMER_ETTM1),
        ("FEDFORMER_ETTM2", FEDFORMER_ETTM2),
        ("FEDFORMER_ECL", FEDFORMER_ECL),
    ]
    
    # Filter out scripts that don't exist
    existing_scripts = []
    for script_name, script_path in all_scripts:
        if Path(script_path).exists():
            existing_scripts.append((script_name, script_path))
        else:
            print(f"⚠️  Warning: Script not found: {script_path}")
    
    system_info = get_system_info()
    
    print(f"🎯 Starting model-by-model parallel training of {len(existing_scripts)} scripts")
    print(f"📊 Using {MAX_PARALLEL_SCRIPTS} parallel workers")
    print(f"🔄 Training order: DLinear → TimesNet → PatchTST → iTransformer → FEDformer")
    print(f"🖥️  System: {system_info.get('hostname', 'unknown')}")
    print(f"🔧 CPUs: {system_info.get('cpu_count', 'unknown')}")
    if 'gpus' in system_info:
        print(f"🎮 GPUs: {system_info['gpus']} ({', '.join(system_info['gpu_names'])})")
    if 'slurm_job_id' in system_info:
        print(f"🔬 SLURM Job: {system_info['slurm_job_id']}")
    print(f"📁 Logs will be saved to: {log_dir.absolute()}")
    print(f"📝 Results will be in: results/")
    print("="*60)
    
    total_start = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SCRIPTS) as executor:
        # Submit all scripts to the thread pool
        future_to_script = {
            executor.submit(run_script_with_logging, script_info): script_info[0] 
            for script_info in existing_scripts
        }
        
        completed_count = 0
        successful_count = 0
        failed_scripts = []
        
        # Process completed scripts as they finish
        for future in as_completed(future_to_script):
            script_name = future_to_script[future]
            
            try:
                success, name, duration = future.result()
                completed_count += 1
                
                if success:
                    successful_count += 1
                else:
                    failed_scripts.append(name)
                
                # Progress update
                remaining = len(existing_scripts) - completed_count
                print(f"📈 Progress: {completed_count}/{len(existing_scripts)} completed, {remaining} remaining")
                
            except Exception as e:
                print(f"💥 Unexpected error with {script_name}: {e}")
                failed_scripts.append(script_name)
                completed_count += 1
    
    total_time = time.time() - total_start
    
    # Final summary
    print("\n" + "="*60)
    print(f"🎊 PARALLEL TRAINING COMPLETED!")
    print(f"✅ Successful: {successful_count}/{len(existing_scripts)}")
    print(f"❌ Failed: {len(failed_scripts)}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"⚡ Time saved vs sequential: ~{((len(existing_scripts) * 30) - total_time)/60:.0f} minutes")
    
    if failed_scripts:
        print(f"\n💔 Failed scripts:")
        for script in failed_scripts:
            print(f"   - {script}")
    
    print(f"\n📊 Check results in:")
    print(f"   - Training logs: {log_dir.absolute()}")
    print(f"   - Model results: results/")
    print(f"   - Model checkpoints: checkpoints/")

def run_specific_scripts(script_names):
    """Run only specific scripts by name"""
    
    if not check_environment():
        return
    
    # Mapping of script names to their paths
    script_mapping = {
        "DLINEAR_M4": DLINEAR_M4,
        "TIMESNET_M4": TIMESNET_M4,
        "PATCHTST_M4": PATCHTST_M4,
        "ITRANSFORMER_M4": ITRANSFORMER_M4,
        "FEDFORMER_M4": FEDFORMER_M4,
        "DLINEAR_ETTH1": DLINEAR_ETTH1,
        "DLINEAR_ETTH2": DLINEAR_ETTH2,
        "DLINEAR_ETTM1": DLINEAR_ETTM1,
        "DLINEAR_ETTM2": DLINEAR_ETTM2,
        "TIMESNET_ETTH1": TIMESNET_ETTH1,
        "TIMESNET_ETTH2": TIMESNET_ETTH2,
        "TIMESNET_ETTM1": TIMESNET_ETTM1,
        "TIMESNET_ETTM2": TIMESNET_ETTM2,
        "PATCHTST_ETTH1": PATCHTST_ETTH1,
        "PATCHTST_ETTH2": PATCHTST_ETTH2,
        "PATCHTST_ETTM1": PATCHTST_ETTM1,
        "PATCHTST_ETTM2": PATCHTST_ETTM2,
        "ITRANSFORMER_ETTH1": ITRANSFORMER_ETTH1,
        "ITRANSFORMER_ETTH2": ITRANSFORMER_ETTH2,
        "ITRANSFORMER_ETTM1": ITRANSFORMER_ETTM1,
        "ITRANSFORMER_ETTM2": ITRANSFORMER_ETTM2,
        "FEDFORMER_ETTH1": FEDFORMER_ETTH1,
        "FEDFORMER_ETTH2": FEDFORMER_ETTH2,
        "FEDFORMER_ETTM1": FEDFORMER_ETTM1,
        "FEDFORMER_ETTM2": FEDFORMER_ETTM2,
        "DLINEAR_ECL": DLINEAR_ECL,
        "TIMESNET_ECL": TIMESNET_ECL,
        "PATCHTST_ECL": PATCHTST_ECL,
        "ITRANSFORMER_ECL": ITRANSFORMER_ECL,
        "FEDFORMER_ECL": FEDFORMER_ECL,
    }
    
    # Create logs directory
    log_dir = Path("parallel_logs")
    log_dir.mkdir(exist_ok=True)
    
    # Filter to only requested scripts
    selected_scripts = []
    for name in script_names:
        name_upper = name.upper()
        if name_upper in script_mapping:
            script_path = script_mapping[name_upper]
            if Path(script_path).exists():
                selected_scripts.append((name_upper, script_path))
            else:
                print(f"⚠️  Warning: Script file not found: {script_path}")
        else:
            print(f"⚠️  Warning: Script '{name}' not recognized!")
    
    if not selected_scripts:
        print("❌ No valid scripts found!")
        return
    
    print(f"🎯 Running {len(selected_scripts)} selected scripts")
    print(f"📊 Using {min(MAX_PARALLEL_SCRIPTS, len(selected_scripts))} parallel workers")
    
    total_start = time.time()
    
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_SCRIPTS, len(selected_scripts))) as executor:
        future_to_script = {
            executor.submit(run_script_with_logging, script_info): script_info[0] 
            for script_info in selected_scripts
        }
        
        results = []
        for future in as_completed(future_to_script):
            success, name, duration = future.result()
            results.append((success, name, duration))
    
    total_time = time.time() - total_start
    successful = sum(1 for success, _, _ in results if success)
    
    print(f"\n🎊 Selected scripts completed!")
    print(f"✅ Successful: {successful}/{len(selected_scripts)}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")

def generate_slurm_script():
    """Generate a SLURM job script for HPC environments"""
    slurm_script = """#!/bin/bash
#SBATCH --job-name=tslib_training
#SBATCH --output=slurm_logs/tslib_%A_%a.out
#SBATCH --error=slurm_logs/tslib_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=1-30

# Create log directory
mkdir -p slurm_logs

# Load your conda environment
# Uncomment and modify as needed:
# module load anaconda3
# conda activate tslib

# Script mapping for array jobs
case $SLURM_ARRAY_TASK_ID in
    1) SCRIPT="scripts/short_term_forecast/DLinear_M4.sh" ;;
    2) SCRIPT="scripts/short_term_forecast/TimesNet_M4.sh" ;;
    3) SCRIPT="scripts/short_term_forecast/PatchTST_M4.sh" ;;
    4) SCRIPT="scripts/short_term_forecast/iTransformer_M4.sh" ;;
    5) SCRIPT="scripts/short_term_forecast/FEDformer_M4.sh" ;;
    6) SCRIPT="scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh" ;;
    7) SCRIPT="scripts/long_term_forecast/ETT_script/DLinear_ETTh2.sh" ;;
    8) SCRIPT="scripts/long_term_forecast/ETT_script/DLinear_ETTm1.sh" ;;
    9) SCRIPT="scripts/long_term_forecast/ETT_script/DLinear_ETTm2.sh" ;;
    10) SCRIPT="scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh" ;;
    11) SCRIPT="scripts/long_term_forecast/ETT_script/TimesNet_ETTh2.sh" ;;
    12) SCRIPT="scripts/long_term_forecast/ETT_script/TimesNet_ETTm1.sh" ;;
    13) SCRIPT="scripts/long_term_forecast/ETT_script/TimesNet_ETTm2.sh" ;;
    14) SCRIPT="scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh" ;;
    15) SCRIPT="scripts/long_term_forecast/ETT_script/PatchTST_ETTh2.sh" ;;
    16) SCRIPT="scripts/long_term_forecast/ETT_script/PatchTST_ETTm1.sh" ;;
    17) SCRIPT="scripts/long_term_forecast/ETT_script/PatchTST_ETTm2.sh" ;;
    18) SCRIPT="scripts/long_term_forecast/ETT_script/iTransformer_ETTh1.sh" ;;
    19) SCRIPT="scripts/long_term_forecast/ETT_script/iTransformer_ETTh2.sh" ;;
    20) SCRIPT="scripts/long_term_forecast/ETT_script/iTransformer_ETTm1.sh" ;;
    21) SCRIPT="scripts/long_term_forecast/ETT_script/iTransformer_ETTm2.sh" ;;
    22) SCRIPT="scripts/long_term_forecast/ETT_script/FEDformer_ETTh1.sh" ;;
    23) SCRIPT="scripts/long_term_forecast/ETT_script/FEDformer_ETTh2.sh" ;;
    24) SCRIPT="scripts/long_term_forecast/ETT_script/FEDformer_ETTm1.sh" ;;
    25) SCRIPT="scripts/long_term_forecast/ETT_script/FEDformer_ETTm2.sh" ;;
    26) SCRIPT="scripts/long_term_forecast/ECL_script/DLinear.sh" ;;
    27) SCRIPT="scripts/long_term_forecast/ECL_script/TimesNet.sh" ;;
    28) SCRIPT="scripts/long_term_forecast/ECL_script/PatchTST.sh" ;;
    29) SCRIPT="scripts/long_term_forecast/ECL_script/iTransformer.sh" ;;
    30) SCRIPT="scripts/long_term_forecast/ECL_script/FEDformer.sh" ;;
    *) echo "Invalid array task ID: $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac

echo "Running script: $SCRIPT"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Make script executable and run it
chmod +x "$SCRIPT"
bash "$SCRIPT"
"""
    
    with open("parallel_slurm.sh", "w") as f:
        f.write(slurm_script)
    
    print("✅ Generated parallel_slurm.sh for SLURM environments")
    print("📝 To use: sbatch --array=1-30 parallel_slurm.sh")

def main():
    if len(sys.argv) == 1:
        # Run all scripts
        print("🚀 Running ALL training scripts in parallel...")
        run_all_scripts_parallel()
        
    elif sys.argv[1].lower() in ["help", "-h", "--help"]:
        print(__doc__)
        print("Available scripts:")
        print("  M4: DLINEAR_M4, TIMESNET_M4, PATCHTST_M4, ITRANSFORMER_M4, FEDFORMER_M4")
        print("  ETT: DLINEAR_ETTH1, TIMESNET_ETTH1, PATCHTST_ETTH1, etc.")
        print("  ECL: DLINEAR_ECL, TIMESNET_ECL, PATCHTST_ECL, ITRANSFORMER_ECL, FEDFORMER_ECL")
        print(f"\nDefault parallel workers: {MAX_PARALLEL_SCRIPTS}")
        print("Override with: PARALLEL_WORKERS=8 python3 parallel_train_linux.py")
        
    elif sys.argv[1].lower() == "slurm":
        # Generate SLURM script
        generate_slurm_script()
        
    else:
        # Run specific scripts
        script_names = sys.argv[1:]
        print(f"🎯 Running specific scripts: {script_names}")
        run_specific_scripts(script_names)

if __name__ == "__main__":
    main()
