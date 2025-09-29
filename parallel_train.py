# Usage:
# Run ALL 30 scripts (will use 4 parallel workers)
# > python parallel_train.py
# Run specific scripts only
# > python parallel_train.py DLINEAR_M4 TIMESNET_M4 PATCHTST_ECL

# Monitoring
# Watch real-time logs:
# > tail -f parallel_logs/*.log
# Check which scripts are running:
# > ps aux | grep python
# Monitor system resources:
# > htop
# or:
# > watch -n 1 'free -h && echo "CPU:" && mpstat 1 1'

import subprocess
import threading
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
MAX_PARALLEL_SCRIPTS = 4  # Number of scripts to run simultaneously

# Scripts to Train Selected Models - 5 Models, 3 Datasets, 30 Scripts
# Short-term forecasting: M4 Dataset Scripts 
DLINEAR_M4 = "scripts/short_term_forecast/DLinear_M4.sh"  # M4 contains 6 models to train (seasonal patterns)
TIMESNET_M4 = "scripts/short_term_forecast/TimesNet_M4.sh"
PATCHTST_M4 = "scripts/short_term_forecast/PatchTST_M4.sh"
ITRANSFORMER_M4 = "scripts/short_term_forecast/iTransformer_M4.sh"
FEDFORMER_M4 = "scripts/short_term_forecast/FEDformer_M4.sh"

# Long-term forecasting: ETT Dataset Scripts
DLINEAR_ETTH1 = "scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh"  # ETT+ECL is experimented with 4 prediction lengths
DLINEAR_ETTH2 = "scripts/long_term_forecast/ETT_script/DLinear_ETTh2.sh"  # So each script does 4 experiments
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

def run_script_with_logging(script_info):
    """Run a script with proper logging and error handling"""
    script_name, script_path = script_info
    
    print(f"🚀 [{script_name}] Starting...")
    start_time = time.time()
    
    # Create log file for this script
    log_file = f"parallel_logs/{script_name}_{int(start_time)}.log"
    
    try:
        with open(log_file, 'w') as f:
            # Write header to log file
            f.write(f"=== {script_name} Training Log ===\n")
            f.write(f"Script: {script_path}\n")
            f.write(f"Started: {time.ctime(start_time)}\n")
            f.write("="*50 + "\n\n")
            f.flush()
            
            # Run the script and capture all output
            result = subprocess.run(
                [r"C:\Program Files\Git\bin\bash.exe", script_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.getcwd()
            )
        
        wall_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ [{script_name}] COMPLETED in {wall_time/60:.1f} minutes")
            return True, script_name, wall_time
        else:
            print(f"❌ [{script_name}] FAILED after {wall_time/60:.1f} minutes (exit code: {result.returncode})")
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

def run_all_scripts_parallel():
    """Run all training scripts in parallel with max workers limit"""
    
    # Create logs directory
    os.makedirs("parallel_logs", exist_ok=True)
    
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
    
    print(f"🎯 Starting model-by-model parallel training of {len(all_scripts)} scripts")
    print(f"📊 Using {MAX_PARALLEL_SCRIPTS} parallel workers")
    print(f"🔄 Training order: DLinear → TimesNet → PatchTST → iTransformer → FEDformer")
    print(f"📁 Logs will be saved to: parallel_logs/")
    print(f"📝 Results will be in: results/")
    print("="*60)
    
    total_start = time.time()
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SCRIPTS) as executor:
        # Submit all scripts to the thread pool
        future_to_script = {
            executor.submit(run_script_with_logging, script_info): script_info[0] 
            for script_info in all_scripts
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
                remaining = len(all_scripts) - completed_count
                print(f"📈 Progress: {completed_count}/{len(all_scripts)} completed, {remaining} remaining")
                
            except Exception as e:
                print(f"💥 Unexpected error with {script_name}: {e}")
                failed_scripts.append(script_name)
                completed_count += 1
    
    total_time = time.time() - total_start
    
    # Final summary
    print("\n" + "="*60)
    print(f"🎊 PARALLEL TRAINING COMPLETED!")
    print(f"✅ Successful: {successful_count}/{len(all_scripts)}")
    print(f"❌ Failed: {len(failed_scripts)}")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"⚡ Time saved vs sequential: ~{((len(all_scripts) * 30) - total_time)/60:.0f} minutes")
    
    if failed_scripts:
        print(f"\n💔 Failed scripts:")
        for script in failed_scripts:
            print(f"   - {script}")
    
    print(f"\n📊 Check results in:")
    print(f"   - Training logs: parallel_logs/")
    print(f"   - Model results: results/")
    print(f"   - Model checkpoints: checkpoints/")

def run_specific_scripts(script_names):
    """Run only specific scripts by name"""
    
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
    os.makedirs("parallel_logs", exist_ok=True)
    
    # Filter to only requested scripts
    selected_scripts = []
    for name in script_names:
        if name.upper() in script_mapping:
            selected_scripts.append((name.upper(), script_mapping[name.upper()]))
        else:
            print(f"⚠️  Warning: Script '{name}' not found!")
    
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

def main():
    if len(sys.argv) == 1:
        # Run all scripts
        print("🚀 Running ALL training scripts in parallel...")
        run_all_scripts_parallel()
        
    elif sys.argv[1].lower() == "help":
        print("Usage:")
        print("  python parallel_train.py                    # Run all scripts")
        print("  python parallel_train.py script1 script2    # Run specific scripts")
        print("  python parallel_train.py help               # Show this help")
        print("\nAvailable scripts:")
        print("  M4: DLINEAR_M4, TIMESNET_M4, PATCHTST_M4, ITRANSFORMER_M4, FEDFORMER_M4")
        print("  ETT: DLINEAR_ETTH1, TIMESNET_ETTH1, PATCHTST_ETTH1, etc.")
        print("  ECL: DLINEAR_ECL, TIMESNET_ECL, PATCHTST_ECL, ITRANSFORMER_ECL, FEDFORMER_ECL")
        print(f"\nParallel workers: {MAX_PARALLEL_SCRIPTS}")
        
    else:
        # Run specific scripts
        script_names = sys.argv[1:]
        print(f"🎯 Running specific scripts: {script_names}")
        run_specific_scripts(script_names)

if __name__ == "__main__":
    main()
