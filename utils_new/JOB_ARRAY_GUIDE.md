# 🎯 Job Array System - Production Ready

## 📋 **System Overview**

The **Job Array + Manifest Method** is the optimal solution for HPC time series experiments:

- **1 manifest file** (CSV) contains all experiment configurations
- **1 SLURM job array script** processes the manifest  
- **1 submission script** launches everything
- **1 monitoring script** tracks progress

**Result**: Manage 252 experiments with 4 files instead of 250+ individual scripts.

---

## 🚀 **Quick Start**

### **1. Generate Job Array System**

#### **Option A: Single Job Array (Most Common)**
```bash
# DLinear complete benchmark (258 experiments: 252 long-term + 6 short-term)
python utils_new/generate_job_array.py --model DLinear --task long_term_forecast short_term_forecast

# Multi-model comparison (774 experiments = 3 models × 258)
python utils_new/generate_job_array.py --model DLinear TimesNet PatchTST --task long_term_forecast short_term_forecast

# Complete benchmark (1548 experiments = 6 models × 258)
python utils_new/generate_job_array.py --model DLinear TiDE TSMixer SegRNN TimesNet PatchTST --task long_term_forecast short_term_forecast
```

#### **Option B: Separate Job Arrays Per Model (Independent Monitoring)**
```bash
# Generate separate job arrays for each model
python utils_new/generate_job_arrays_batch.py --models DLinear TiDE --tasks long_term_forecast short_term_forecast

# This creates multiple independent job arrays that can be submitted separately
```

#### **Option C: Batch Launch All Models (Sequential Generation)**
```bash
# Generate all models sequentially - useful for organized deployment
python utils_new/generate_all_job_arrays.py --models DLinear TiDE TSMixer --dry-run
```

### **2. Deploy to HPC**
```bash
# Upload to HPC server
rsync -avz ./job_array_experiments/ username@hpc-server:~/experiments/

# SSH and submit
ssh username@hpc-server
cd ~/experiments/
./submit_job_array.sh
```

### **3. Monitor Progress**
```bash
# Check status
./monitor_jobs.sh

# SLURM commands
squeue -u $USER
sacct -u $USER -S today
```

---

## 📊 **Generated Files**

### **Core Files (4 total)**
- **`experiments_manifest.csv`** - All experiment configurations
- **`run_job_array.sh`** - SLURM job array script  
- **`submit_job_array.sh`** - Submission script
- **`monitor_jobs.sh`** - Progress monitoring

### **Generated Logs** 
- **`job_<job_id>_<task_id>.out`** - Individual experiment logs
- **`job_<job_id>_<task_id>.err`** - Individual error logs

---

## ⚙️ **Configuration Options**

### **Models Available**
```bash
--model DLinear TiDE TSMixer SegRNN TimesNet PatchTST
```

### **Tasks Available** 
```bash
--task long_term_forecast short_term_forecast
```

### **Dataset Options**
```bash
# Long-term: All datasets (default)
--datasets ETTh1 ETTh2 ETTm1 ETTm2 ECL Exchange Traffic Weather ILI

# Short-term: M4 with seasonal patterns (automatic)
# Monthly, Yearly, Weekly, Daily, Hourly, Quarterly
```

### **Advanced Options**
```bash
--max-concurrent 50        # Max parallel jobs (default: 50)
--output-dir ./my_exp/     # Custom output directory
--preview                  # Show experiment count without generating
```

---

## 🔧 **SLURM Job Array Features**

### **Resource Allocation**
```bash
#SBATCH --partition=cpu           # CPU-only partition
#SBATCH --cpus-per-task=8         # 8 CPU cores per experiment
#SBATCH --mem=32G                 # 32GB RAM per experiment  
#SBATCH --time=24:00:00           # 24-hour time limit
#SBATCH --array=1-N%50            # N experiments, max 50 concurrent
```

### **Automatic Features**
- ✅ **Load balancing**: SLURM schedules tasks optimally
- ✅ **Resource management**: Efficient CPU/memory allocation
- ✅ **Fault tolerance**: Individual task failures don't affect others
- ✅ **Monitoring**: Built-in job array status tracking
- ✅ **Scalability**: Handle 1-1000+ experiments efficiently

---

## 📈 **Experiment Scale**

### **Per Model Experiments (Updated)**
- **Long-term forecasting**: 252 experiments
  - ETTh1, ETTh2, ETTm1, ETTm2: 4 datasets × 7 signals × 4 pred_lengths = 112
  - Weather: 1 dataset × 7 signals × 4 pred_lengths = 28
  - Traffic: 1 dataset × 7 signals × 4 pred_lengths = 28  
  - ECL: 1 dataset × 7 signals × 4 pred_lengths = 28
  - Exchange: 1 dataset × 7 signals × 4 pred_lengths = 28
  - ILI: 1 dataset × 7 signals × 4 pred_lengths = 28
- **Short-term forecasting**: 6 experiments
  - M4: 6 seasonal patterns (Monthly, Yearly, Weekly, Daily, Hourly, Quarterly)

### **Total Experiments**
- **Single model**: 258 experiments (252 + 6)
- **3 models**: 774 experiments (3 × 258)
- **6 models**: 1548 experiments (6 × 258)

---

## 🎯 **Usage Examples**

### **Development/Testing**
```bash
# Single model, subset of datasets
python utils_new/generate_job_array.py --model DLinear --task long_term_forecast --datasets ETTh1 ETTh2

# Preview experiment count
python utils_new/generate_job_array.py --model DLinear TimesNet --task long_term_forecast --preview

# Independent job arrays per model
python utils_new/generate_job_arrays_batch.py --models DLinear TiDE --tasks long_term_forecast --dry-run
```

### **Production Benchmarks**
```bash
# Linear models comparison (single job array)
python utils_new/generate_job_array.py --model DLinear TiDE TSMixer --task long_term_forecast short_term_forecast

# CNN vs Transformer comparison (separate job arrays)  
python utils_new/generate_job_arrays_batch.py --models TimesNet PatchTST --tasks long_term_forecast short_term_forecast

# Complete benchmark (all models in one job array)
python utils_new/generate_job_array.py --model DLinear TiDE TSMixer SegRNN TimesNet PatchTST --task long_term_forecast short_term_forecast --max-concurrent 100

# Complete benchmark (separate job arrays per model)
python utils_new/generate_all_job_arrays.py --models DLinear TiDE TSMixer SegRNN TimesNet PatchTST
```

---

## 🔍 **Monitoring & Management**

### **Job Status**
```bash
# Quick status check
./monitor_jobs.sh

# Detailed SLURM info
squeue -u $USER --format="%.10i %.20j %.8T %.10M %.6D %R"
scontrol show job <job_id>
```

### **Log Management**
```bash
# View specific experiment log
tail -f job_<job_id>_<task_id>.out

# Check for errors
grep -l "ERROR" job_*_*.out
grep -l "SUCCESS" job_*_*.out

# Monitor progress across all logs
tail -f job_*_*.out | grep -E "(Running|SUCCESS|ERROR)"
```

### **Job Control**
```bash
# Cancel all experiments
scancel -u $USER --name=ts_experiments

# Cancel specific job array
scancel <job_id>

# Hold/release jobs
scontrol hold <job_id>
scontrol release <job_id>
```

---

## 📊 **Performance Expectations**

### **Execution Times (CPU-only) - Updated**
| Model | Time per Experiment | 252 Long-term | 6 Short-term | Total per Model |
|-------|-------------------|---------------|---------------|----------------|
| DLinear | 1-2 hours | 252-504 hours | 6-12 hours | 258-516 hours |
| TimesNet | 3-6 hours | 756-1512 hours | 18-36 hours | 774-1548 hours |
| PatchTST | 2-4 hours | 504-1008 hours | 12-24 hours | 516-1032 hours |
| TiDE | 2-3 hours | 504-756 hours | 12-18 hours | 516-774 hours |
| TSMixer | 2-4 hours | 504-1008 hours | 12-24 hours | 516-1032 hours |
| SegRNN | 1-3 hours | 252-756 hours | 6-18 hours | 258-774 hours |

### **Concurrent Execution**
With `--max-concurrent 50`:
- **1548 experiments** (6 models) complete in ~516-1548 hours (21-64 days)
- **774 experiments** (3 models) complete in ~258-774 hours (10.5-32 days)  
- **258 experiments** (1 model) complete in ~86-258 hours (3.5-10.5 days)

---

## 🎯 **Next Steps**

1. **Test the system**:
   ```bash
   python utils_new/generate_job_array.py --model DLinear --task long_term_forecast --datasets ETTh1 --preview
   ```

2. **Generate your experiments**:
   ```bash
   python utils_new/generate_job_array.py --model DLinear --task long_term_forecast short_term_forecast
   ```

3. **Alternative: Use batch generation for multiple models**:
   ```bash
   python utils_new/generate_job_arrays_batch.py --models DLinear TiDE --tasks long_term_forecast short_term_forecast --dry-run
   ```

3. **Deploy to HPC**:
   ```bash
   rsync -avz ./job_array_experiments/ username@hpc:~/experiments/
   ```

4. **Run experiments**:
   ```bash
   cd ~/experiments/ && ./submit_job_array.sh
   ```

5. **Monitor and collect results**:
   ```bash
   ./monitor_jobs.sh
   ```

**The job array system is production-ready and vastly superior to individual scripts! 🚀**