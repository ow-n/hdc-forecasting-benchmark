#!/usr/bin/env python3
"""
Configuration Constants for Time Series Benchmarking System

Centralized configuration for job arrays, orchestration, and master controller.
"""

from typing import Dict, List, Any


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

# Long-term forecasting datasets with specified signals (7 per dataset)
LONG_TERM_DATASETS = {
    'ETTh1': {
        'data_path': 'ETTh1.csv',
        'root_path': './dataset/ETT-small/',
        'seq_len': 96,
        'label_len': 48,
        'signals': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'pred_lengths': [96, 192, 336, 720]
    },
    'ETTh2': {
        'data_path': 'ETTh2.csv', 
        'root_path': './dataset/ETT-small/',
        'seq_len': 96,
        'label_len': 48,
        'signals': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'pred_lengths': [96, 192, 336, 720]
    },
    'ETTm1': {
        'data_path': 'ETTm1.csv',
        'root_path': './dataset/ETT-small/',
        'seq_len': 96,
        'label_len': 48,
        'signals': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'pred_lengths': [96, 192, 336, 720]
    },
    'ETTm2': {
        'data_path': 'ETTm2.csv',
        'root_path': './dataset/ETT-small/',
        'seq_len': 96,
        'label_len': 48,
        'signals': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
        'pred_lengths': [96, 192, 336, 720]
    },
    'ECL': {
        'data_path': 'ECL.csv',
        'root_path': './dataset/electricity/',
        'seq_len': 96,
        'label_len': 48,
        'signals': ['MT_0', 'MT_50', 'MT_100', 'MT_150', 'MT_200', 'MT_250', 'OT'],
        'pred_lengths': [96, 192, 336, 720]
    },
    'Exchange': {
        'data_path': 'exchange_rate.csv',
        'root_path': './dataset/exchange_rate/',
        'seq_len': 96,
        'label_len': 48,
        'signals': ['0', '1', '2', '3', '4', '5', 'OT'],
        'pred_lengths': [96, 192, 336, 720]
    },
    'Traffic': {
        'data_path': 'traffic.csv',
        'root_path': './dataset/traffic/',
        'seq_len': 96,
        'label_len': 48,
        'signals': ['0', '100', '200', '300', '400', '500', 'OT'],
        'pred_lengths': [96, 192, 336, 720]
    },
    'Weather': {
        'data_path': 'weather.csv',
        'root_path': './dataset/weather/',
        'seq_len': 96,
        'label_len': 48,
        'signals': ['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)', 'VPmax (mbar)', 'OT'],
        'pred_lengths': [96, 192, 336, 720]
    },
    'ILI': {
        'data_path': 'national_illness.csv',
        'root_path': './dataset/illness/',
        'seq_len': 36,
        'label_len': 18,
        'data_name': 'custom',  # ILI uses 'custom' as data parameter, not 'ILI'
        'signals': ['% WEIGHTED ILI', '%UNWEIGHTED ILI', 'AGE 0-4', 'AGE 5-24', 'ILITOTAL', 'NUM. OF PROVIDERS', 'OT'],
        'pred_lengths': [24, 36, 48, 60]
    }
}

# Short-term forecasting datasets
SHORT_TERM_DATASETS = {
    'M4': {
        'root_path': './dataset/m4',
        'seasonal_patterns': {
            # Using M4Meta horizons_map values as pred_len, seq_len = 2 * pred_len
            'Yearly': {'seq_len': 12, 'pred_len': 6},      # horizons_map: 6 
            'Quarterly': {'seq_len': 16, 'pred_len': 8},    # horizons_map: 8
            'Monthly': {'seq_len': 36, 'pred_len': 18},    # horizons_map: 18
            'Weekly': {'seq_len': 26, 'pred_len': 13},     # horizons_map: 13
            'Daily': {'seq_len': 28, 'pred_len': 14},      # horizons_map: 14
            'Hourly': {'seq_len': 96, 'pred_len': 48}     # horizons_map: 48
        }
    }
}


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

# Model hyperparameters and architecture settings
MODEL_CONFIGS = {
    'DLinear': {
        'e_layers': 2,
        'd_layers': 1,
        'factor': 3,
        'moving_avg': 25  # Only used for DLinear
    },
    'TimesNet': {
        'e_layers': 2,
        'd_layers': 1,
        'factor': 3,
        'd_model': 16,  # ETTh1 long-term: 16, M4 varies: Monthly=32, Yearly=16
        'd_ff': 32,     # ETTh1 long-term: 32, M4: 32
        'top_k': 5,
        # Note: TimesNet d_model varies by dataset - scripts use different values
    },
    'PatchTST': {
        'e_layers': 1,
        'd_layers': 1,
        'factor': 3,
        'n_heads': 2,   # ETTh1_96_96: 2, ETTh1_96_192: 8 (varies by pred_len)
    },
    'TiDE': {
        'e_layers': 2,
        'd_layers': 2,
        'd_model': 256,
        'd_ff': 256,
        'dropout': 0.3,
        'batch_size': 512,      # TiDE-specific overrides
        'learning_rate': 0.1,   # TiDE-specific overrides
        'patience': 5,          # TiDE-specific overrides
        'train_epochs': 10      # TiDE-specific overrides
    },
    'TSMixer': {
        'e_layers': 2,
        'd_layers': 1,
        'factor': 3
    },
    'SegRNN': {
        'seg_len': 24,
        'd_model': 512,
        'dropout': 0.5,
        'learning_rate': 0.0001  # SegRNN-specific override
    }
}

# Supported models list
SUPPORTED_MODELS = ['DLinear', 'TiDE', 'TSMixer', 'SegRNN', 'TimesNet', 'PatchTST']


# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Default training hyperparameters
TRAINING_PARAMS = {
    'train_epochs': 10,
    'batch_size': 32,
    'patience': 3,
    'learning_rate': 0.0001,
    'des': 'Exp',
    'itr': 1
}

# Default task configuration
DEFAULT_TASKS = ['short_term_forecast', 'long_term_forecast']
SUPPORTED_TASKS = ['long_term_forecast', 'short_term_forecast']


# =============================================================================
# HPC RESOURCE CONFIGURATION
# =============================================================================

# SLURM job array settings
DEFAULT_CONCURRENT_PER_MODEL = 500
MAX_CONCURRENT_JOBS = 500

# Model-specific concurrent job limits (500 cores per model)
MODEL_CONCURRENT_LIMITS = {
    'DLinear': 500,
    'TiDE': 500,
    'TSMixer': 500,
    'SegRNN': 500,
    'TimesNet': 500,
    'PatchTST': 500
}

# Timeout settings
JOB_TIMEOUT_SECONDS = 300  # 5 minutes for job array generation
MAX_MODEL_TIMEOUT = 1800   # 30 minutes per model in master controller

# Delays between operations
MODEL_DELAY_SECONDS = 5    # Brief delay between models in batch launcher


# =============================================================================
# SCRIPT PATHS AND DEFAULTS
# =============================================================================

# Script locations
JOB_ARRAYS_BATCH_SCRIPT = './utils_new/generate_job_arrays_batch.py'
GENERATOR_SCRIPT = './utils_new/generate_job_array.py'

# Default directories
DEFAULT_JOB_ARRAY_OUTPUT_DIR = './job_array_experiments/'
DEFAULT_EXPERIMENTS_OUTPUT_DIR = './experiments'


# =============================================================================
# EXPERIMENT DESCRIPTIONS
# =============================================================================

# Model descriptions for job array batch generation
EXPERIMENT_CONFIGS = {
    'DLinear': {
        'max_concurrent': MODEL_CONCURRENT_LIMITS['DLinear'],
        'description': 'Linear baseline model - lightweight and fast'
    },
    'TiDE': {
        'max_concurrent': MODEL_CONCURRENT_LIMITS['TiDE'],
        'description': 'Time-series Dense Encoder with linear projections'
    },
    'TSMixer': {
        'max_concurrent': MODEL_CONCURRENT_LIMITS['TSMixer'],
        'description': 'MLP-based mixer architecture for time series'
    },
    'SegRNN': {
        'max_concurrent': MODEL_CONCURRENT_LIMITS['SegRNN'],
        'description': 'Segmented RNN for time series forecasting'
    },
    'TimesNet': {
        'max_concurrent': MODEL_CONCURRENT_LIMITS['TimesNet'],
        'description': 'CNN-based model with temporal convolutions'
    },
    'PatchTST': {
        'max_concurrent': MODEL_CONCURRENT_LIMITS['PatchTST'],
        'description': 'Transformer with patching mechanism'
    }
}