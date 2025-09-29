#!/usr/bin/env python3
"""
Parameter Extraction and Validation Script

This script extracts key parameters from TSLib scripts and creates a reference
configuration that you can compare against your config.py.
"""

import re
import os
from pathlib import Path
from collections import defaultdict


def extract_key_parameters():
    """Extract key parameters from original scripts."""
    
    # Key datasets and their expected parameters
    expected_configs = {}
    
    print("Extracting parameters from Time-Series-Library scripts...")
    print("=" * 60)
    
    # Check ETT datasets
    ett_script = Path("scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh")
    if ett_script.exists():
        with open(ett_script, 'r') as f:
            content = f.read()
        
        seq_len = re.search(r'--seq_len (\d+)', content)
        label_len = re.search(r'--label_len (\d+)', content)
        
        if seq_len and label_len:
            expected_configs['ETT_datasets'] = {
                'seq_len': int(seq_len.group(1)),
                'label_len': int(label_len.group(1)),
                'pred_lengths': [96, 192, 336, 720]  # From script analysis
            }
            print(f"✓ ETT datasets: seq_len={seq_len.group(1)}, label_len={label_len.group(1)}")
    
    # Check ILI dataset
    ili_script = Path("scripts/long_term_forecast/ILI_script/TimesNet.sh")
    if ili_script.exists():
        with open(ili_script, 'r') as f:
            content = f.read()
        
        seq_len = re.search(r'--seq_len (\d+)', content)
        label_len = re.search(r'--label_len (\d+)', content)
        
        if seq_len and label_len:
            expected_configs['ILI'] = {
                'seq_len': int(seq_len.group(1)),
                'label_len': int(label_len.group(1)),
                'pred_lengths': [24, 36, 48, 60]  # From script analysis
            }
            print(f"✓ ILI dataset: seq_len={seq_len.group(1)}, label_len={label_len.group(1)}")
    
    # Check M4 (short-term) - extract from M4 data meta
    m4_meta_file = Path("data_provider/m4.py")
    if m4_meta_file.exists():
        with open(m4_meta_file, 'r') as f:
            content = f.read()
        
        # Extract horizons_map
        horizons_match = re.search(r"horizons_map = \{([^}]+)\}", content, re.DOTALL)
        if horizons_match:
            horizons_text = horizons_match.group(1)
            print("✓ M4 prediction lengths found:")
            
            patterns = re.findall(r"'(\w+)':\s*(\d+)", horizons_text)
            m4_configs = {}
            for pattern, horizon in patterns:
                pred_len = int(horizon)
                seq_len = 2 * pred_len  # Following the 2*pred_len rule
                m4_configs[pattern] = {'seq_len': seq_len, 'pred_len': pred_len}
                print(f"  {pattern}: seq_len={seq_len}, pred_len={pred_len}")
            
            expected_configs['M4'] = m4_configs
    
    # Check model parameters
    print("\nExtracting model parameters...")
    print("-" * 40)
    
    model_configs = {}
    
    # DLinear - check ETT script
    dlinear_script = Path("scripts/long_term_forecast/ETT_script/DLinear_ETTh1.sh")
    if dlinear_script.exists():
        with open(dlinear_script, 'r') as f:
            content = f.read()
        
        params = {}
        for param in ['e_layers', 'd_layers', 'factor']:
            match = re.search(rf'--{param} (\d+)', content)
            if match:
                params[param] = int(match.group(1))
        
        model_configs['DLinear'] = params
        print(f"✓ DLinear: {params}")
    
    # TimesNet - check ETT script
    timesnet_script = Path("scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh")
    if timesnet_script.exists():
        with open(timesnet_script, 'r') as f:
            content = f.read()
        
        params = {}
        for param in ['e_layers', 'd_layers', 'factor', 'd_model', 'd_ff', 'top_k']:
            match = re.search(rf'--{param} (\d+)', content)
            if match:
                params[param] = int(match.group(1))
        
        model_configs['TimesNet'] = params
        print(f"✓ TimesNet: {params}")
    
    # PatchTST - check ETT script
    patchtst_script = Path("scripts/long_term_forecast/ETT_script/PatchTST_ETTh1.sh")
    if patchtst_script.exists():
        with open(patchtst_script, 'r') as f:
            content = f.read()
        
        params = {}
        for param in ['e_layers', 'd_layers', 'factor', 'n_heads']:
            match = re.search(rf'--{param} (\d+)', content)
            if match:
                params[param] = int(match.group(1))
        
        model_configs['PatchTST'] = params
        print(f"✓ PatchTST: {params}")
    
    expected_configs['models'] = model_configs
    
    return expected_configs


def validate_our_config(expected):
    """Compare our config with expected values."""
    
    try:
        # Import our config (adjust path as needed)
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from config import LONG_TERM_DATASETS, SHORT_TERM_DATASETS, MODEL_CONFIGS
        
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        # Validate dataset configs
        issues = []
        
        # Check ETT datasets
        if 'ETT_datasets' in expected:
            ett_expected = expected['ETT_datasets']
            
            for dataset in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
                if dataset in LONG_TERM_DATASETS:
                    our_config = LONG_TERM_DATASETS[dataset]
                    
                    if our_config['seq_len'] != ett_expected['seq_len']:
                        issues.append(f"❌ {dataset} seq_len: our={our_config['seq_len']}, expected={ett_expected['seq_len']}")
                    else:
                        print(f"✅ {dataset} seq_len: {our_config['seq_len']}")
                    
                    if our_config['label_len'] != ett_expected['label_len']:
                        issues.append(f"❌ {dataset} label_len: our={our_config['label_len']}, expected={ett_expected['label_len']}")
                    else:
                        print(f"✅ {dataset} label_len: {our_config['label_len']}")
        
        # Check ILI dataset
        if 'ILI' in expected and 'ILI' in LONG_TERM_DATASETS:
            ili_expected = expected['ILI']
            our_ili = LONG_TERM_DATASETS['ILI']
            
            if our_ili['seq_len'] != ili_expected['seq_len']:
                issues.append(f"❌ ILI seq_len: our={our_ili['seq_len']}, expected={ili_expected['seq_len']}")
            else:
                print(f"✅ ILI seq_len: {our_ili['seq_len']}")
            
            if our_ili['label_len'] != ili_expected['label_len']:
                issues.append(f"❌ ILI label_len: our={our_ili['label_len']}, expected={ili_expected['label_len']}")
            else:
                print(f"✅ ILI label_len: {our_ili['label_len']}")
        
        # Check M4 dataset
        if 'M4' in expected and 'M4' in SHORT_TERM_DATASETS:
            m4_expected = expected['M4']
            our_m4 = SHORT_TERM_DATASETS['M4']['seasonal_patterns']
            
            for pattern, expected_values in m4_expected.items():
                if pattern in our_m4:
                    our_values = our_m4[pattern]
                    
                    if our_values['seq_len'] != expected_values['seq_len']:
                        issues.append(f"❌ M4 {pattern} seq_len: our={our_values['seq_len']}, expected={expected_values['seq_len']}")
                    else:
                        print(f"✅ M4 {pattern} seq_len: {our_values['seq_len']}")
                    
                    if our_values['pred_len'] != expected_values['pred_len']:
                        issues.append(f"❌ M4 {pattern} pred_len: our={our_values['pred_len']}, expected={expected_values['pred_len']}")
                    else:
                        print(f"✅ M4 {pattern} pred_len: {our_values['pred_len']}")
        
        # Check model configs
        if 'models' in expected:
            model_expected = expected['models']
            
            for model, expected_params in model_expected.items():
                if model in MODEL_CONFIGS:
                    our_params = MODEL_CONFIGS[model]
                    
                    for param, expected_value in expected_params.items():
                        if param in our_params:
                            if our_params[param] != expected_value:
                                issues.append(f"❌ {model} {param}: our={our_params[param]}, expected={expected_value}")
                            else:
                                print(f"✅ {model} {param}: {our_params[param]}")
                        else:
                            issues.append(f"❌ {model} missing parameter: {param}")
        
        # Summary
        print("\n" + "-" * 40)
        if issues:
            print(f"❌ FOUND {len(issues)} ISSUES:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print("✅ ALL CONFIGURATIONS MATCH!")
        
        return len(issues) == 0
        
    except ImportError as e:
        print(f"❌ Could not import config: {e}")
        return False


def main():
    """Main validation function."""
    
    # Extract expected configurations
    expected = extract_key_parameters()
    
    # Validate our config
    is_valid = validate_our_config(expected)
    
    # Save reference config
    # import json
    # with open('tslib_reference_config.json', 'w') as f:
    #     json.dump(expected, f, indent=2)
    
    # print(f"\n📋 Reference configuration saved to: tslib_reference_config.json")
    print(f"🔍 Validation {'PASSED' if is_valid else 'FAILED'}")
    
    if not is_valid:
        print("\n💡 To fix issues:")
        print("1. Check the specific parameters mentioned above")
        print("2. Compare with original TSLib scripts in scripts/ directory")
        print("3. Update config.py accordingly")


if __name__ == "__main__":
    main()