#!/usr/bin/env python3
"""
Plot Target Signal Only - plots the exact signal used during training with --features S

Usage (from repo root):
    python utils/plot_target_signals.py --input_dir dataset --output_dir plots_target_signals --target OT
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import os

def determine_target_signal(csv_path: Path, target: str = 'OT') -> Optional[pd.DataFrame]:
    """
    Determines the target signal that would be used during training with --features S
    
    Args:
        csv_path: Path to the CSV file
        target: Target column name (default 'OT')
        
    Returns:
        DataFrame with only the target signal, or None if not found
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check if this is an M4-style dataset (first column contains series IDs)
        first_col = df.columns[0]
        
        # M4 datasets have series IDs in first column and time series values in subsequent columns
        if first_col in ['V1'] or str(df.iloc[0, 0]).startswith('H'):
            print(f"Detected M4-style dataset: {csv_path.name}")
            # For M4, each row is a separate time series. With --features S, model trains on one series at a time
            # Plot just the first series as example since all series have same structure
            
            first_row = df.iloc[0]
            series_id = first_row.iloc[0]
            # Get numeric values (skip first column which is series ID)
            values = first_row.iloc[1:].dropna().astype(float)
            # Create time index
            time_idx = range(len(values))
            result_df = pd.DataFrame({
                f'{series_id} (example)': values.values
            }, index=time_idx)
            
            return result_df
            
        else:
            # Regular dataset - check if target column exists
            if target in df.columns:
                print(f"Using target column '{target}' from {csv_path.name}")
                return df[[target]]
            else:
                print(f"Target column '{target}' not found in {csv_path.name}. Available columns: {list(df.columns)}")
                # Fall back to last column (common convention)
                last_col = df.columns[-1]
                print(f"Using last column '{last_col}' instead")
                return df[[last_col]]
                
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def plot_target_signal(df: pd.DataFrame, output_path: Path, title: str):
    """
    Plot the target signal(s) and save to PNG
    """
    plt.figure(figsize=(12, 6))
    
    for col in df.columns:
        plt.plot(df[col], label=col, alpha=0.8)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

def process_datasets(input_dir: Path, output_dir: Path, target: str = 'OT', max_files: int = 0):
    """
    Process all CSV files in the dataset directory and plot target signals
    """
    # Skip metadata/results files that aren't training data
    skip_files = {'M4-info.csv', 'submission-Naive2.csv'}
    
    csv_files = []
    for csv_file in input_dir.rglob('*.csv'):
        if csv_file.name not in skip_files:
            csv_files.append(csv_file)
    
    if max_files > 0:
        csv_files = csv_files[:max_files]
    
    print(f"Found {len(csv_files)} CSV files to process")
    
    processed = 0
    for csv_path in csv_files:
        relative_path = csv_path.relative_to(input_dir)
        output_path = output_dir / relative_path.with_suffix('.png')
        
        print(f"Processing: {relative_path}")
        
        target_df = determine_target_signal(csv_path, target)
        if target_df is not None and not target_df.empty:
            plot_target_signal(target_df, output_path, f"Target Signal: {relative_path}")
            print(f"  → Saved: {output_path}")
            processed += 1
        else:
            print(f"  → Skipped: No target signal found")
    
    print(f"\\nCompleted. Processed {processed} files.")

def main():
    parser = argparse.ArgumentParser(description='Plot target signals used in training with --features S')
    parser.add_argument('--input_dir', type=Path, default=Path('dataset'), 
                       help='Input directory containing datasets')
    parser.add_argument('--output_dir', type=Path, default=Path('plots_target_signals'), 
                       help='Output directory for plots')
    parser.add_argument('--target', type=str, default='OT', 
                       help='Target column name (default: OT)')
    parser.add_argument('--max_files', type=int, default=0, 
                       help='Maximum number of files to process (0 = no limit)')
    
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print(f"Input directory does not exist: {args.input_dir}")
        return
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target column: {args.target}")
    print(f"Max files: {args.max_files if args.max_files > 0 else 'unlimited'}")
    print()
    
    process_datasets(args.input_dir, args.output_dir, args.target, args.max_files)

if __name__ == '__main__':
    main()
