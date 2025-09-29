"""Plot every dataset file under the `dataset/` folder.

Usage (from repo root):
    python utils/plot_datasets.py --input_dir dataset --output_dir plots --max_files 100

The script will recursively walk `input_dir`, load CSV/NPY/NPZ files and create
PNG line plots (multi-column plotted as multiple lines) preserving relative paths
under `output_dir`.

It uses pandas/numpy/matplotlib which are listed in `requirements.txt`.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional


SUPPORTED_EXT = {'.csv', '.npy', '.npz'}


def load_array(path: Path) -> Optional[pd.DataFrame]:
    ext = path.suffix.lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(path)
            # If the csv has a single column with no header, try to coerce
            if df.shape[1] == 1:
                df.columns = ["value"]
            return df
        elif ext == '.npy':
            arr = np.load(path, allow_pickle=True)
            return _array_to_df(arr, path)
        elif ext == '.npz':
            npz = np.load(path, allow_pickle=True)
            # pick the first array-like entry
            for key in npz.files:
                arr = npz[key]
                return _array_to_df(arr, path, key)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
    return None


def _array_to_df(arr: np.ndarray, path: Path, key: Optional[str] = None) -> pd.DataFrame:
    # Convert 1D or 2D numpy arrays to DataFrame. If 3D, flatten first dimension
    if arr is None:
        raise ValueError("Empty array")
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return pd.DataFrame(arr, columns=[key or 'value'])
    elif arr.ndim == 2:
        # columns as 0..n-1 or use names if structured
        if hasattr(arr.dtype, 'names') and arr.dtype.names:
            return pd.DataFrame(arr)
        else:
            return pd.DataFrame(arr)
    else:
        # collapse leading dims into columns
        flat = arr.reshape(arr.shape[0], -1) if arr.shape[0] > 1 else arr.reshape(-1, arr.shape[-1])
        return pd.DataFrame(flat)


def plot_dataframe(df: pd.DataFrame, outpath: Path, title: Optional[str] = None):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    for col in df.columns:
        try:
            plt.plot(df.index, pd.to_numeric(df[col], errors='coerce'), label=str(col))
        except Exception:
            # If column can't be converted, skip
            continue
    if len(df.columns) > 1:
        plt.legend(loc='upper right', fontsize='small')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def should_skip_file(path: Path) -> bool:
    # skip hidden files and non-supported extensions
    if path.name.startswith('.'):
        return True
    if path.suffix.lower() not in SUPPORTED_EXT:
        return True
    # Skip metadata/results files that aren't training data
    if path.name in {'M4-info.csv', 'submission-Naive2.csv'}:
        return True
    return False


def walk_and_plot(input_dir: Path, output_dir: Path, max_files: int = 0):
    files_processed = 0
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()

    for root, dirs, files in os.walk(input_dir):
        rel_root = Path(root).relative_to(input_dir)
        for fname in files:
            if max_files and files_processed >= max_files:
                print(f"Reached max_files={max_files}, stopping")
                return
            fpath = Path(root) / fname
            if should_skip_file(fpath):
                continue
            df = load_array(fpath)
            if df is None or df.shape[0] == 0:
                print(f"Skipping empty or unreadable: {fpath}")
                continue
            # Create an output png path preserving structure
            out_subdir = output_dir / rel_root
            out_subdir.mkdir(parents=True, exist_ok=True)
            outpath = out_subdir / (fpath.stem + '.png')
            title = str(fpath.relative_to(input_dir))
            try:
                plot_dataframe(df, outpath, title=title)
                files_processed += 1
                print(f"Plotted {fpath} -> {outpath}")
            except Exception as e:
                print(f"Failed plotting {fpath}: {e}")
    print(f"Done. Plotted {files_processed} files.")


def main():
    p = argparse.ArgumentParser(description='Plot dataset files under a folder')
    p.add_argument('--input_dir', default='dataset', help='root dataset folder')
    p.add_argument('--output_dir', default='plots', help='where to save PNGs')
    p.add_argument('--max_files', type=int, default=0, help='max files to process (0 = unlimited)')
    args = p.parse_args()

    walk_and_plot(Path(args.input_dir), Path(args.output_dir), max_files=args.max_files)


if __name__ == '__main__':
    main()
