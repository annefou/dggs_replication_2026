#!/usr/bin/env python3
"""
Raster Benchmark with Polars/Parquet Implementation

This script compares three approaches for raster-based land-use classification:
1. Traditional raster (NumPy arrays)
2. DGGS with Pandas DataFrames (basic implementation)
3. DGGS with Polars + Parquet (paper's implementation)

From Law & Ardo (2024), Section 4:
"Classification of DGGS data is one order of magnitude faster than the raster 
method due to benefits accruing primarily to using a columnar data store 
(Apache Parquet) classified with a multi-threaded OLAP query engine (Polars)"

DOI: 10.1080/20964471.2024.2429847
"""

import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

import numpy as np
import pandas as pd
import h3

# Optional imports
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("Warning: Polars not available. Install with: pip install polars")

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("Warning: PyArrow not available. Install with: pip install pyarrow")

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "random_seed": 42,
    "h3_resolution": 9,
    "raster_size": (100, 100),  # Paper: 100x100 pixels
    "num_layers_list": [10, 50, 100, 500, 1000, 5000, 10000],
    "bbox": (-0.5, -0.5, 0.5, 0.5),  # Approx 100km x 100km at equator
    "parquet_compression": "snappy",
}

CODE_VERSION = "2026-01-20-polars-benchmark-v1"

# =============================================================================
# Classification Functions (from paper Section 3.2.1)
# =============================================================================

@lru_cache(maxsize=10000)
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

@lru_cache(maxsize=10000)
def is_perfect(n: int) -> bool:
    if n < 2:
        return False
    divisor_sum = 1
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            divisor_sum += i
            if i != n // i:
                divisor_sum += n // i
    return divisor_sum == n

@lru_cache(maxsize=10000)
def is_triangular(n: int) -> bool:
    if n < 0:
        return False
    k = int((2 * n) ** 0.5)
    return k * (k + 1) // 2 == n

@lru_cache(maxsize=10000)
def is_square(n: int) -> bool:
    if n < 0:
        return False
    root = int(n ** 0.5)
    return root * root == n

@lru_cache(maxsize=10000)
def is_pentagonal(n: int) -> bool:
    if n < 0:
        return False
    k = (1 + (1 + 24 * n) ** 0.5) / 6
    return k == int(k) and k > 0

@lru_cache(maxsize=10000)
def is_hexagonal(n: int) -> bool:
    if n < 0:
        return False
    k = (1 + (1 + 8 * n) ** 0.5) / 4
    return k == int(k) and k > 0

@lru_cache(maxsize=10000)
def is_fibonacci(n: int) -> bool:
    if n < 0:
        return False
    def is_perfect_square(x):
        s = int(x ** 0.5)
        return s * s == x
    return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)

@lru_cache(maxsize=100000)
def classify_value(sum_value: int) -> int:
    """Apply all 7 classification functions to produce a 7-bit class ID."""
    bits = (
        (1 if is_prime(sum_value) else 0) << 0 |
        (1 if is_perfect(sum_value) else 0) << 1 |
        (1 if is_triangular(sum_value) else 0) << 2 |
        (1 if is_square(sum_value) else 0) << 3 |
        (1 if is_pentagonal(sum_value) else 0) << 4 |
        (1 if is_hexagonal(sum_value) else 0) << 5 |
        (1 if is_fibonacci(sum_value) else 0) << 6
    )
    return bits

# Vectorized version for Polars
def classify_array(arr: np.ndarray) -> np.ndarray:
    """Vectorized classification using numpy."""
    return np.array([classify_value(int(v)) for v in arr])


# =============================================================================
# Data Generation
# =============================================================================

def generate_raster_layer(size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Generate a spatially correlated raster layer."""
    base = rng.uniform(0, 1, size)
    if HAS_SCIPY:
        smoothed = gaussian_filter(base, sigma=2)
        # Normalize to 0-1
        smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-10)
        return smoothed
    return base


def generate_all_rasters(num_layers: int, size: Tuple[int, int], 
                         rng: np.random.Generator, data_dir: Path) -> List[Path]:
    """Generate all raster layers and save to disk."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    
    for i in tqdm(range(num_layers), desc=f"Generating {num_layers} rasters"):
        raster = generate_raster_layer(size, rng)
        path = data_dir / f"raster_{i:05d}.npy"
        np.save(path, raster)
        paths.append(path)
    
    return paths


def raster_to_h3_cells(raster: np.ndarray, bbox: Tuple[float, float, float, float],
                       resolution: int) -> List[Tuple[str, float]]:
    """Convert raster to H3 cells with values."""
    minx, miny, maxx, maxy = bbox
    rows, cols = raster.shape
    
    cell_values = []
    for row in range(rows):
        for col in range(cols):
            # Calculate lat/lng for cell center
            lng = minx + (col + 0.5) * (maxx - minx) / cols
            lat = miny + (row + 0.5) * (maxy - miny) / rows
            
            cell_id = h3.latlng_to_cell(lat, lng, resolution)
            cell_values.append((cell_id, raster[row, col]))
    
    return cell_values


# =============================================================================
# Method 1: Traditional Raster (NumPy)
# =============================================================================

def benchmark_raster_method(raster_paths: List[Path], 
                            num_layers: int) -> Dict[str, float]:
    """Benchmark traditional raster overlay and classification."""
    paths = raster_paths[:num_layers]
    
    # Load and stack rasters (simulating warp/alignment)
    warp_start = time.perf_counter()
    
    # In a real scenario, we'd warp to common grid. Here they're already aligned.
    stacked = np.stack([np.load(p) for p in paths], axis=0)
    
    warp_time = time.perf_counter() - warp_start
    
    # Classification
    classify_start = time.perf_counter()
    
    # Sum across layers and convert to integer bins (0-10 per layer -> 0 to 10*n)
    sum_values = (stacked * 10).astype(int).sum(axis=0)
    
    # Apply classification (vectorized)
    classified = np.vectorize(classify_value)(sum_values)
    
    classify_time = time.perf_counter() - classify_start
    
    return {
        "warp_time": warp_time,
        "classify_time": classify_time,
        "total_time": warp_time + classify_time,
    }


# =============================================================================
# Method 2: DGGS with Pandas (basic implementation)
# =============================================================================

def benchmark_dggs_pandas(raster_paths: List[Path], num_layers: int,
                          bbox: Tuple, resolution: int) -> Dict[str, float]:
    """Benchmark DGGS method using Pandas DataFrames."""
    paths = raster_paths[:num_layers]
    
    # Indexing phase: convert rasters to H3
    index_start = time.perf_counter()
    
    all_dfs = []
    for i, path in enumerate(paths):
        raster = np.load(path)
        cells = raster_to_h3_cells(raster, bbox, resolution)
        df = pd.DataFrame(cells, columns=['h3_cell', f'value_{i}'])
        all_dfs.append(df)
    
    # Merge all on h3_cell
    merged = all_dfs[0]
    for df in all_dfs[1:]:
        merged = merged.merge(df, on='h3_cell', how='outer')
    
    index_time = time.perf_counter() - index_start
    
    # Classification phase
    classify_start = time.perf_counter()
    
    value_cols = [c for c in merged.columns if c.startswith('value_')]
    merged['sum_value'] = (merged[value_cols].fillna(0) * 10).sum(axis=1).astype(int)
    merged['class'] = merged['sum_value'].apply(classify_value)
    
    classify_time = time.perf_counter() - classify_start
    
    return {
        "index_time": index_time,
        "classify_time": classify_time,
        "total_time": index_time + classify_time,
    }


# =============================================================================
# Method 3: DGGS with Polars + Parquet (paper's implementation)
# =============================================================================

def benchmark_dggs_polars(raster_paths: List[Path], num_layers: int,
                          bbox: Tuple, resolution: int,
                          parquet_dir: Path) -> Dict[str, float]:
    """
    Benchmark DGGS method using Polars + Parquet.
    
    This matches the paper's approach:
    "using a columnar data store (Apache Parquet) classified with 
    a multi-threaded OLAP query engine (Polars)"
    """
    if not HAS_POLARS or not HAS_PARQUET:
        return {"error": "Polars/Parquet not available"}
    
    paths = raster_paths[:num_layers]
    parquet_dir.mkdir(parents=True, exist_ok=True)
    
    # Indexing phase: convert rasters to H3 and save as Parquet
    index_start = time.perf_counter()
    
    # Create a single wide table (columnar format)
    # First, get all H3 cells from first raster
    first_raster = np.load(paths[0])
    base_cells = raster_to_h3_cells(first_raster, bbox, resolution)
    
    # Build data dict with h3_cell as first column
    data = {'h3_cell': [c[0] for c in base_cells]}
    
    # Add value columns for each layer
    for i, path in enumerate(paths):
        raster = np.load(path)
        cells = raster_to_h3_cells(raster, bbox, resolution)
        data[f'value_{i}'] = [c[1] for c in cells]
    
    # Create Polars DataFrame and write to Parquet
    df = pl.DataFrame(data)
    parquet_path = parquet_dir / "dggs_data.parquet"
    df.write_parquet(parquet_path, compression=CONFIG["parquet_compression"])
    
    index_time = time.perf_counter() - index_start
    
    # Classification phase using Polars lazy evaluation
    classify_start = time.perf_counter()
    
    # Read from Parquet (lazy)
    lf = pl.scan_parquet(parquet_path)
    
    # Sum value columns using Polars expressions
    value_cols = [f'value_{i}' for i in range(num_layers)]
    
    # Polars horizontal sum with scaling
    result = (
        lf
        .with_columns([
            (pl.sum_horizontal([pl.col(c) * 10 for c in value_cols]))
            .cast(pl.Int64)
            .alias('sum_value')
        ])
        .collect()
    )
    
    # Apply classification (need to use map for custom function)
    sum_values = result['sum_value'].to_numpy()
    classes = classify_array(sum_values)
    result = result.with_columns(pl.Series('class', classes))
    
    classify_time = time.perf_counter() - classify_start
    
    return {
        "index_time": index_time,
        "classify_time": classify_time,
        "total_time": index_time + classify_time,
    }


# =============================================================================
# Method 4: DGGS with Polars (in-memory, no Parquet I/O)
# =============================================================================

def benchmark_dggs_polars_memory(raster_paths: List[Path], num_layers: int,
                                  bbox: Tuple, resolution: int) -> Dict[str, float]:
    """
    Benchmark DGGS method using Polars in-memory (no Parquet I/O overhead).
    This isolates the query engine performance.
    """
    if not HAS_POLARS:
        return {"error": "Polars not available"}
    
    paths = raster_paths[:num_layers]
    
    # Indexing phase
    index_start = time.perf_counter()
    
    first_raster = np.load(paths[0])
    base_cells = raster_to_h3_cells(first_raster, bbox, resolution)
    
    data = {'h3_cell': [c[0] for c in base_cells]}
    
    for i, path in enumerate(paths):
        raster = np.load(path)
        cells = raster_to_h3_cells(raster, bbox, resolution)
        data[f'value_{i}'] = [c[1] for c in cells]
    
    df = pl.DataFrame(data)
    
    index_time = time.perf_counter() - index_start
    
    # Classification phase
    classify_start = time.perf_counter()
    
    value_cols = [f'value_{i}' for i in range(num_layers)]
    
    result = df.with_columns([
        (pl.sum_horizontal([pl.col(c) * 10 for c in value_cols]))
        .cast(pl.Int64)
        .alias('sum_value')
    ])
    
    sum_values = result['sum_value'].to_numpy()
    classes = classify_array(sum_values)
    result = result.with_columns(pl.Series('class', classes))
    
    classify_time = time.perf_counter() - classify_start
    
    return {
        "index_time": index_time,
        "classify_time": classify_time,
        "total_time": index_time + classify_time,
    }


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def run_benchmark(output_dir: Path, num_layers_list: List[int] = None):
    """Run the complete benchmark comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    parquet_dir = output_dir / "parquet"
    
    if num_layers_list is None:
        num_layers_list = CONFIG["num_layers_list"]
    
    max_layers = max(num_layers_list)
    rng = np.random.default_rng(CONFIG["random_seed"])
    
    print(f"Code version: {CODE_VERSION}")
    print(f"Polars available: {HAS_POLARS}")
    print(f"Parquet available: {HAS_PARQUET}")
    print(f"Layers to test: {num_layers_list}")
    print("=" * 60)
    
    # Generate data
    print(f"\nGenerating {max_layers} raster layers...")
    raster_paths = generate_all_rasters(max_layers, CONFIG["raster_size"], rng, data_dir)
    
    # Run benchmarks
    results = []
    
    for n in num_layers_list:
        print(f"\n--- Benchmarking with {n} layers ---")
        
        row = {"num_layers": n}
        
        # Method 1: Traditional Raster
        print("  Running: Traditional Raster (NumPy)...")
        raster_result = benchmark_raster_method(raster_paths, n)
        row["raster_total"] = raster_result["total_time"]
        row["raster_warp"] = raster_result["warp_time"]
        row["raster_classify"] = raster_result["classify_time"]
        print(f"    Total: {raster_result['total_time']:.3f}s")
        
        # Method 2: DGGS with Pandas
        print("  Running: DGGS + Pandas...")
        pandas_result = benchmark_dggs_pandas(raster_paths, n, CONFIG["bbox"], 
                                               CONFIG["h3_resolution"])
        row["dggs_pandas_total"] = pandas_result["total_time"]
        row["dggs_pandas_index"] = pandas_result["index_time"]
        row["dggs_pandas_classify"] = pandas_result["classify_time"]
        print(f"    Total: {pandas_result['total_time']:.3f}s")
        
        # Method 3: DGGS with Polars + Parquet
        if HAS_POLARS and HAS_PARQUET:
            print("  Running: DGGS + Polars + Parquet...")
            polars_result = benchmark_dggs_polars(raster_paths, n, CONFIG["bbox"],
                                                   CONFIG["h3_resolution"], parquet_dir)
            row["dggs_polars_parquet_total"] = polars_result["total_time"]
            row["dggs_polars_parquet_index"] = polars_result["index_time"]
            row["dggs_polars_parquet_classify"] = polars_result["classify_time"]
            print(f"    Total: {polars_result['total_time']:.3f}s")
        
        # Method 4: DGGS with Polars (in-memory)
        if HAS_POLARS:
            print("  Running: DGGS + Polars (in-memory)...")
            polars_mem_result = benchmark_dggs_polars_memory(raster_paths, n, CONFIG["bbox"],
                                                              CONFIG["h3_resolution"])
            row["dggs_polars_mem_total"] = polars_mem_result["total_time"]
            row["dggs_polars_mem_index"] = polars_mem_result["index_time"]
            row["dggs_polars_mem_classify"] = polars_mem_result["classify_time"]
            print(f"    Total: {polars_mem_result['total_time']:.3f}s")
        
        results.append(row)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "raster_benchmark_comparison.csv", index=False)
    print(f"\nResults saved to {output_dir / 'raster_benchmark_comparison.csv'}")
    
    # Generate plots
    if HAS_MATPLOTLIB:
        plot_comparison(df, output_dir)
    
    # Print summary
    print_summary(df)
    
    return df


def plot_comparison(df: pd.DataFrame, output_dir: Path):
    """Generate comparison plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left panel: Total time comparison
    ax1 = axes[0]
    ax1.loglog(df['num_layers'], df['raster_total'], 'o-', 
               label='Raster (NumPy)', color='orange', linewidth=2, markersize=8)
    ax1.loglog(df['num_layers'], df['dggs_pandas_total'], 's-',
               label='DGGS + Pandas', color='blue', linewidth=2, markersize=8)
    
    if 'dggs_polars_parquet_total' in df.columns:
        ax1.loglog(df['num_layers'], df['dggs_polars_parquet_total'], '^-',
                   label='DGGS + Polars + Parquet', color='green', linewidth=2, markersize=8)
    
    if 'dggs_polars_mem_total' in df.columns:
        ax1.loglog(df['num_layers'], df['dggs_polars_mem_total'], 'd-',
                   label='DGGS + Polars (memory)', color='red', linewidth=2, markersize=8)
    
    ax1.set_xlabel('Number of input layers', fontsize=12)
    ax1.set_ylabel('Total compute time (s)', fontsize=12)
    ax1.set_title('Total Time Comparison\n(cf. Figure 7 in Law & Ardo 2024)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    # Right panel: Classification time only
    ax2 = axes[1]
    ax2.loglog(df['num_layers'], df['raster_classify'], 'o-',
               label='Raster (NumPy)', color='orange', linewidth=2, markersize=8)
    ax2.loglog(df['num_layers'], df['dggs_pandas_classify'], 's-',
               label='DGGS + Pandas', color='blue', linewidth=2, markersize=8)
    
    if 'dggs_polars_parquet_classify' in df.columns:
        ax2.loglog(df['num_layers'], df['dggs_polars_parquet_classify'], '^-',
                   label='DGGS + Polars + Parquet', color='green', linewidth=2, markersize=8)
    
    if 'dggs_polars_mem_classify' in df.columns:
        ax2.loglog(df['num_layers'], df['dggs_polars_mem_classify'], 'd-',
                   label='DGGS + Polars (memory)', color='red', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of input layers', fontsize=12)
    ax2.set_ylabel('Classification time only (s)', fontsize=12)
    ax2.set_title('Classification Time Comparison\n(isolating query engine performance)', fontsize=14)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / "raster_benchmark_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "raster_benchmark_comparison.pdf", bbox_inches='tight')
    print(f"Plots saved to {output_dir}")


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nAverage speedup ratios (vs Raster baseline):")
    
    raster_avg = df['raster_total'].mean()
    pandas_avg = df['dggs_pandas_total'].mean()
    
    print(f"  DGGS + Pandas:           {raster_avg/pandas_avg:.2f}x {'faster' if pandas_avg < raster_avg else 'slower'}")
    
    if 'dggs_polars_parquet_total' in df.columns:
        polars_parquet_avg = df['dggs_polars_parquet_total'].mean()
        print(f"  DGGS + Polars + Parquet: {raster_avg/polars_parquet_avg:.2f}x {'faster' if polars_parquet_avg < raster_avg else 'slower'}")
    
    if 'dggs_polars_mem_total' in df.columns:
        polars_mem_avg = df['dggs_polars_mem_total'].mean()
        print(f"  DGGS + Polars (memory):  {raster_avg/polars_mem_avg:.2f}x {'faster' if polars_mem_avg < raster_avg else 'slower'}")
    
    print("\nClassification time only (isolating query engine):")
    raster_class_avg = df['raster_classify'].mean()
    pandas_class_avg = df['dggs_pandas_classify'].mean()
    
    print(f"  DGGS + Pandas:           {raster_class_avg/pandas_class_avg:.2f}x {'faster' if pandas_class_avg < raster_class_avg else 'slower'}")
    
    if 'dggs_polars_parquet_classify' in df.columns:
        polars_parquet_class_avg = df['dggs_polars_parquet_classify'].mean()
        print(f"  DGGS + Polars + Parquet: {raster_class_avg/polars_parquet_class_avg:.2f}x {'faster' if polars_parquet_class_avg < raster_class_avg else 'slower'}")
    
    if 'dggs_polars_mem_classify' in df.columns:
        polars_mem_class_avg = df['dggs_polars_mem_classify'].mean()
        print(f"  DGGS + Polars (memory):  {raster_class_avg/polars_mem_class_avg:.2f}x {'faster' if polars_mem_class_avg < raster_class_avg else 'slower'}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Raster benchmark with Polars/Parquet comparison"
    )
    parser.add_argument("--output", "-o", type=str, default="results_polars",
                        help="Output directory")
    parser.add_argument("--layers", "-l", type=str, default=None,
                        help="Comma-separated list of layer counts (e.g., '10,50,100')")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    if args.layers:
        num_layers_list = [int(x.strip()) for x in args.layers.split(",")]
    else:
        # Default quick test
        num_layers_list = [10, 50, 100, 500]
    
    run_benchmark(output_dir, num_layers_list)


if __name__ == "__main__":
    main()
