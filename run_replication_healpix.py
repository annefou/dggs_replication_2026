#!/usr/bin/env python3
"""
DGGS Benchmark: Multi-DGGS Comparison Study

Extends the replication of Law & Ardo (2024) to compare multiple DGGS:
- H3 (Uber's hexagonal grid) - original paper
- HEALPix (Hierarchical Equal Area isoLatitude Pixelization) - extension

Original paper:
"Using a discrete global grid system for a scalable, interoperable, 
and reproducible system of land-use mapping"
DOI: 10.1080/20964471.2024.2429847

DGGS COMPARISON:
| DGGS    | Cell Shape     | Equal Area | Hierarchical | Primary Use      |
|---------|----------------|------------|--------------|------------------|
| H3      | Hexagons       | ~Approx    | Yes (7:1)    | Industry/Uber    |
| HEALPix | Quadrilaterals | Yes        | Yes (4:1)    | Astronomy/CMB    |

Author: Anne Fouilloux
Date: 2026-01-22
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

# Optional imports
try:
    import xdggs
    HAS_XDGGS = True
except ImportError:
    HAS_XDGGS = False

try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# =============================================================================
# DGGS Configuration and Abstraction
# =============================================================================

class DGGSType(Enum):
    """Supported DGGS types."""
    H3 = "h3"
    HEALPIX = "healpix"


@dataclass
class DGGSConfig:
    """Configuration for a specific DGGS."""
    name: str
    dggs_type: DGGSType
    resolution: int
    description: str
    avg_cell_area_km2: float  # Approximate for reference


# Resolution equivalence table (approximate cell areas)
# Goal: Compare DGGS at similar spatial resolutions
#
# H3 cell areas: https://h3geo.org/docs/core-library/restable/
# HEALPix: Npix = 12 * 4^level, so area = 4π steradians / Npix
#          In km²: area ≈ 510.1e6 km² / (12 * 4^level)
#
# CORRECTED TABLE - previous version had wrong HEALPix levels!
RESOLUTION_EQUIVALENCE = {
    # H3 res -> HEALPix level (matched by approximate cell area in km²)
    # 
    # H3 res | H3 area (km²) | HEALPix level | HEALPix area (km²) | Npix
    # -------|---------------|---------------|--------------------|---------
    #   5    |    252.9      |      8        |      ~162          | 786,432
    #   6    |     36.1      |      9        |      ~40           | 3,145,728
    #   7    |      5.16     |     10        |      ~10           | 12,582,912
    #   8    |      0.737    |     12        |      ~0.63         | 201,326,592
    #   9    |      0.105    |     13        |      ~0.16         | 805,306,368
    #  10    |      0.015    |     15        |      ~0.01         | 12,884,901,888
    #
    5: 8,    # H3:  252.9 km² vs HEALPix level 8:  ~162 km²
    6: 9,    # H3:   36.1 km² vs HEALPix level 9:  ~40 km²
    7: 10,   # H3:    5.16 km² vs HEALPix level 10: ~10 km²
    8: 12,   # H3:    0.74 km² vs HEALPix level 12: ~0.63 km²
    9: 13,   # H3:    0.105 km² vs HEALPix level 13: ~0.16 km²
    10: 15,  # H3:   0.015 km² vs HEALPix level 15: ~0.01 km²
    11: 16,  # H3:  0.0022 km² vs HEALPix level 16: ~0.0025 km²
    12: 17,  # H3: 0.00031 km² vs HEALPix level 17: ~0.0006 km²
}


def get_equivalent_healpix_level(h3_resolution: int) -> int:
    """Get HEALPix level approximately equivalent to H3 resolution."""
    return RESOLUTION_EQUIVALENCE.get(h3_resolution, h3_resolution)


def get_dggs_info(dggs_type: DGGSType, resolution: int):
    """
    Factory function to create xdggs info objects.
    
    Returns the appropriate xdggs object for coordinate-to-cell conversion.
    """
    if not HAS_XDGGS:
        raise ImportError("xdggs is required for DGGS operations")
    
    if dggs_type == DGGSType.H3:
        return xdggs.H3Info(level=resolution)
    elif dggs_type == DGGSType.HEALPIX:
        return xdggs.HealpixInfo(level=resolution, indexing_scheme="nested")
    else:
        raise ValueError(f"Unknown DGGS type: {dggs_type}")


# =============================================================================
# Configuration
# =============================================================================

CODE_VERSION = "2026-01-22-healpix-v1"

CONFIG = {
    "random_seed": 42,
    
    "raster": {
        "h3_resolution": 9,
        "healpix_level": None,  # None = auto from h3_resolution
        "num_layers_list": [10, 50, 100, 500, 1000],
        "raster_size": (100, 100),
        "bbox": (-0.5, -0.5, 0.5, 0.5),
    },
    
    # DGGS types to benchmark
    "dggs_types": ["h3", "healpix"],
}


# =============================================================================
# Classification Functions (Paper Section 3.2.1)
# =============================================================================

@lru_cache(maxsize=100000)
def classify_value(sum_value: int) -> int:
    """7-bit classification based on number properties."""
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0: return False
        return True
    
    def is_perfect(n):
        if n < 2: return False
        s = sum(i for i in range(1, n) if n % i == 0)
        return s == n
    
    def is_triangular(n):
        if n < 0: return False
        k = int((2 * n) ** 0.5)
        return k * (k + 1) // 2 == n
    
    def is_square(n):
        if n < 0: return False
        r = int(n ** 0.5)
        return r * r == n
    
    def is_pentagonal(n):
        if n < 0: return False
        k = (1 + (1 + 24 * n) ** 0.5) / 6
        return k == int(k) and k > 0
    
    def is_hexagonal(n):
        if n < 0: return False
        k = (1 + (1 + 8 * n) ** 0.5) / 4
        return k == int(k) and k > 0
    
    def is_fibonacci(n):
        if n < 0: return False
        def is_sq(x): s = int(x**0.5); return s*s == x
        return is_sq(5*n*n + 4) or is_sq(5*n*n - 4)
    
    return (
        (1 if is_prime(sum_value) else 0) |
        (1 if is_perfect(sum_value) else 0) << 1 |
        (1 if is_triangular(sum_value) else 0) << 2 |
        (1 if is_square(sum_value) else 0) << 3 |
        (1 if is_pentagonal(sum_value) else 0) << 4 |
        (1 if is_hexagonal(sum_value) else 0) << 5 |
        (1 if is_fibonacci(sum_value) else 0) << 6
    )


# =============================================================================
# System Information
# =============================================================================

def get_system_info() -> Dict:
    """Collect system and dependency information."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "code_version": CODE_VERSION,
        "python_version": sys.version.split()[0],
        "paper": {
            "doi": "10.1080/20964471.2024.2429847",
            "authors": "Law, R.M. & Ardo, J.",
            "year": 2024,
        },
        "extension": {
            "description": "Multi-DGGS comparison (H3 vs HEALPix)",
            "dggs_types": ["H3", "HEALPix"],
        },
        "dependencies": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "h3": h3.__version__ if HAS_H3 and hasattr(h3, '__version__') else ("available" if HAS_H3 else "NOT INSTALLED"),
            "xdggs": "available" if HAS_XDGGS else "NOT INSTALLED",
        },
    }
    if HAS_PSUTIL:
        info["system"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        }
    return info


# =============================================================================
# Data Generation
# =============================================================================

def generate_raster_layer(size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Generate a spatially-correlated raster layer."""
    base = rng.uniform(0, 1, size)
    if HAS_SCIPY:
        smoothed = gaussian_filter(base, sigma=2)
        return (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-10)
    return base


# =============================================================================
# INDEXING METHODS - Unified interface for multiple DGGS
# =============================================================================

def index_raster_h3_loop(lats: np.ndarray, lngs: np.ndarray, 
                         resolution: int) -> np.ndarray:
    """
    Index raster to H3 using LOOP-based conversion (REPRODUCTION).
    
    This is the paper's approach - calling h3.latlng_to_cell() for each pixel.
    """
    if not HAS_H3:
        raise ImportError("h3 library not available")
    
    cell_ids = np.array([
        h3.latlng_to_cell(lat, lng, resolution)
        for lat, lng in zip(lats.ravel(), lngs.ravel())
    ])
    return cell_ids


def index_raster_xdggs(lats: np.ndarray, lngs: np.ndarray,
                       dggs_type: DGGSType, resolution: int) -> np.ndarray:
    """
    Index raster to DGGS using xdggs VECTORIZED conversion.
    
    This works for both H3 and HEALPix through the unified xdggs interface.
    
    Parameters:
    -----------
    lats : np.ndarray
        Latitude values
    lngs : np.ndarray  
        Longitude values
    dggs_type : DGGSType
        Which DGGS to use (H3 or HEALPix)
    resolution : int
        Resolution/level for the DGGS
        
    Returns:
    --------
    np.ndarray
        Cell IDs (integers for HEALPix, strings for H3)
    """
    if not HAS_XDGGS:
        raise ImportError("xdggs not available")
    
    dggs_info = get_dggs_info(dggs_type, resolution)
    cell_ids = dggs_info.geographic2cell_ids(lngs.ravel(), lats.ravel())
    return np.asarray(cell_ids)


def aggregate_to_cells(values: np.ndarray, cell_ids: np.ndarray, 
                       unique_cells: np.ndarray) -> np.ndarray:
    """Aggregate pixel values to DGGS cells (mean per cell)."""
    # Create mapping from cell_id to index
    cell_to_idx = {cell: idx for idx, cell in enumerate(unique_cells)}
    indices = np.array([cell_to_idx[cell] for cell in cell_ids])
    
    num_cells = len(unique_cells)
    sums = np.bincount(indices, weights=values.ravel(), minlength=num_cells)
    counts = np.bincount(indices, minlength=num_cells)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nan_to_num(sums / counts, nan=0.0)


# =============================================================================
# RASTER BENCHMARK - Multi-DGGS Comparison
# =============================================================================

def run_raster_benchmark(config: Dict, output_dir: Path, 
                         dggs_types: List[str] = None) -> pd.DataFrame:
    """
    Run raster benchmark comparing multiple DGGS.
    
    Methods compared:
    1. Traditional Raster (NumPy) - BASELINE
    2. H3 via xdggs - REPRODUCTION
    3. HEALPix via xdggs - EXTENSION
    4. Pre-indexed versions of each DGGS
    """
    from tqdm import tqdm
    
    if dggs_types is None:
        dggs_types = config.get("dggs_types", ["h3", "healpix"])
    
    print("\n" + "=" * 70)
    print("RASTER BENCHMARK - Multi-DGGS Comparison")
    print("=" * 70)
    print("\nDGGS types to compare:")
    for dt in dggs_types:
        print(f"  - {dt.upper()}")
    
    rng = np.random.default_rng(config["random_seed"])
    cfg = config["raster"]
    max_layers = max(cfg["num_layers_list"])
    h3_resolution = cfg["h3_resolution"]
    
    # Generate data
    print(f"\nGenerating {max_layers} raster layers...")
    rasters = np.stack([generate_raster_layer(cfg["raster_size"], rng) 
                        for _ in tqdm(range(max_layers))])
    
    # Pre-compute coordinate grids
    rows, cols = cfg["raster_size"]
    minx, miny, maxx, maxy = cfg["bbox"]
    lngs = minx + (np.arange(cols) + 0.5) * (maxx - minx) / cols
    lats = miny + (np.arange(rows) + 0.5) * (maxy - miny) / rows
    lng_grid, lat_grid = np.meshgrid(lngs, lats)
    
    # ==========================================================================
    # INDEXING BENCHMARK - Compare DGGS indexing performance
    # ==========================================================================
    print("\n" + "-" * 70)
    print("INDEXING BENCHMARK (coordinate → cell conversion)")
    print("-" * 70)
    
    indexing_results = {}
    cell_ids_by_dggs = {}
    unique_cells_by_dggs = {}
    
    for dggs_name in dggs_types:
        dggs_type = DGGSType(dggs_name)
        
        # Determine resolution
        if dggs_type == DGGSType.H3:
            resolution = h3_resolution
        else:
            # Use explicit healpix_level if set, otherwise auto-calculate
            if cfg.get("healpix_level") is not None:
                resolution = cfg["healpix_level"]
            else:
                resolution = get_equivalent_healpix_level(h3_resolution)
        
        print(f"\n  {dggs_name.upper()} (resolution/level {resolution}):")
        
        # xdggs indexing
        print(f"    xdggs indexing...", end=" ", flush=True)
        start = time.perf_counter()
        cell_ids = index_raster_xdggs(lat_grid, lng_grid, dggs_type, resolution)
        index_time = time.perf_counter() - start
        print(f"{index_time:.4f}s")
        
        unique_cells = np.unique(cell_ids)
        num_cells = len(unique_cells)
        print(f"    Grid: {rows}x{cols} pixels → {num_cells} cells")
        
        cell_ids_by_dggs[dggs_name] = cell_ids
        unique_cells_by_dggs[dggs_name] = unique_cells
        
        indexing_results[dggs_name] = {
            "resolution": resolution,
            "index_time": index_time,
            "num_cells": num_cells,
            "pixels": rows * cols,
        }
    
    # Compare indexing times
    if len(dggs_types) > 1:
        times = [indexing_results[dt]["index_time"] for dt in dggs_types]
        fastest = dggs_types[np.argmin(times)]
        print(f"\n  Fastest indexing: {fastest.upper()}")
    
    # ==========================================================================
    # PRE-INDEX ALL LAYERS
    # ==========================================================================
    print("\n" + "-" * 70)
    print("PRE-INDEXING ALL LAYERS")
    print("-" * 70)
    
    preindexed_by_dggs = {}
    
    for dggs_name in dggs_types:
        cell_ids = cell_ids_by_dggs[dggs_name]
        unique_cells = unique_cells_by_dggs[dggs_name]
        num_cells = len(unique_cells)
        
        print(f"\n  Pre-indexing {max_layers} layers for {dggs_name.upper()}...", end=" ", flush=True)
        start = time.perf_counter()
        preindexed = np.zeros((num_cells, max_layers), dtype=np.float32)
        for i in range(max_layers):
            preindexed[:, i] = aggregate_to_cells(rasters[i], cell_ids, unique_cells)
        preindex_time = time.perf_counter() - start
        print(f"{preindex_time:.3f}s")
        
        preindexed_by_dggs[dggs_name] = preindexed
        indexing_results[dggs_name]["preindex_time"] = preindex_time
    
    # ==========================================================================
    # CLASSIFICATION BENCHMARK
    # ==========================================================================
    print("\n" + "-" * 70)
    print("CLASSIFICATION BENCHMARK")
    print("-" * 70)
    
    results = []
    
    for n in cfg["num_layers_list"]:
        print(f"\n--- {n} layers ---")
        row = {"num_layers": n}
        data = rasters[:n]
        
        # 1. Traditional Raster (NumPy) - BASELINE
        start = time.perf_counter()
        stacked = data.copy()
        warp_time = time.perf_counter() - start
        
        classify_start = time.perf_counter()
        sum_vals = (stacked * 10).astype(int).sum(axis=0)
        classified = np.vectorize(classify_value)(sum_vals)
        classify_time = time.perf_counter() - classify_start
        
        row["raster_warp"] = warp_time
        row["raster_classify"] = classify_time
        row["raster_total"] = warp_time + classify_time
        print(f"  Raster (NumPy):     {row['raster_total']:.4f}s")
        
        # 2. DGGS methods (for each type)
        for dggs_name in dggs_types:
            dggs_type = DGGSType(dggs_name)
            cell_ids = cell_ids_by_dggs[dggs_name]
            unique_cells = unique_cells_by_dggs[dggs_name]
            num_cells = len(unique_cells)
            preindexed = preindexed_by_dggs[dggs_name]
            
            if dggs_type == DGGSType.H3:
                resolution = h3_resolution
            else:
                # Use explicit healpix_level if set, otherwise auto-calculate
                if cfg.get("healpix_level") is not None:
                    resolution = cfg["healpix_level"]
                else:
                    resolution = get_equivalent_healpix_level(h3_resolution)
            
            # Index each layer (simulates fresh data)
            start = time.perf_counter()
            cell_values = np.zeros((num_cells, n), dtype=np.float32)
            for i in range(n):
                layer_cell_ids = index_raster_xdggs(lat_grid, lng_grid, dggs_type, resolution)
                cell_values[:, i] = aggregate_to_cells(data[i], layer_cell_ids, unique_cells)
            index_time = time.perf_counter() - start
            
            classify_start = time.perf_counter()
            sums = (cell_values * 10).astype(int).sum(axis=1)
            classes = np.array([classify_value(int(v)) for v in sums])
            classify_time = time.perf_counter() - classify_start
            
            row[f"{dggs_name}_index"] = index_time
            row[f"{dggs_name}_classify"] = classify_time
            row[f"{dggs_name}_total"] = index_time + classify_time
            print(f"  {dggs_name.upper():8s} (xdggs):   {row[f'{dggs_name}_total']:.4f}s")
            
            # Pre-indexed version
            start = time.perf_counter()
            pre_data = preindexed[:, :n].copy()
            read_time = time.perf_counter() - start
            
            classify_start = time.perf_counter()
            sums = (pre_data * 10).astype(int).sum(axis=1)
            classes = np.array([classify_value(int(v)) for v in sums])
            classify_time = time.perf_counter() - classify_start
            
            row[f"{dggs_name}_preindex_read"] = read_time
            row[f"{dggs_name}_preindex_classify"] = classify_time
            row[f"{dggs_name}_preindex_total"] = read_time + classify_time
            print(f"  {dggs_name.upper():8s} (pre-idx): {row[f'{dggs_name}_preindex_total']:.4f}s")
        
        results.append(row)
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "raster_benchmark_multi_dggs.csv", index=False)
    
    with open(output_dir / "indexing_benchmark_multi_dggs.json", 'w') as f:
        json.dump(indexing_results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print("\n📊 INDEXING PERFORMANCE:")
    for dggs_name in dggs_types:
        idx = indexing_results[dggs_name]
        print(f"   {dggs_name.upper():8s}: {idx['index_time']:.4f}s for {idx['pixels']} pixels → {idx['num_cells']} cells")
    
    print("\n📊 CLASSIFICATION (pre-indexed):")
    for _, row in df.iterrows():
        n = int(row['num_layers'])
        print(f"   {n:5d} layers:")
        print(f"         Raster: {row['raster_classify']:.4f}s")
        for dggs_name in dggs_types:
            print(f"         {dggs_name.upper():8s}: {row[f'{dggs_name}_preindex_classify']:.4f}s")
    
    return df


# =============================================================================
# Plotting
# =============================================================================

def plot_results(df: pd.DataFrame, dggs_types: List[str], output_dir: Path):
    """Generate comparison plots for multi-DGGS benchmark."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    # Color scheme
    colors = {
        "raster": "orange",
        "h3": "blue",
        "healpix": "green",
    }
    
    markers = {
        "raster": "o",
        "h3": "s",
        "healpix": "^",
    }
    
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("DGGS Benchmark: H3 vs HEALPix Comparison\n"
                 "Extension of Law & Ardo (2024) DOI: 10.1080/20964471.2024.2429847", 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Total time (with indexing)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.loglog(df['num_layers'], df['raster_total'], 
               f'{markers["raster"]}-', label='Raster (baseline)', 
               color=colors["raster"], linewidth=2, markersize=8)
    
    for dggs_name in dggs_types:
        ax1.loglog(df['num_layers'], df[f'{dggs_name}_total'],
                   f'{markers.get(dggs_name, "d")}--', 
                   label=f'{dggs_name.upper()} (with indexing)', 
                   color=colors.get(dggs_name, "gray"), linewidth=2, markersize=6)
    
    ax1.set_xlabel('Number of layers')
    ax1.set_ylabel('Total time (s)')
    ax1.set_title('Total Time (including indexing)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, which='both')
    
    # Plot 2: Pre-indexed classification only
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.loglog(df['num_layers'], df['raster_classify'],
               f'{markers["raster"]}-', label='Raster', 
               color=colors["raster"], linewidth=2, markersize=8)
    
    for dggs_name in dggs_types:
        ax2.loglog(df['num_layers'], df[f'{dggs_name}_preindex_classify'],
                   f'{markers.get(dggs_name, "d")}-', 
                   label=f'{dggs_name.upper()} (pre-indexed)', 
                   color=colors.get(dggs_name, "gray"), linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of layers')
    ax2.set_ylabel('Classification time (s)')
    ax2.set_title('Classification Only (Pre-indexed)\n"Paper\'s key comparison"')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: DGGS comparison (H3 vs HEALPix)
    ax3 = fig.add_subplot(2, 2, 3)
    
    for dggs_name in dggs_types:
        ax3.loglog(df['num_layers'], df[f'{dggs_name}_total'],
                   f'{markers.get(dggs_name, "d")}--', 
                   label=f'{dggs_name.upper()} (total)', 
                   color=colors.get(dggs_name, "gray"), linewidth=2, markersize=6)
        ax3.loglog(df['num_layers'], df[f'{dggs_name}_preindex_total'],
                   f'{markers.get(dggs_name, "d")}-', 
                   label=f'{dggs_name.upper()} (pre-indexed)', 
                   color=colors.get(dggs_name, "gray"), linewidth=2, markersize=8)
    
    ax3.set_xlabel('Number of layers')
    ax3.set_ylabel('Time (s)')
    ax3.set_title('H3 vs HEALPix Comparison')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    
    # Plot 4: Speedup ratio
    ax4 = fig.add_subplot(2, 2, 4)
    
    for dggs_name in dggs_types:
        speedup = df['raster_classify'] / df[f'{dggs_name}_preindex_classify']
        ax4.plot(df['num_layers'], speedup,
                 f'{markers.get(dggs_name, "d")}-', 
                 label=f'{dggs_name.upper()} vs Raster', 
                 color=colors.get(dggs_name, "gray"), linewidth=2, markersize=8)
    
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Equal performance')
    ax4.set_xlabel('Number of layers')
    ax4.set_ylabel('Speedup (Raster time / DGGS time)')
    ax4.set_title('Classification Speedup vs Raster')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "benchmark_multi_dggs.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "benchmark_multi_dggs.pdf", bbox_inches='tight')
    print(f"\nPlots saved to {output_dir}")


# =============================================================================
# Summary Generation
# =============================================================================

def generate_summary(df: pd.DataFrame, dggs_types: List[str], 
                     indexing_results: Dict, output_dir: Path) -> Dict:
    """Generate summary of multi-DGGS comparison."""
    
    summary = {
        "paper": {
            "doi": "10.1080/20964471.2024.2429847",
            "title": "Using a discrete global grid system for a scalable, interoperable, and reproducible system of land-use mapping",
            "claim": "DGGS and raster methods show roughly equivalent performance",
        },
        "extension": {
            "description": "Multi-DGGS comparison: H3 vs HEALPix",
            "dggs_types": dggs_types,
        },
        "indexing": indexing_results,
        "results": {},
    }
    
    # Compare classification performance
    raster_avg = df['raster_classify'].mean()
    
    for dggs_name in dggs_types:
        dggs_avg = df[f'{dggs_name}_preindex_classify'].mean()
        ratio = raster_avg / dggs_avg
        
        summary["results"][dggs_name] = {
            "avg_classify_time": f"{dggs_avg:.4f}s",
            "vs_raster_ratio": f"{ratio:.2f}x",
            "validates_paper_claim": bool(0.3 < ratio < 3.0),
        }
    
    # Compare H3 vs HEALPix
    if "h3" in dggs_types and "healpix" in dggs_types:
        h3_avg = df['h3_preindex_classify'].mean()
        healpix_avg = df['healpix_preindex_classify'].mean()
        ratio = h3_avg / healpix_avg
        
        summary["results"]["h3_vs_healpix"] = {
            "ratio": f"{ratio:.2f}x",
            "faster": "H3" if ratio > 1 else "HEALPix" if ratio < 1 else "Equal",
        }
    
    with open(output_dir / "summary_multi_dggs.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"\n📄 Paper claim: {summary['paper']['claim']}")
    
    print("\n✅ RESULTS:")
    for dggs_name in dggs_types:
        r = summary['results'][dggs_name]
        status = "VALIDATED" if r['validates_paper_claim'] else "NOT VALIDATED"
        print(f"   {dggs_name.upper():8s}: {r['vs_raster_ratio']} vs raster → {status}")
    
    if "h3_vs_healpix" in summary["results"]:
        comp = summary["results"]["h3_vs_healpix"]
        print(f"\n📊 H3 vs HEALPix: {comp['faster']} is faster ({comp['ratio']})")
    
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DGGS Benchmark: Multi-DGGS Comparison (H3 vs HEALPix)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Extension of Law & Ardo (2024) replication to compare multiple DGGS.
DOI: 10.1080/20964471.2024.2429847

Supported DGGS:
  - H3 (Uber's hexagonal grid)
  - HEALPix (Hierarchical Equal Area isoLatitude Pixelization)

Examples:
  python run_replication_healpix.py --all
  python run_replication_healpix.py --dggs h3,healpix
  python run_replication_healpix.py --dggs healpix --raster-layers 10,50,100
        """
    )
    parser.add_argument("--all", action="store_true", 
                        help="Run benchmark with all DGGS types")
    parser.add_argument("--output", "-o", default="results",
                        help="Output directory (default: results)")
    parser.add_argument("--dggs", type=str, default="h3,healpix",
                        help="Comma-separated DGGS types to benchmark (default: h3,healpix)")
    parser.add_argument("--raster-layers", type=str, default=None,
                        help="Comma-separated raster layer counts")
    parser.add_argument("--h3-resolution", type=int, default=None,
                        help="H3 resolution (default: 9)")
    parser.add_argument("--healpix-level", type=int, default=None,
                        help="HEALPix level (default: auto from H3 resolution)")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse DGGS types
    dggs_types = [dt.strip().lower() for dt in args.dggs.split(",")]
    valid_types = {"h3", "healpix"}
    for dt in dggs_types:
        if dt not in valid_types:
            print(f"Error: Unknown DGGS type '{dt}'. Valid types: {valid_types}")
            sys.exit(1)
    
    # Apply CLI arguments
    if args.raster_layers:
        CONFIG["raster"]["num_layers_list"] = [int(x) for x in args.raster_layers.split(",")]
    if args.h3_resolution:
        CONFIG["raster"]["h3_resolution"] = args.h3_resolution
    if args.healpix_level:
        CONFIG["raster"]["healpix_level"] = args.healpix_level
    
    CONFIG["dggs_types"] = dggs_types
    
    # Determine actual HEALPix level to use
    h3_res = CONFIG["raster"]["h3_resolution"]
    if CONFIG["raster"].get("healpix_level") is not None:
        healpix_level = CONFIG["raster"]["healpix_level"]
        healpix_source = "(manual override)"
    else:
        healpix_level = get_equivalent_healpix_level(h3_res)
        healpix_source = "(auto from H3)"
    
    # Header
    print("=" * 70)
    print("DGGS BENCHMARK: Multi-DGGS Comparison Study")
    print("=" * 70)
    print(f"Code version: {CODE_VERSION}")
    print(f"\nConfiguration:")
    print(f"  Output directory: {output_dir}")
    print(f"  DGGS types:       {dggs_types}")
    print(f"  Random seed:      {CONFIG['random_seed']}")
    print(f"  Raster layers:    {CONFIG['raster']['num_layers_list']}")
    print(f"  H3 resolution:    {h3_res}")
    print(f"  HEALPix level:    {healpix_level} {healpix_source}")
    print(f"\nDependencies:")
    print(f"  xdggs:  {'✅ Available' if HAS_XDGGS else '❌ Not installed (REQUIRED)'}")
    print(f"  h3:     {'✅ Available' if HAS_H3 else '⚠️  Not installed (optional)'}")
    print(f"  SciPy:  {'✅ Available' if HAS_SCIPY else '❌ Not installed'}")
    
    if not HAS_XDGGS:
        print("\n❌ ERROR: xdggs is required for this benchmark.")
        print("   Install with: pip install xdggs")
        sys.exit(1)
    
    # System info
    sys_info = get_system_info()
    sys_info["configuration"] = {
        "dggs_types": dggs_types,
        "raster_layers": CONFIG["raster"]["num_layers_list"],
        "h3_resolution": h3_res,
        "healpix_level": healpix_level,
        "healpix_source": healpix_source,
        "random_seed": CONFIG["random_seed"],
    }
    with open(output_dir / "system_info.json", 'w') as f:
        json.dump(sys_info, f, indent=2)
    
    # Run benchmark
    df = run_raster_benchmark(CONFIG, output_dir, dggs_types)
    
    # Load indexing results for summary
    with open(output_dir / "indexing_benchmark_multi_dggs.json") as f:
        indexing_results = json.load(f)
    
    # Generate outputs
    plot_results(df, dggs_types, output_dir)
    generate_summary(df, dggs_types, indexing_results, output_dir)
    
    print(f"\n📁 Results saved to: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        print(f"   {f.name}")


if __name__ == "__main__":
    main()
