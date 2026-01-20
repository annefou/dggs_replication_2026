#!/usr/bin/env python3
"""
DGGS Benchmark Replication Study

This script replicates the benchmarks from:
Law & Ardo (2024) - "Using a discrete global grid system for a scalable,
interoperable, and reproducible system of land-use mapping"
https://doi.org/10.1080/20964471.2024.2429847

Original benchmark code version is defined in config.env

Usage:
    python run_replication.py --all              # Run all benchmarks
    python run_replication.py --vector           # Run vector benchmark only (Figure 6)
    python run_replication.py --raster           # Run raster benchmark only (Figure 7)
    python run_replication.py --generate-data    # Generate synthetic data only
    python run_replication.py --compare          # Compare results with paper
    python run_replication.py --explore          # Explore original benchmark repository
    python run_replication.py --run-original     # Run original benchmark scripts

Author: Anne Fouilloux
Date: 2026-01-17
"""

import os
import sys
import json
import time
import random
import argparse
import platform
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import h3
from tqdm import tqdm
import xdggs

# Try to import optional dependencies
try:
    import nlmpy
    HAS_NLMPY = True
except ImportError:
    HAS_NLMPY = False
    print("Warning: nlmpy not available. Raster benchmarks will use alternative method.")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# Load Configuration from config.env
# =============================================================================

def load_config_env(config_path: Path = None) -> Dict[str, str]:
    """Load configuration from config.env file."""
    if config_path is None:
        # Look for config.env in script directory or current directory
        script_dir = Path(__file__).parent
        config_path = script_dir / "config.env"
        if not config_path.exists():
            config_path = Path("config.env")
    
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes
                    value = value.strip('"').strip("'")
                    config[key] = value
    
    return config

# Load config
_env_config = load_config_env()

# Validate required configuration
_required_config = ["DGGS_BENCHMARKS_REPO", "DGGS_BENCHMARKS_VERSION"]
_missing = [k for k in _required_config if not _env_config.get(k) and not os.environ.get(k)]
if _missing:
    print(f"ERROR: Missing required configuration: {_missing}")
    print("Ensure config.env exists and contains these variables, or set them as environment variables.")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================

# Code version marker - change this to verify image rebuild
CODE_VERSION = "2026-01-19-v3"

def parse_layer_list(value: str) -> List[int]:
    """Parse comma-separated layer list string to list of ints."""
    if not value:
        return []
    return [int(x.strip()) for x in value.split(',')]

CONFIG = {
    "random_seed": int(os.environ.get("RANDOM_SEED", _env_config.get("RANDOM_SEED", "42"))),
    "h3_resolution": int(os.environ.get("H3_RESOLUTION", _env_config.get("H3_RESOLUTION", "9"))),
    
    # Original benchmark reference - MUST come from config.env
    "original_benchmark": {
        "repo": os.environ.get("DGGS_BENCHMARKS_REPO", _env_config.get("DGGS_BENCHMARKS_REPO")),
        "version": os.environ.get("DGGS_BENCHMARKS_VERSION", _env_config.get("DGGS_BENCHMARKS_VERSION")),
    },
    
    # Original paper reference
    "original_paper": {
        "doi": os.environ.get("ORIGINAL_PAPER_DOI", _env_config.get("ORIGINAL_PAPER_DOI")),
        "title": os.environ.get("ORIGINAL_PAPER_TITLE", _env_config.get("ORIGINAL_PAPER_TITLE")),
    },
    
    # Vector benchmark parameters (Figure 6)
    # Layer counts from config.env, with paper defaults as fallback
    "vector": {
        "num_layers_list": parse_layer_list(
            os.environ.get("VECTOR_LAYERS", _env_config.get("VECTOR_LAYERS", "10,20,50,100,200,500,1000"))
        ),
        "num_polygons_per_layer": int(
            os.environ.get("POLYGONS_PER_LAYER", _env_config.get("POLYGONS_PER_LAYER", "100"))
        ),
        "bbox": (-180, -85, 180, 85),  # Global extent
        "max_layers_before_failure": 500,  # Expected failure point for vector method
    },
    
    # Raster benchmark parameters (Figure 7)
    # Layer counts from config.env, with paper defaults as fallback
    "raster": {
        "num_layers_list": parse_layer_list(
            os.environ.get("RASTER_LAYERS", _env_config.get("RASTER_LAYERS", "10,50,100,500,1000,5000,10000"))
        ),
        "raster_size": (100, 100),  # 100x100 pixels per layer
        "num_classes": 10,
    },
    
    # Output
    "results_dir": "results",
    "data_dir": "data",
}


# =============================================================================
# System Information
# =============================================================================

def get_system_info() -> Dict:
    """Collect system information for reproducibility documentation."""
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        
        # Original benchmark being replicated
        "original_benchmark": CONFIG["original_benchmark"],
        "original_paper": CONFIG["original_paper"],
    }
    
    if HAS_PSUTIL:
        info["cpu_count"] = psutil.cpu_count()
        info["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        info["memory_available_gb"] = round(psutil.virtual_memory().available / (1024**3), 2)
    
    # Package versions
    info["packages"] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "geopandas": gpd.__version__,
        "h3": h3.__version__ if hasattr(h3, '__version__') else "unknown",
    }
    
    return info


# =============================================================================
# Data Generation
# =============================================================================

def generate_random_polygon(bbox: Tuple[float, float, float, float], 
                            num_vertices: int = 6,
                            rng: np.random.Generator = None) -> Polygon:
    """Generate a random polygon within bounding box."""
    if rng is None:
        rng = np.random.default_rng()
    
    minx, miny, maxx, maxy = bbox
    
    # Generate random center
    cx = rng.uniform(minx + 1, maxx - 1)
    cy = rng.uniform(miny + 1, maxy - 1)
    
    # Generate random vertices around center
    angles = np.sort(rng.uniform(0, 2 * np.pi, num_vertices))
    radii = rng.uniform(0.5, 2.0, num_vertices)
    
    vertices = []
    for angle, radius in zip(angles, radii):
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        # Clamp to bbox
        x = max(minx, min(maxx, x))
        y = max(miny, min(maxy, y))
        vertices.append((x, y))
    
    # Close the polygon
    vertices.append(vertices[0])
    
    try:
        poly = Polygon(vertices)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except:
        # Fallback to simple box
        return Polygon([
            (cx - 0.5, cy - 0.5),
            (cx + 0.5, cy - 0.5),
            (cx + 0.5, cy + 0.5),
            (cx - 0.5, cy + 0.5),
            (cx - 0.5, cy - 0.5),
        ])


def generate_voronoi_layer(num_polygons: int, 
                           bbox: Tuple[float, float, float, float],
                           rng: np.random.Generator) -> gpd.GeoDataFrame:
    """
    Generate a layer of random polygons (simulating Voronoi-like regions).
    
    The paper uses Voronoi polygons for the vector benchmark.
    This is a simplified version that generates random polygons.
    """
    polygons = []
    values = []
    
    for i in range(num_polygons):
        poly = generate_random_polygon(bbox, rng=rng)
        polygons.append(poly)
        # Random value 0-1 (continuous) as in paper
        values.append(rng.uniform(0, 1))
    
    gdf = gpd.GeoDataFrame({
        'value': values,
        'geometry': polygons
    }, crs="EPSG:4326")
    
    return gdf


def generate_nlm_raster(size: Tuple[int, int], 
                        rng: np.random.Generator,
                        continuous: bool = True) -> np.ndarray:
    """
    Generate a Neutral Landscape Model (NLM) raster.
    
    The paper uses mid-point displacement NLM (Etherington et al., 2015).
    """
    if HAS_NLMPY:
        # Use nlmpy for proper NLM generation
        try:
            raster = nlmpy.mpd(size[0], size[1], h=0.5)
            if not continuous:
                # Convert to discrete classes (1-10)
                raster = np.digitize(raster, np.linspace(0, 1, 11)[1:-1]) + 1
            return raster
        except:
            pass
    
    # Fallback: simple random raster
    if continuous:
        return rng.uniform(0, 1, size)
    else:
        return rng.integers(1, 11, size)


def generate_vector_data(config: Dict, output_dir: Path) -> List[Path]:
    """Generate all vector benchmark data."""
    print("\n" + "=" * 60)
    print("GENERATING VECTOR BENCHMARK DATA")
    print("=" * 60)
    
    rng = np.random.default_rng(config["random_seed"])
    output_dir = output_dir / "vector"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    max_layers = max(config["vector"]["num_layers_list"])
    files = []
    
    print(f"Generating {max_layers} vector layers...")
    for i in tqdm(range(max_layers)):
        gdf = generate_voronoi_layer(
            config["vector"]["num_polygons_per_layer"],
            config["vector"]["bbox"],
            rng
        )
        filepath = output_dir / f"layer_{i:04d}.parquet"
        gdf.to_parquet(filepath)
        files.append(filepath)
    
    print(f"Generated {len(files)} vector layers in {output_dir}")
    return files


def generate_raster_data(config: Dict, output_dir: Path) -> Dict[str, np.ndarray]:
    """Generate all raster benchmark data."""
    print("\n" + "=" * 60)
    print("GENERATING RASTER BENCHMARK DATA")
    print("=" * 60)
    
    rng = np.random.default_rng(config["random_seed"])
    output_dir = output_dir / "raster"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    max_layers = max(config["raster"]["num_layers_list"])
    size = config["raster"]["raster_size"]
    
    data = {
        "continuous": [],
        "discrete": []
    }
    
    print(f"Generating {max_layers} raster layers (continuous and discrete)...")
    for i in tqdm(range(max_layers)):
        # Continuous
        continuous = generate_nlm_raster(size, rng, continuous=True)
        data["continuous"].append(continuous)
        
        # Discrete
        discrete = generate_nlm_raster(size, rng, continuous=False)
        data["discrete"].append(discrete)
    
    # Save as numpy arrays
    np.save(output_dir / "continuous_layers.npy", np.array(data["continuous"]))
    np.save(output_dir / "discrete_layers.npy", np.array(data["discrete"]))
    
    print(f"Generated {max_layers} raster layers in {output_dir}")
    return data


# =============================================================================
# Benchmark: Vector (Figure 6)
# =============================================================================

def benchmark_vector_union(layers: List[gpd.GeoDataFrame]) -> Tuple[float, bool]:
    """
    Benchmark vector spatial union method.
    
    This simulates the traditional GIS approach of overlaying multiple
    vector layers using spatial union operations.
    
    Note: We only keep geometry to avoid column naming conflicts.
    The benchmark measures geometry overlay time, which is the bottleneck.
    """
    start_time = time.perf_counter()
    success = True
    
    try:
        # Keep only geometry for clean overlay (avoids column conflicts)
        result = layers[0][['geometry']].copy()
        
        for i, layer in enumerate(layers[1:], start=1):
            layer_geom = layer[['geometry']].copy()
            # This is expensive and can cause memory issues
            result = gpd.overlay(result, layer_geom, how='union')
        
    except MemoryError:
        success = False
    except Exception as e:
        print(f"Vector union failed: {e}")
        success = False
    
    elapsed = time.perf_counter() - start_time
    return elapsed, success


def benchmark_dggs_vector(layers: List[gpd.GeoDataFrame], 
                          h3_resolution: int) -> Tuple[float, float, float]:
    """
    Benchmark DGGS-based method for vector data.
    
    Steps:
    1. Index: Convert each polygon centroid to H3 cell
    2. Classify: Aggregate values by H3 cell ID
    
    Uses efficient concat + groupby instead of iterative merge (O(n) vs O(n²)).
    
    Returns: (indexing_time, classifying_time, total_time)
    """
    # Step 1: Indexing - convert centroids to H3
    index_start = time.perf_counter()
    
    all_records = []
    for layer_idx, gdf in enumerate(layers):
        # Vectorized centroid calculation
        centroids = gdf.geometry.centroid
        
        for idx, (centroid, value) in enumerate(zip(centroids, gdf['value'])):
            if centroid is not None:
                cell = h3.geo_to_h3(centroid.y, centroid.x, h3_resolution)
                all_records.append({
                    'h3_cell': cell,
                    'layer': layer_idx,
                    'value': value
                })
    
    index_time = time.perf_counter() - index_start
    
    # Step 2: Classifying - aggregate by H3 cell
    classify_start = time.perf_counter()
    
    # Convert to DataFrame and pivot (much faster than iterative merge)
    df = pd.DataFrame(all_records)
    
    # Count values > 0.5 per H3 cell
    df['above_threshold'] = (df['value'] > 0.5).astype(int)
    result = df.groupby('h3_cell')['above_threshold'].sum().reset_index()
    result.columns = ['h3_cell', 'class']
    
    classify_time = time.perf_counter() - classify_start
    total_time = index_time + classify_time
    
    return index_time, classify_time, total_time


def run_vector_benchmark(config: Dict, data_dir: Path) -> pd.DataFrame:
    """Run the complete vector benchmark (Figure 6)."""
    print("\n" + "=" * 60)
    print("RUNNING VECTOR BENCHMARK (Figure 6)")
    print("=" * 60)
    
    results = []
    vector_dir = data_dir / "vector"
    
    for num_layers in config["vector"]["num_layers_list"]:
        print(f"\nBenchmarking with {num_layers} layers...")
        
        # Load layers
        layers = []
        for i in range(num_layers):
            filepath = vector_dir / f"layer_{i:04d}.parquet"
            if filepath.exists():
                layers.append(gpd.read_parquet(filepath))
            else:
                print(f"Warning: {filepath} not found")
                break
        
        if len(layers) < num_layers:
            print(f"Only {len(layers)} layers available, skipping...")
            continue
        
        # Benchmark DGGS method (always runs - scales well)
        dggs_index, dggs_classify, dggs_total = benchmark_dggs_vector(
            layers, config["h3_resolution"]
        )
        
        # Benchmark traditional vector method
        # Only run if num_layers <= MAX_TRADITIONAL_LAYERS (to avoid OOM in CI)
        max_traditional = int(os.environ.get("MAX_TRADITIONAL_LAYERS", 
                                              config["vector"]["max_layers_before_failure"]))
        
        if num_layers <= max_traditional:
            print(f"  Running traditional vector method...")
            vector_time, vector_success = benchmark_vector_union(layers)
        else:
            vector_time = np.nan
            vector_success = None  # None = skipped (vs False = failed)
            print(f"  Skipping traditional method (layers {num_layers} > max {max_traditional})")
        
        result = {
            "num_layers": num_layers,
            "dggs_index_time": dggs_index,
            "dggs_classify_time": dggs_classify,
            "dggs_total_time": dggs_total,
            "vector_time": vector_time,
            "vector_success": vector_success,
        }
        results.append(result)
        
        print(f"  DGGS: {dggs_total:.2f}s (index: {dggs_index:.2f}s, classify: {dggs_classify:.2f}s)")
        if vector_success is None:
            pass  # Already printed skip message
        elif vector_success:
            print(f"  Vector: {vector_time:.2f}s")
        else:
            print(f"  Vector: FAILED (memory/timeout)")
    
    return pd.DataFrame(results)


# =============================================================================
# Benchmark: Raster (Figure 7)
# =============================================================================

def benchmark_raster_warp_classify(layers: np.ndarray) -> Tuple[float, float, float]:
    """
    Benchmark traditional raster method.
    
    Steps:
    1. Warp: Align all rasters (simulated - they're already aligned)
    2. Classify: Stack and compute classification
    """
    # Step 1: Warping (simulated - in reality would use rasterio.warp)
    warp_start = time.perf_counter()
    # Simulate alignment check
    aligned = np.stack(layers, axis=0)
    warp_time = time.perf_counter() - warp_start
    
    # Step 2: Classification
    classify_start = time.perf_counter()
    # Boolean threshold and sum (as in paper)
    binary = (aligned > 0.5).astype(np.int8)
    result = np.sum(binary, axis=0)
    classify_time = time.perf_counter() - classify_start
    
    total_time = warp_time + classify_time
    return warp_time, classify_time, total_time



def benchmark_dggs_raster(layers: np.ndarray, h3_resolution: int) -> Tuple[float, float, float]:
    """
    Optimized DGGS-based method for raster data using xdggs.
    """
    if isinstance(layers, list):
        layers = np.stack(layers, axis=0)
    
    num_layers, nrows, ncols = layers.shape
    
    # ===== STEP 1: INDEXING with xdggs =====
    index_start = time.perf_counter()
    
    # Create coordinate meshgrid
    lats = np.linspace(-5, 5, nrows)
    lons = np.linspace(-5, 5, ncols)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Flatten for xdggs
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    
    # VECTORIZED H3 conversion using xdggs!
    # Note: xdggs uses 'level' instead of 'resolution'
    h3_info = xdggs.H3Info(level=h3_resolution)
    cell_ids_arrow = h3_info.geographic2cell_ids(lon_flat, lat_flat)
    cell_ids = np.asarray(cell_ids_arrow)
    
    # Get unique cells for efficient grouping
    unique_cells, inverse_indices = np.unique(cell_ids, return_inverse=True)
    num_cells = len(unique_cells)
    
    index_time = time.perf_counter() - index_start
    
    # ===== STEP 2: CLASSIFICATION (vectorized) =====
    classify_start = time.perf_counter()
    
    # Threshold all layers at once
    binary = (layers > 0.5).astype(np.float32)
    flat_binary = binary.reshape(num_layers, -1)
    
    # Pixel counts per cell
    cell_pixel_counts = np.bincount(inverse_indices, minlength=num_cells)
    
    # Vectorized aggregation with bincount
    cell_sums = np.zeros((num_layers, num_cells), dtype=np.float32)
    for layer_idx in range(num_layers):
        cell_sums[layer_idx] = np.bincount(
            inverse_indices,
            weights=flat_binary[layer_idx],
            minlength=num_cells
        )
    
    # Compute means and count
    with np.errstate(divide='ignore', invalid='ignore'):
        cell_means = cell_sums / cell_pixel_counts
        cell_means = np.nan_to_num(cell_means, nan=0.0)
    
    result_counts = (cell_means > 0.5).sum(axis=0)
    
    classify_time = time.perf_counter() - classify_start
    total_time = index_time + classify_time
    
    return index_time, classify_time, total_time

def run_raster_benchmark(config: Dict, data_dir: Path) -> pd.DataFrame:
    """Run the complete raster benchmark (Figure 7)."""
    print("\n" + "=" * 60)
    print("RUNNING RASTER BENCHMARK (Figure 7)")
    print("=" * 60)
    
    results = []
    raster_dir = data_dir / "raster"
    
    # Load raster data
    continuous_path = raster_dir / "continuous_layers.npy"
    if not continuous_path.exists():
        print("Raster data not found. Generate data first with --generate-data")
        return pd.DataFrame()
    
    all_layers = np.load(continuous_path)
    
    for num_layers in config["raster"]["num_layers_list"]:
        if num_layers > len(all_layers):
            print(f"Only {len(all_layers)} layers available, skipping {num_layers}...")
            continue
        
        print(f"\nBenchmarking with {num_layers} layers...")
        layers = all_layers[:num_layers]
        
        # Benchmark DGGS method
        dggs_index, dggs_classify, dggs_total = benchmark_dggs_raster(
            layers, config["h3_resolution"]
        )
        
        # Benchmark raster method
        raster_warp, raster_classify, raster_total = benchmark_raster_warp_classify(layers)
        
        result = {
            "num_layers": num_layers,
            "dggs_index_time": dggs_index,
            "dggs_classify_time": dggs_classify,
            "dggs_total_time": dggs_total,
            "raster_warp_time": raster_warp,
            "raster_classify_time": raster_classify,
            "raster_total_time": raster_total,
        }
        results.append(result)
        
        print(f"  DGGS: {dggs_total:.2f}s (index: {dggs_index:.2f}s, classify: {dggs_classify:.2f}s)")
        print(f"  Raster: {raster_total:.2f}s (warp: {raster_warp:.2f}s, classify: {raster_classify:.2f}s)")
    
    return pd.DataFrame(results)


# =============================================================================
# Results Analysis
# =============================================================================

def compare_with_paper(vector_results: pd.DataFrame, 
                       raster_results: pd.DataFrame,
                       output_dir: Path) -> Dict:
    """
    Compare replication results with expected patterns from paper.
    
    Expected patterns:
    - Figure 6: DGGS >> Vector performance, Vector fails at ~500 layers
    - Figure 7: DGGS ≈ Raster performance
    """
    comparison = {
        "vector_benchmark": {
            "claim": "DGGS provides significant performance benefits over vector methods",
            "expected": "DGGS should be orders of magnitude faster; vector should fail at ~500 layers",
            "observed": None,
            "replicated": None,
        },
        "raster_benchmark": {
            "claim": "DGGS and raster methods show roughly equivalent performance",
            "expected": "Similar timing curves on log-log plot",
            "observed": None,
            "replicated": None,
        }
    }
    
    # Analyze vector results
    if not vector_results.empty:
        # Check if DGGS is faster
        valid = vector_results[vector_results['vector_success'] == True]
        if not valid.empty:
            speedup = valid['vector_time'].mean() / valid['dggs_total_time'].mean()
            comparison["vector_benchmark"]["observed"] = f"DGGS speedup: {speedup:.1f}x"
            comparison["vector_benchmark"]["replicated"] = speedup > 2  # At least 2x faster
        
        # Check vector failure point
        failures = vector_results[vector_results['vector_success'] == False]
        if not failures.empty:
            first_failure = failures['num_layers'].min()
            comparison["vector_benchmark"]["observed"] += f"; Vector failed at {first_failure} layers"
    
    # Analyze raster results
    if not raster_results.empty:
        # Check if times are similar (within 2x)
        ratio = raster_results['dggs_total_time'].mean() / raster_results['raster_total_time'].mean()
        comparison["raster_benchmark"]["observed"] = f"DGGS/Raster ratio: {ratio:.2f}"
        comparison["raster_benchmark"]["replicated"] = 0.5 < ratio < 2.0  # Within 2x
    
    # Save comparison
    with open(output_dir / "comparison_with_paper.json", 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    return comparison


def plot_results(vector_results: pd.DataFrame, 
                 raster_results: pd.DataFrame,
                 output_dir: Path):
    """Generate plots similar to Figures 6 and 7 in the paper."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Figure 6: Vector benchmark
    ax1 = axes[0]
    if not vector_results.empty:
        ax1.loglog(vector_results['num_layers'], vector_results['dggs_total_time'], 
                   'o-', label='DGGS Total', color='blue')
        ax1.loglog(vector_results['num_layers'], vector_results['dggs_index_time'], 
                   's--', label='DGGS Index', color='lightblue')
        
        valid_vector = vector_results[vector_results['vector_success'] == True]
        if not valid_vector.empty:
            ax1.loglog(valid_vector['num_layers'], valid_vector['vector_time'], 
                       'o-', label='Vector', color='orange')
        
        # Mark failure points
        failed = vector_results[vector_results['vector_success'] == False]
        if not failed.empty:
            ax1.axvline(x=failed['num_layers'].min(), color='red', linestyle=':', 
                        label='Vector failure')
    
    ax1.set_xlabel('Number of input layers')
    ax1.set_ylabel('Compute time (s)')
    ax1.set_title('Vector Input Benchmark (cf. Figure 6)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Figure 7: Raster benchmark
    ax2 = axes[1]
    if not raster_results.empty:
        ax2.loglog(raster_results['num_layers'], raster_results['dggs_total_time'], 
                   'o-', label='DGGS Total', color='blue')
        ax2.loglog(raster_results['num_layers'], raster_results['dggs_index_time'], 
                   's--', label='DGGS Index', color='lightblue')
        ax2.loglog(raster_results['num_layers'], raster_results['raster_total_time'], 
                   'o-', label='Raster Total', color='orange')
        ax2.loglog(raster_results['num_layers'], raster_results['raster_warp_time'], 
                   's--', label='Raster Warp', color='lightsalmon')
    
    ax2.set_xlabel('Number of input layers')
    ax2.set_ylabel('Compute time (s)')
    ax2.set_title('Raster Input Benchmark (cf. Figure 7)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_results.png", dpi=150)
    plt.savefig(output_dir / "benchmark_results.pdf")
    print(f"Plots saved to {output_dir}")


# =============================================================================
# Original Benchmark Repository Functions
# =============================================================================

# Path to the original benchmark code (cloned by Dockerfile)
ORIGINAL_BENCHMARKS_DIR = Path("/app/original_benchmarks")


def explore_original_repo() -> bool:
    """
    Explore the original benchmark repository from the paper authors.
    
    The original code is cloned into /app/original_benchmarks by the Dockerfile.
    This function shows what's available so we can understand how to run it.
    """
    print("\n" + "=" * 60)
    print("ORIGINAL BENCHMARK REPOSITORY")
    print("=" * 60)
    print(f"Paper: Law & Ardo (2024)")
    print(f"DOI: {_env_config.get('ORIGINAL_PAPER_DOI', 'N/A')}")
    print(f"Repo: {_env_config.get('DGGS_BENCHMARKS_REPO', 'N/A')}")
    print(f"Version: {_env_config.get('DGGS_BENCHMARKS_VERSION', 'N/A')}")
    
    if not ORIGINAL_BENCHMARKS_DIR.exists():
        print(f"\nERROR: {ORIGINAL_BENCHMARKS_DIR} not found!")
        print("The original benchmark repo should be cloned by the Dockerfile.")
        return False
    
    print(f"\n{'=' * 60}")
    print(f"CONTENTS OF {ORIGINAL_BENCHMARKS_DIR}")
    print("=" * 60)
    
    for item in sorted(ORIGINAL_BENCHMARKS_DIR.iterdir()):
        if item.name.startswith('.'):
            continue
        if item.is_dir():
            print(f"  📁 {item.name}/")
            # Show first-level contents
            try:
                subitems = sorted([f for f in item.iterdir() if not f.name.startswith('.')])
                for subitem in subitems[:5]:
                    print(f"      {subitem.name}")
                if len(subitems) > 5:
                    print(f"      ... and {len(subitems) - 5} more")
            except PermissionError:
                print("      (permission denied)")
        else:
            print(f"  📄 {item.name}")
    
    # Find Python files
    print(f"\n{'=' * 60}")
    print("PYTHON FILES")
    print("=" * 60)
    
    py_files = list(ORIGINAL_BENCHMARKS_DIR.rglob("*.py"))
    if py_files:
        for f in sorted(py_files)[:20]:
            rel_path = f.relative_to(ORIGINAL_BENCHMARKS_DIR)
            print(f"  {rel_path}")
        if len(py_files) > 20:
            print(f"  ... and {len(py_files) - 20} more")
    else:
        print("  No Python files found")
    
    # Show README if exists
    for readme_name in ["README.md", "readme.md", "README.rst", "README"]:
        readme_path = ORIGINAL_BENCHMARKS_DIR / readme_name
        if readme_path.exists():
            print(f"\n{'=' * 60}")
            print(f"README ({readme_name})")
            print("=" * 60)
            with open(readme_path) as f:
                content = f.read()
                if len(content) > 3000:
                    print(content[:3000])
                    print(f"\n... (truncated, {len(content)} total characters)")
                else:
                    print(content)
            break
    
    return True


def run_original_benchmarks(script_name: str = None) -> int:
    """
    Attempt to run the original benchmark code from the paper authors.
    
    Args:
        script_name: Specific script to run. If None, tries to auto-detect.
    
    Returns:
        Exit code (0 for success)
    """
    if not ORIGINAL_BENCHMARKS_DIR.exists():
        print(f"ERROR: {ORIGINAL_BENCHMARKS_DIR} not found!")
        return 1
    
    os.chdir(ORIGINAL_BENCHMARKS_DIR)
    sys.path.insert(0, str(ORIGINAL_BENCHMARKS_DIR))
    
    if script_name:
        script_path = ORIGINAL_BENCHMARKS_DIR / script_name
        if script_path.exists():
            print(f"\nRunning: {script_path}")
            result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
            return result.returncode
        else:
            print(f"ERROR: Script not found: {script_path}")
            return 1
    
    # Auto-detect main script
    possible_scripts = [
        "benchmark.py",
        "run_benchmark.py",
        "run_benchmarks.py",
        "main.py",
        "run.py",
    ]
    
    for script in possible_scripts:
        script_path = ORIGINAL_BENCHMARKS_DIR / script
        if script_path.exists():
            print(f"\nAuto-detected main script: {script}")
            print(f"Running: {script_path}")
            result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
            return result.returncode
    
    print("\nCould not auto-detect main benchmark script.")
    print("Available Python files:")
    for f in sorted(ORIGINAL_BENCHMARKS_DIR.rglob("*.py"))[:10]:
        print(f"  {f.relative_to(ORIGINAL_BENCHMARKS_DIR)}")
    print("\nUse --run-original <script_name> to specify which script to run.")
    return 1


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DGGS Benchmark Replication Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script replicates the benchmarks from Law & Ardo (2024):
"Using a discrete global grid system for a scalable, interoperable, 
and reproducible system of land-use mapping"
https://doi.org/10.1080/20964471.2024.2429847

Examples:
  python run_replication.py --all              # Run complete replication
  python run_replication.py --generate-data    # Generate synthetic data only
  python run_replication.py --vector           # Run vector benchmark only
  python run_replication.py --raster           # Run raster benchmark only
  python run_replication.py --explore          # Explore original benchmark repo
  python run_replication.py --run-original     # Run original benchmark scripts
        """
    )
    
    # Original benchmark repo options
    parser.add_argument('--explore', action='store_true', 
                        help='Explore the original benchmark repository')
    parser.add_argument('--run-original', nargs='?', const='auto', metavar='SCRIPT',
                        help='Run original benchmark scripts (auto-detect or specify script name)')
    
    # Replication options
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--vector', action='store_true', help='Run vector benchmark (Figure 6)')
    parser.add_argument('--raster', action='store_true', help='Run raster benchmark (Figure 7)')
    parser.add_argument('--compare', action='store_true', help='Compare results with paper')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output', '-o', default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config.env/RANDOM_SEED)')
    parser.add_argument('--vector-layers', type=str, default=None,
                        help='Comma-separated list of vector layer counts (overrides config.env/VECTOR_LAYERS)')
    parser.add_argument('--raster-layers', type=str, default=None,
                        help='Comma-separated list of raster layer counts (overrides config.env/RASTER_LAYERS)')
    
    args = parser.parse_args()
    
    # Handle exploration and original benchmark options first (exit early)
    if args.explore:
        explore_original_repo()
        return 0
    
    if args.run_original is not None:
        script = None if args.run_original == 'auto' else args.run_original
        return run_original_benchmarks(script)
    
    # CLI args override config (config already includes env var overrides)
    if args.seed is not None:
        CONFIG["random_seed"] = args.seed
    CONFIG["results_dir"] = args.output
    
    # Override layer counts if specified via CLI
    if args.vector_layers:
        CONFIG["vector"]["num_layers_list"] = [int(x.strip()) for x in args.vector_layers.split(',')]
    
    if args.raster_layers:
        CONFIG["raster"]["num_layers_list"] = [int(x.strip()) for x in args.raster_layers.split(',')]
    
    # Print effective configuration
    max_traditional = int(os.environ.get("MAX_TRADITIONAL_LAYERS", 
                                          CONFIG["vector"]["max_layers_before_failure"]))
    print(f"Code version: {CODE_VERSION}")
    print(f"Configuration:")
    print(f"  Random seed: {CONFIG['random_seed']}")
    print(f"  H3 resolution: {CONFIG['h3_resolution']}")
    print(f"  Vector layers: {CONFIG['vector']['num_layers_list']}")
    print(f"  Polygons per layer: {CONFIG['vector']['num_polygons_per_layer']}")
    print(f"  Raster layers: {CONFIG['raster']['num_layers_list']}")
    print(f"  Max layers for traditional methods: {max_traditional}")
    print(f"  Output directory: {CONFIG['results_dir']}")
    
    # Setup directories
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(CONFIG["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect system info
    sys_info = get_system_info()
    with open(results_dir / "system_info.json", 'w') as f:
        json.dump(sys_info, f, indent=2)
    print(f"System info saved to {results_dir / 'system_info.json'}")
    
    # Run requested operations
    if args.all or args.generate_data:
        generate_vector_data(CONFIG, data_dir)
        generate_raster_data(CONFIG, data_dir)
    
    vector_results = pd.DataFrame()
    raster_results = pd.DataFrame()
    
    if args.all or args.vector:
        vector_results = run_vector_benchmark(CONFIG, data_dir)
        vector_results.to_csv(results_dir / "vector_benchmark.csv", index=False)
        print(f"\nVector results saved to {results_dir / 'vector_benchmark.csv'}")
    
    if args.all or args.raster:
        raster_results = run_raster_benchmark(CONFIG, data_dir)
        raster_results.to_csv(results_dir / "raster_benchmark.csv", index=False)
        print(f"\nRaster results saved to {results_dir / 'raster_benchmark.csv'}")
    
    if args.all or args.compare:
        # Load results if not already in memory
        if vector_results.empty:
            csv_path = results_dir / "vector_benchmark.csv"
            if csv_path.exists():
                vector_results = pd.read_csv(csv_path)
        if raster_results.empty:
            csv_path = results_dir / "raster_benchmark.csv"
            if csv_path.exists():
                raster_results = pd.read_csv(csv_path)
        
        comparison = compare_with_paper(vector_results, raster_results, results_dir)
        
        print("\n" + "=" * 60)
        print("COMPARISON WITH PAPER")
        print("=" * 60)
        for benchmark, info in comparison.items():
            print(f"\n{benchmark}:")
            print(f"  Claim: {info['claim']}")
            print(f"  Expected: {info['expected']}")
            print(f"  Observed: {info['observed']}")
            print(f"  Replicated: {'✅ YES' if info['replicated'] else '❌ NO'}")
    
    if args.all or args.plot:
        plot_results(vector_results, raster_results, results_dir)
    
    print("\n" + "=" * 60)
    print("REPLICATION COMPLETE")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    print(f"Files generated:")
    for f in results_dir.iterdir():
        print(f"  - {f.name}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
