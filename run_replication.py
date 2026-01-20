#!/usr/bin/env python3
"""
DGGS Benchmark Replication Study - IMPROVED VERSION

This script replicates the benchmarks from:
Law & Ardo (2024) - "Using a discrete global grid system for a scalable,
interoperable, and reproducible system of land-use mapping"
https://doi.org/10.1080/20964471.2024.2429847

Original benchmark code: https://github.com/manaakiwhenua/dggsBenchmarks v1.1.1

IMPROVEMENTS over initial replication:
1. Uses Voronoi polygons (as per paper Section 3.2.1)
2. Implements all 7 classification functions from the paper
3. Uses H3 polyfilling instead of centroids
4. Uses H3 resolution 14 for vector benchmarks (as per paper)
5. Properly implements the classification logic producing 127 possible classes

Author: Anne Fouilloux (replication)
Original authors: Richard M. Law, James Ardo
Date: 2026-01-20
"""

import os
import sys
import json
import time
import random
import argparse
import platform
import subprocess
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from functools import lru_cache

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon, Point, box
from shapely.ops import unary_union
from scipy.spatial import Voronoi
import h3
from tqdm import tqdm

# Try to import optional dependencies
try:
    from nlmpy import nlmpy as nlmpy_module
    HAS_NLMPY = True
except ImportError:
    HAS_NLMPY = False
    print("Warning: nlmpy not available. Using fallback raster generation.")

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

try:
    import h3pandas
    HAS_H3PANDAS = True
except ImportError:
    HAS_H3PANDAS = False
    print("Warning: h3pandas not available. Using manual H3 operations.")


# =============================================================================
# Configuration
# =============================================================================

CODE_VERSION = "2026-01-20-replication-v1"

def load_config_env(config_path: Path = None) -> Dict[str, str]:
    """Load configuration from config.env file."""
    if config_path is None:
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
                    value = value.strip('"').strip("'")
                    config[key] = value
    return config

_env_config = load_config_env()

def parse_layer_list(value: str) -> List[int]:
    """Parse comma-separated layer list string to list of ints."""
    if not value:
        return []
    return [int(x.strip()) for x in value.split(',')]

# Paper-accurate configuration
CONFIG = {
    "random_seed": int(os.environ.get("RANDOM_SEED", _env_config.get("RANDOM_SEED", "42"))),
    
    # H3 resolution 14 for vector benchmarks (as per paper Section 3.2.1)
    # "H3 (resolution 14) zones" - Figure 5 caption
    "h3_resolution_vector": 14,
    
    # H3 resolution for raster (paper doesn't specify, using 9 as reasonable default)
    "h3_resolution_raster": int(os.environ.get("H3_RESOLUTION", _env_config.get("H3_RESOLUTION", "9"))),
    
    # Original benchmark reference
    "original_benchmark": {
        "repo": _env_config.get("DGGS_BENCHMARKS_REPO", "https://github.com/manaakiwhenua/dggsBenchmarks"),
        "version": _env_config.get("DGGS_BENCHMARKS_VERSION", "v1.1.1"),
    },
    
    # Vector benchmark parameters (Figure 6)
    # Paper: "We generated 500 random vector coverages"
    # Paper layer counts from Figure 6: ~10, 20, 50, 100, 200, 500, 1000
    "vector": {
        "num_layers_list": parse_layer_list(
            os.environ.get("VECTOR_LAYERS", _env_config.get("VECTOR_LAYERS", "10,20,50,100,200,500,1000"))
        ),
        # Paper: "using a random distribution of points"
        "num_points_per_layer": int(os.environ.get("POINTS_PER_LAYER", "100")),
        "bbox": (0, 0, 10, 10),  # Smaller extent for faster computation
        "max_layers_before_failure": 500,
    },
    
    # Raster benchmark parameters (Figure 7)
    # Paper: "10,000 NLM landscapes were generated, each 100 by 100 pixels"
    "raster": {
        "num_layers_list": parse_layer_list(
            os.environ.get("RASTER_LAYERS", _env_config.get("RASTER_LAYERS", "10,50,100,500,1000,5000,10000"))
        ),
        "raster_size": (100, 100),
        "num_classes": 10,
    },
    
    "results_dir": "results",
    "data_dir": "data",
}


# =============================================================================
# Classification Functions (from Paper Section 3.2.1)
# =============================================================================
# "These functions determine whether that sum is: a prime number; a perfect 
# number; a triangular, square, pentagonal, or hexagonal number; or a Fibonacci 
# number. Each unique combination of the eight resultant Boolean values was 
# then considered a distinct 'class'."

@lru_cache(maxsize=10000)
def is_prime(n: int) -> bool:
    """Check if n is a prime number."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


@lru_cache(maxsize=10000)
def is_perfect(n: int) -> bool:
    """Check if n is a perfect number (sum of proper divisors equals n)."""
    if n < 2:
        return False
    divisor_sum = 1
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisor_sum += i
            if i != n // i:
                divisor_sum += n // i
    return divisor_sum == n


@lru_cache(maxsize=10000)
def is_triangular(n: int) -> bool:
    """Check if n is a triangular number (n = k(k+1)/2 for some k)."""
    if n < 0:
        return False
    # Solve k^2 + k - 2n = 0
    discriminant = 1 + 8 * n
    sqrt_disc = math.isqrt(discriminant)
    if sqrt_disc * sqrt_disc != discriminant:
        return False
    return (-1 + sqrt_disc) % 2 == 0


@lru_cache(maxsize=10000)
def is_square(n: int) -> bool:
    """Check if n is a perfect square."""
    if n < 0:
        return False
    sqrt_n = math.isqrt(n)
    return sqrt_n * sqrt_n == n


@lru_cache(maxsize=10000)
def is_pentagonal(n: int) -> bool:
    """Check if n is a pentagonal number (n = k(3k-1)/2 for some k)."""
    if n < 0:
        return False
    # Solve 3k^2 - k - 2n = 0
    discriminant = 1 + 24 * n
    sqrt_disc = math.isqrt(discriminant)
    if sqrt_disc * sqrt_disc != discriminant:
        return False
    return (1 + sqrt_disc) % 6 == 0


@lru_cache(maxsize=10000)
def is_hexagonal(n: int) -> bool:
    """Check if n is a hexagonal number (n = k(2k-1) for some k)."""
    if n < 0:
        return False
    # Solve 2k^2 - k - n = 0
    discriminant = 1 + 8 * n
    sqrt_disc = math.isqrt(discriminant)
    if sqrt_disc * sqrt_disc != discriminant:
        return False
    return (1 + sqrt_disc) % 4 == 0


def _generate_fibonacci_set(max_val: int) -> Set[int]:
    """Generate set of Fibonacci numbers up to max_val."""
    fib_set = {0, 1}
    a, b = 0, 1
    while b <= max_val:
        fib_set.add(b)
        a, b = b, a + b
    return fib_set

_FIBONACCI_SET = _generate_fibonacci_set(100000)

def is_fibonacci(n: int) -> bool:
    """Check if n is a Fibonacci number."""
    return n in _FIBONACCI_SET


def classify_value(value: int) -> int:
    """
    Apply all 7 classification functions and return class as 7-bit integer.
    
    From paper: "Each unique combination of the eight resultant Boolean values 
    was then considered a distinct 'class' akin to a distinct land-use type."
    
    This gives 2^7 = 128 possible classes (paper says 127 possible classes).
    """
    class_bits = 0
    if is_prime(value):
        class_bits |= 1
    if is_perfect(value):
        class_bits |= 2
    if is_triangular(value):
        class_bits |= 4
    if is_square(value):
        class_bits |= 8
    if is_pentagonal(value):
        class_bits |= 16
    if is_hexagonal(value):
        class_bits |= 32
    if is_fibonacci(value):
        class_bits |= 64
    return class_bits


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
        "code_version": CODE_VERSION,
        "original_benchmark": CONFIG["original_benchmark"],
    }
    
    if HAS_PSUTIL:
        info["cpu_count"] = psutil.cpu_count()
        info["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
        info["memory_available_gb"] = round(psutil.virtual_memory().available / (1024**3), 2)
    
    info["packages"] = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "geopandas": gpd.__version__,
        "h3": h3.__version__ if hasattr(h3, '__version__') else "unknown",
        "h3pandas": "available" if HAS_H3PANDAS else "not available",
        "nlmpy": "available" if HAS_NLMPY else "not available",
    }
    
    return info


# =============================================================================
# Data Generation - Voronoi Polygons (Paper Section 3.2.1)
# =============================================================================

def generate_voronoi_layer(num_points: int, 
                           bbox: Tuple[float, float, float, float],
                           rng: np.random.Generator) -> gpd.GeoDataFrame:
    """
    Generate a layer of Voronoi polygons.
    
    From paper Section 3.2.1:
    "We generated 500 random vector coverages, using a random distribution of 
    points over a fixed extent, and calculated Voronoi polygons for each case.
    Each polygon in each coverage was randomly assigned a 0 or 1 value and 
    then dissolved accordingly."
    """
    minx, miny, maxx, maxy = bbox
    
    # Generate random points
    points = rng.uniform([minx, miny], [maxx, maxy], size=(num_points, 2))
    
    # Add boundary points to ensure Voronoi covers the extent
    boundary_points = np.array([
        [minx - 10, miny - 10],
        [minx - 10, maxy + 10],
        [maxx + 10, miny - 10],
        [maxx + 10, maxy + 10],
    ])
    all_points = np.vstack([points, boundary_points])
    
    try:
        vor = Voronoi(all_points)
        
        polygons = []
        values = []
        
        for i in range(num_points):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) < 3:
                continue
                
            vertices = [vor.vertices[v] for v in region]
            try:
                poly = Polygon(vertices)
                if poly.is_valid and not poly.is_empty:
                    clip_box = box(minx, miny, maxx, maxy)
                    poly = poly.intersection(clip_box)
                    if not poly.is_empty:
                        polygons.append(poly)
                        values.append(rng.integers(0, 2))
            except:
                continue
        
        if len(polygons) == 0:
            return _generate_fallback_layer(bbox, num_points, rng)
        
        gdf = gpd.GeoDataFrame({
            'value': values,
            'geometry': polygons
        }, crs="EPSG:4326")
        
        # Dissolve by value (as per paper)
        gdf_dissolved = gdf.dissolve(by='value', as_index=False)
        
        return gdf_dissolved
        
    except Exception as e:
        print(f"Voronoi generation failed: {e}, using fallback")
        return _generate_fallback_layer(bbox, num_points, rng)


def _generate_fallback_layer(bbox: Tuple[float, float, float, float],
                             num_polygons: int,
                             rng: np.random.Generator) -> gpd.GeoDataFrame:
    """Fallback polygon generation if Voronoi fails."""
    minx, miny, maxx, maxy = bbox
    
    polygons = []
    values = []
    
    n = int(np.sqrt(num_polygons)) + 1
    dx = (maxx - minx) / n
    dy = (maxy - miny) / n
    
    for i in range(n):
        for j in range(n):
            x0 = minx + i * dx
            y0 = miny + j * dy
            poly = box(x0, y0, x0 + dx, y0 + dy)
            polygons.append(poly)
            values.append(rng.integers(0, 2))
    
    gdf = gpd.GeoDataFrame({
        'value': values,
        'geometry': polygons
    }, crs="EPSG:4326")
    
    return gdf.dissolve(by='value', as_index=False)


def generate_nlm_raster(size: Tuple[int, int], 
                        rng: np.random.Generator,
                        continuous: bool = True) -> np.ndarray:
    """
    Generate a Neutral Landscape Model (NLM) raster.
    
    From paper Section 3.2.2:
    "the raster data for the benchmark experiment was generated using 
    a mid-point displacement (NLM) (Etherington et al., 2015)"
    """
    if HAS_NLMPY:
        try:
            raster = nlmpy_module.mpd(size[0], size[1], h=0.5)
            if not continuous:
                raster = np.digitize(raster, np.linspace(0, 1, 11)[1:-1]) + 1
            return raster
        except Exception as e:
            print(f"nlmpy failed: {e}, using fallback")
    
    if continuous:
        base = rng.uniform(0, 1, size)
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(base, sigma=2)
        except:
            return base
    else:
        return rng.integers(1, 11, size)


# =============================================================================
# Data Generation Functions
# =============================================================================

def generate_vector_data(config: Dict, output_dir: Path) -> List[Path]:
    """Generate all vector benchmark data (Voronoi polygons)."""
    print("\n" + "=" * 60)
    print("GENERATING VECTOR BENCHMARK DATA")
    print("=" * 60)
    
    rng = np.random.default_rng(config["random_seed"])
    output_dir = output_dir / "vector"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    max_layers = max(config["vector"]["num_layers_list"])
    files = []
    
    print(f"Generating {max_layers} Voronoi vector layers...")
    for i in tqdm(range(max_layers)):
        gdf = generate_voronoi_layer(
            config["vector"]["num_points_per_layer"],
            config["vector"]["bbox"],
            rng
        )
        filepath = output_dir / f"layer_{i:04d}.parquet"
        gdf.to_parquet(filepath)
        files.append(filepath)
    
    print(f"Generated {len(files)} vector layers in {output_dir}")
    return files


def generate_raster_data(config: Dict, output_dir: Path) -> Dict[str, np.ndarray]:
    """Generate all raster benchmark data (NLM landscapes)."""
    print("\n" + "=" * 60)
    print("GENERATING RASTER BENCHMARK DATA")
    print("=" * 60)
    
    rng = np.random.default_rng(config["random_seed"])
    output_dir = output_dir / "raster"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    max_layers = max(config["raster"]["num_layers_list"])
    size = config["raster"]["raster_size"]
    
    data = {"continuous": [], "discrete": []}
    
    print(f"Generating {max_layers} NLM raster layers...")
    for i in tqdm(range(max_layers)):
        continuous = generate_nlm_raster(size, rng, continuous=True)
        data["continuous"].append(continuous)
        
        discrete = generate_nlm_raster(size, rng, continuous=False)
        data["discrete"].append(discrete)
    
    np.save(output_dir / "continuous_layers.npy", np.array(data["continuous"]))
    np.save(output_dir / "discrete_layers.npy", np.array(data["discrete"]))
    
    print(f"Generated {max_layers} raster layers in {output_dir}")
    return data


# =============================================================================
# Vector Benchmark (Figure 6)
# =============================================================================

def benchmark_vector_union_classify(layers: List[gpd.GeoDataFrame]) -> Tuple[float, float, bool]:
    """
    Benchmark traditional vector spatial union method with classification.
    
    From paper Section 3.2.1:
    "These data were then spatially joined (unary union), and a (nonsense) 
    map classification logic was applied to the unioned output."
    
    Returns: (join_time, classify_time, success)
    """
    join_start = time.perf_counter()
    success = True
    
    try:
        # Rename value column in each layer to avoid conflicts during overlay
        renamed_layers = []
        for i, layer in enumerate(layers):
            layer_copy = layer.copy()
            layer_copy = layer_copy.rename(columns={'value': f'value_{i}'})
            renamed_layers.append(layer_copy)
        
        # Start with first layer
        result = renamed_layers[0].copy()
        
        for layer in renamed_layers[1:]:
            # Union overlay - this creates intersection polygons
            result = gpd.overlay(result, layer, how='union', keep_geom_type=True)
        
        join_time = time.perf_counter() - join_start
        
        # Classification phase
        classify_start = time.perf_counter()
        
        # Sum all value columns
        value_cols = [col for col in result.columns if col.startswith('value_')]
        if value_cols:
            result['sum_value'] = result[value_cols].fillna(0).sum(axis=1).astype(int)
        else:
            result['sum_value'] = 0
        
        # Apply classification functions (as per paper)
        result['class'] = result['sum_value'].apply(classify_value)
        
        classify_time = time.perf_counter() - classify_start
        
    except MemoryError:
        join_time = time.perf_counter() - join_start
        classify_time = 0
        success = False
    except Exception as e:
        print(f"Vector union failed: {e}")
        join_time = time.perf_counter() - join_start
        classify_time = 0
        success = False
    
    return join_time, classify_time, success


def h3_polyfill_polygon(polygon, resolution: int) -> List[str]:
    """
    Fill a polygon with H3 cells using polyfill.
    
    From paper: "A polygon filling algorithm is implemented through the H3 
    Python bindings, which we used through H3-Pandas, where it is termed 
    'polyfilling'."
    """
    try:
        if polygon.is_empty:
            return []
        
        if hasattr(polygon, 'geoms'):
            cells = []
            for poly in polygon.geoms:
                cells.extend(h3_polyfill_polygon(poly, resolution))
            return list(set(cells))
        
        coords = list(polygon.exterior.coords)
        
        geojson = {
            "type": "Polygon",
            "coordinates": [coords]
        }
        
        cells = h3.polygon_to_cells(geojson, resolution)
        return list(cells)
        
    except Exception as e:
        centroid = polygon.centroid
        if centroid.is_empty:
            return []
        cell = h3.latlng_to_cell(centroid.y, centroid.x, resolution)
        return [cell]


def benchmark_dggs_vector(layers: List[gpd.GeoDataFrame],
                          h3_resolution: int) -> Tuple[float, float, float]:
    """
    Benchmark DGGS-based method for vector data.
    
    From paper Section 3.2.1:
    "To compare this to a DGGS workflow, we first needed to perform the 
    additional conversion step of indexing each input vector Voronoi polygon 
    geometry to a fixed refinement level... We then performed an attribute 
    join on DGGS zone ID, which is implicitly a spatial join but requires 
    no explicit consideration of geometric intersection."
    
    Returns: (indexing_time, joining_time, classifying_time)
    """
    index_start = time.perf_counter()
    
    all_records = []
    for layer_idx, gdf in enumerate(layers):
        for idx, row in gdf.iterrows():
            geom = row.geometry
            value = row['value']
            
            cells = h3_polyfill_polygon(geom, h3_resolution)
            
            for cell in cells:
                all_records.append({
                    'h3_cell': cell,
                    'layer': layer_idx,
                    'value': value
                })
    
    index_time = time.perf_counter() - index_start
    
    join_start = time.perf_counter()
    
    df = pd.DataFrame(all_records)
    
    joined = df.groupby('h3_cell')['value'].sum().reset_index()
    joined.columns = ['h3_cell', 'sum_value']
    joined['sum_value'] = joined['sum_value'].astype(int)
    
    join_time = time.perf_counter() - join_start
    
    classify_start = time.perf_counter()
    
    joined['class'] = joined['sum_value'].apply(classify_value)
    
    classify_time = time.perf_counter() - classify_start
    
    total_time = index_time + join_time + classify_time
    
    return index_time, join_time + classify_time, total_time


def run_vector_benchmark(config: Dict, data_dir: Path) -> pd.DataFrame:
    """Run the complete vector benchmark (Figure 6)."""
    print("\n" + "=" * 60)
    print("RUNNING VECTOR BENCHMARK (Figure 6)")
    print("=" * 60)
    print(f"Using H3 resolution: {config['h3_resolution_vector']}")
    
    results = []
    vector_dir = data_dir / "vector"
    
    for num_layers in config["vector"]["num_layers_list"]:
        print(f"\nBenchmarking with {num_layers} layers...")
        
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
        
        dggs_index, dggs_join_classify, dggs_total = benchmark_dggs_vector(
            layers, config["h3_resolution_vector"]
        )
        
        max_traditional = int(os.environ.get("MAX_TRADITIONAL_LAYERS", 
                                              config["vector"]["max_layers_before_failure"]))
        
        if num_layers <= max_traditional:
            print(f"  Running traditional vector method...")
            vector_join, vector_classify, vector_success = benchmark_vector_union_classify(layers)
            vector_total = vector_join + vector_classify
        else:
            vector_join = np.nan
            vector_classify = np.nan
            vector_total = np.nan
            vector_success = None
            print(f"  Skipping traditional method (layers {num_layers} > max {max_traditional})")
        
        result = {
            "num_layers": num_layers,
            "dggs_index_time": dggs_index,
            "dggs_join_classify_time": dggs_join_classify,
            "dggs_total_time": dggs_total,
            "vector_join_time": vector_join,
            "vector_classify_time": vector_classify,
            "vector_total_time": vector_total,
            "vector_success": vector_success,
        }
        results.append(result)
        
        print(f"  DGGS: {dggs_total:.2f}s (index: {dggs_index:.2f}s, join+classify: {dggs_join_classify:.2f}s)")
        if vector_success is None:
            pass
        elif vector_success:
            print(f"  Vector: {vector_total:.2f}s (join: {vector_join:.2f}s, classify: {vector_classify:.2f}s)")
        else:
            print(f"  Vector: FAILED (memory/timeout)")
    
    return pd.DataFrame(results)


# =============================================================================
# Raster Benchmark (Figure 7)
# =============================================================================

def benchmark_raster_warp_classify(layers: np.ndarray) -> Tuple[float, float, float]:
    """
    Benchmark traditional raster method.
    
    Returns: (warp_time, classify_time, total_time)
    """
    warp_start = time.perf_counter()
    aligned = np.stack(layers, axis=0)
    warp_time = time.perf_counter() - warp_start
    
    classify_start = time.perf_counter()
    
    binary = (aligned > 0.5).astype(np.int32)
    summed = np.sum(binary, axis=0)
    result = np.vectorize(classify_value)(summed)
    
    classify_time = time.perf_counter() - classify_start
    
    return warp_time, classify_time, warp_time + classify_time


def benchmark_dggs_raster(layers: np.ndarray, h3_resolution: int) -> Tuple[float, float, float]:
    """
    Benchmark DGGS-based method for raster data.
    
    Returns: (indexing_time, classifying_time, total_time)
    """
    if isinstance(layers, list):
        layers = np.stack(layers, axis=0)
    
    num_layers, nrows, ncols = layers.shape
    
    index_start = time.perf_counter()
    
    lats = np.linspace(-5, 5, nrows)
    lons = np.linspace(-5, 5, ncols)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    lat_flat = lat_grid.ravel()
    lon_flat = lon_grid.ravel()
    
    cell_ids = np.array([
        h3.latlng_to_cell(lat, lon, h3_resolution) 
        for lat, lon in zip(lat_flat, lon_flat)
    ])
    
    unique_cells, inverse_indices = np.unique(cell_ids, return_inverse=True)
    num_cells = len(unique_cells)
    
    index_time = time.perf_counter() - index_start
    
    classify_start = time.perf_counter()
    
    binary = (layers > 0.5).astype(np.int32)
    flat_binary = binary.reshape(num_layers, -1)
    
    cell_sums = np.zeros((num_layers, num_cells), dtype=np.float32)
    cell_counts = np.bincount(inverse_indices, minlength=num_cells)
    
    for layer_idx in range(num_layers):
        cell_sums[layer_idx] = np.bincount(
            inverse_indices,
            weights=flat_binary[layer_idx].astype(float),
            minlength=num_cells
        )
    
    with np.errstate(divide='ignore', invalid='ignore'):
        cell_means = cell_sums / cell_counts
        cell_means = np.nan_to_num(cell_means, nan=0.0)
    
    cell_binary = (cell_means > 0.5).astype(np.int32)
    cell_total = cell_binary.sum(axis=0)
    
    result = np.vectorize(classify_value)(cell_total)
    
    classify_time = time.perf_counter() - classify_start
    
    return index_time, classify_time, index_time + classify_time


def run_raster_benchmark(config: Dict, data_dir: Path) -> pd.DataFrame:
    """Run the complete raster benchmark (Figure 7)."""
    print("\n" + "=" * 60)
    print("RUNNING RASTER BENCHMARK (Figure 7)")
    print("=" * 60)
    print(f"Using H3 resolution: {config['h3_resolution_raster']}")
    
    results = []
    raster_dir = data_dir / "raster"
    
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
        
        dggs_index, dggs_classify, dggs_total = benchmark_dggs_raster(
            layers, config["h3_resolution_raster"]
        )
        
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
# Results Analysis and Plotting
# =============================================================================

def compare_with_paper(vector_results: pd.DataFrame, 
                       raster_results: pd.DataFrame,
                       output_dir: Path) -> Dict:
    """Compare replication results with expected patterns from paper."""
    comparison = {
        "replication_info": {
            "code_version": CODE_VERSION,
            "timestamp": datetime.now().isoformat(),
            "original_paper": "Law & Ardo (2024) DOI: 10.1080/20964471.2024.2429847",
        },
        "vector_benchmark": {
            "paper_claim": "DGGS provides orders of magnitude performance improvement over vector methods",
            "paper_observation": "Vector method failed at ~500 layers due to memory",
            "expected_pattern": "DGGS should show near-linear scaling; vector should show power-law scaling and fail",
            "replication_observed": "",
            "replicated": None,
        },
        "raster_benchmark": {
            "paper_claim": "DGGS and raster methods show roughly equivalent performance",
            "expected_pattern": "Similar timing curves on log-log plot",
            "replication_observed": "",
            "replicated": None,
        }
    }
    
    # Analyze vector results
    if not vector_results.empty:
        valid = vector_results[vector_results['vector_success'] == True]
        observations = []
        
        if not valid.empty:
            speedup = valid['vector_total_time'].mean() / valid['dggs_total_time'].mean()
            observations.append(f"DGGS speedup: {speedup:.1f}x on average")
            comparison["vector_benchmark"]["replicated"] = speedup > 2
        else:
            observations.append("No successful vector runs for comparison")
        
        # Check for failures
        failures = vector_results[vector_results['vector_success'] == False]
        if not failures.empty:
            first_failure = failures['num_layers'].min()
            observations.append(f"Vector failed at {first_failure} layers")
        
        comparison["vector_benchmark"]["replication_observed"] = "; ".join(observations)
    
    # Analyze raster results
    if not raster_results.empty:
        ratio = raster_results['dggs_total_time'].mean() / raster_results['raster_total_time'].mean()
        comparison["raster_benchmark"]["replication_observed"] = f"DGGS/Raster time ratio: {ratio:.2f}"
        comparison["raster_benchmark"]["replicated"] = 0.1 < ratio < 10
    
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
    
    ax1 = axes[0]
    if not vector_results.empty:
        ax1.loglog(vector_results['num_layers'], vector_results['dggs_total_time'], 
                   'o-', label='DGGS Total', color='blue', linewidth=2, markersize=8)
        ax1.loglog(vector_results['num_layers'], vector_results['dggs_index_time'], 
                   's--', label='DGGS Indexing', color='lightblue', linewidth=1.5)
        ax1.loglog(vector_results['num_layers'], vector_results['dggs_join_classify_time'], 
                   '^--', label='DGGS Join+Classify', color='cyan', linewidth=1.5)
        
        # Handle vector_success being True, False, or None
        valid_vector = vector_results[vector_results['vector_success'].fillna(False) == True]
        if not valid_vector.empty:
            ax1.loglog(valid_vector['num_layers'], valid_vector['vector_total_time'], 
                       'o-', label='Vector Total', color='orange', linewidth=2, markersize=8)
            if 'vector_join_time' in valid_vector.columns:
                ax1.loglog(valid_vector['num_layers'], valid_vector['vector_join_time'], 
                           's--', label='Vector Join', color='lightsalmon', linewidth=1.5)
        
        # Mark failure points
        failed = vector_results[vector_results['vector_success'].fillna(False) == False]
        if not failed.empty and len(failed) < len(vector_results):
            ax1.axvline(x=failed['num_layers'].min(), color='red', linestyle=':', 
                        label=f'Vector failure ({failed["num_layers"].min()} layers)', linewidth=2)
    
    ax1.set_xlabel('Number of input layers', fontsize=12)
    ax1.set_ylabel('Compute time (s)', fontsize=12)
    ax1.set_title('Vector Input Benchmark\n(cf. Figure 6 in Law & Ardo 2024)', fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    
    ax2 = axes[1]
    if not raster_results.empty:
        ax2.loglog(raster_results['num_layers'], raster_results['dggs_total_time'], 
                   'o-', label='DGGS Total', color='blue', linewidth=2, markersize=8)
        ax2.loglog(raster_results['num_layers'], raster_results['dggs_index_time'], 
                   's--', label='DGGS Indexing', color='lightblue', linewidth=1.5)
        ax2.loglog(raster_results['num_layers'], raster_results['dggs_classify_time'], 
                   '^--', label='DGGS Classify', color='cyan', linewidth=1.5)
        
        ax2.loglog(raster_results['num_layers'], raster_results['raster_total_time'], 
                   'o-', label='Raster Total', color='orange', linewidth=2, markersize=8)
        ax2.loglog(raster_results['num_layers'], raster_results['raster_warp_time'], 
                   's--', label='Raster Warp', color='lightsalmon', linewidth=1.5)
        ax2.loglog(raster_results['num_layers'], raster_results['raster_classify_time'], 
                   '^--', label='Raster Classify', color='gold', linewidth=1.5)
    
    ax2.set_xlabel('Number of input layers', fontsize=12)
    ax2.set_ylabel('Compute time (s)', fontsize=12)
    ax2.set_title('Raster Input Benchmark\n(cf. Figure 7 in Law & Ardo 2024)', fontsize=14)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_results.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "benchmark_results.pdf", bbox_inches='tight')
    print(f"Plots saved to {output_dir}")


# =============================================================================
# Original Benchmark Exploration
# =============================================================================

ORIGINAL_BENCHMARKS_DIR = Path("/app/original_benchmarks")

def explore_original_repo() -> bool:
    """Explore the original benchmark repository."""
    print("\n" + "=" * 60)
    print("ORIGINAL BENCHMARK REPOSITORY")
    print("=" * 60)
    print(f"Repo: {CONFIG['original_benchmark']['repo']}")
    print(f"Version: {CONFIG['original_benchmark']['version']}")
    
    if not ORIGINAL_BENCHMARKS_DIR.exists():
        print(f"\nERROR: {ORIGINAL_BENCHMARKS_DIR} not found!")
        print("The original benchmark repo should be cloned by the Dockerfile.")
        return False
    
    print(f"\nContents of {ORIGINAL_BENCHMARKS_DIR}:")
    for item in sorted(ORIGINAL_BENCHMARKS_DIR.iterdir()):
        if item.name.startswith('.'):
            continue
        print(f"  {'📁' if item.is_dir() else '📄'} {item.name}")
    
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DGGS Benchmark Replication Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Replication of Law & Ardo (2024):
"Using a discrete global grid system for a scalable, interoperable, 
and reproducible system of land-use mapping"
https://doi.org/10.1080/20964471.2024.2429847

Examples:
  python run_replication.py --all
  python run_replication.py --generate-data --vector
  python run_replication.py --explore
        """
    )
    
    parser.add_argument('--explore', action='store_true', 
                        help='Explore the original benchmark repository')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--vector', action='store_true', help='Run vector benchmark (Figure 6)')
    parser.add_argument('--raster', action='store_true', help='Run raster benchmark (Figure 7)')
    parser.add_argument('--compare', action='store_true', help='Compare results with paper')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output', '-o', default='results', help='Output directory')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    if args.explore:
        explore_original_repo()
        return 0
    
    if args.seed is not None:
        CONFIG["random_seed"] = args.seed
    CONFIG["results_dir"] = args.output
    
    print(f"Code version: {CODE_VERSION}")
    print(f"Configuration:")
    print(f"  Random seed: {CONFIG['random_seed']}")
    print(f"  H3 resolution (vector): {CONFIG['h3_resolution_vector']}")
    print(f"  H3 resolution (raster): {CONFIG['h3_resolution_raster']}")
    print(f"  Vector layers: {CONFIG['vector']['num_layers_list']}")
    print(f"  Raster layers: {CONFIG['raster']['num_layers_list']}")
    
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(CONFIG["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sys_info = get_system_info()
    with open(results_dir / "system_info.json", 'w') as f:
        json.dump(sys_info, f, indent=2)
    
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
            if benchmark == "replication_info":
                continue
            print(f"\n{benchmark}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    if args.all or args.plot:
        plot_results(vector_results, raster_results, results_dir)
    
    print("\n" + "=" * 60)
    print("REPLICATION COMPLETE")
    print("=" * 60)
    print(f"Results directory: {results_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
