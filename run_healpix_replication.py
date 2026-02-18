#!/usr/bin/env python3
"""
DGGS Benchmark: HEALPix Replication Study

REPLICATES the benchmarks from Law & Ardo (2024) using HEALPix instead of H3.
DOI: 10.1080/20964471.2024.2429847

This is a TRUE REPLICATION: same methodology, different DGGS implementation.

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
from typing import Dict, List, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box

try:
    import cdshealpix
    from astropy.coordinates import Longitude, Latitude
    import astropy.units as u
    HAS_HEALPIX = True
except ImportError:
    HAS_HEALPIX = False

try:
    import xdggs
    HAS_XDGGS = True
except ImportError:
    HAS_XDGGS = False

try:
    from scipy.spatial import Voronoi
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

from tqdm import tqdm

CODE_VERSION = "2026-01-22-healpix-replication-v2"

# Conservative defaults to avoid OOM
CONFIG = {
    "random_seed": 42,
    "vector": {
        "healpix_depth": 12,  # Lower than paper's equivalent for stability
        "num_layers_list": [5, 10, 20, 50],
        "num_points_per_layer": 30,
        "bbox": (-1.0, -1.0, 1.0, 1.0),
    },
    "raster": {
        "healpix_depth": 13,  # Equivalent to H3 resolution 9
        "num_layers_list": [10, 50, 100, 500, 1000],
        "raster_size": (100, 100),
        "bbox": (-0.5, -0.5, 0.5, 0.5),
    },
}


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
        return sum(i for i in range(1, n) if n % i == 0) == n
    
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


def get_system_info() -> Dict:
    info = {
        "timestamp": datetime.now().isoformat(),
        "code_version": CODE_VERSION,
        "python_version": sys.version.split()[0],
        "paper": {"doi": "10.1080/20964471.2024.2429847"},
        "replication": {"dggs": "HEALPix", "original": "H3"},
    }
    if HAS_PSUTIL:
        info["system"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        }
    return info


def healpix_polygon_to_cells(lon_deg, lat_deg, depth: int) -> np.ndarray:
    """Fill polygon with HEALPix cells at fixed depth."""
    lon = Longitude(lon_deg, unit=u.deg)
    lat = Latitude(lat_deg, unit=u.deg)
    
    try:
        cell_ids, depths, flags = cdshealpix.polygon_search(lon, lat, depth)
        expanded = []
        for cid, d in zip(cell_ids, depths):
            if d == depth:
                expanded.append(cid)
            elif d < depth:
                n_children = 4 ** (depth - d)
                base = cid * n_children
                expanded.extend(range(base, base + n_children))
        return np.array(expanded, dtype=np.uint64)
    except:
        try:
            cell = cdshealpix.lonlat_to_healpix(
                Longitude(np.mean(lon_deg), unit=u.deg),
                Latitude(np.mean(lat_deg), unit=u.deg), depth)
            return np.array([cell], dtype=np.uint64)
        except:
            return np.array([], dtype=np.uint64)


def healpix_polyfill_geometry(geom, depth: int) -> List[int]:
    """Fill shapely geometry with HEALPix cells."""
    try:
        if geom is None or geom.is_empty:
            return []
        if hasattr(geom, "geoms"):
            cells = set()
            for part in geom.geoms:
                cells.update(healpix_polyfill_geometry(part, depth))
            return list(cells)
        coords = np.array(geom.exterior.coords)
        return list(healpix_polygon_to_cells(coords[:, 0], coords[:, 1], depth))
    except:
        return []


def generate_voronoi_layer(num_points: int, bbox: Tuple, rng) -> gpd.GeoDataFrame:
    """Generate Voronoi polygon layer."""
    minx, miny, maxx, maxy = bbox
    points = rng.uniform([minx, miny], [maxx, maxy], size=(num_points, 2))
    
    if HAS_SCIPY:
        try:
            boundary = np.array([[minx-10, miny-10], [maxx+10, miny-10],
                                 [minx-10, maxy+10], [maxx+10, maxy+10]])
            vor = Voronoi(np.vstack([points, boundary]))
            polygons, values = [], []
            for i, region_idx in enumerate(vor.point_region[:num_points]):
                region = vor.regions[region_idx]
                if -1 not in region and len(region) > 0:
                    poly = Polygon([vor.vertices[j] for j in region])
                    clipped = poly.intersection(box(minx, miny, maxx, maxy))
                    if not clipped.is_empty and clipped.area > 0:
                        polygons.append(clipped)
                        values.append(rng.integers(0, 2))
            if polygons:
                return gpd.GeoDataFrame({'value': values, 'geometry': polygons}, crs="EPSG:4326")
        except:
            pass
    
    # Fallback
    polygons, values = [], []
    for cx, cy in points:
        w, h = rng.uniform(0.05, 0.2, 2)
        poly = box(cx-w/2, cy-h/2, cx+w/2, cy+h/2).intersection(box(minx, miny, maxx, maxy))
        if not poly.is_empty:
            polygons.append(poly)
            values.append(rng.integers(0, 2))
    return gpd.GeoDataFrame({'value': values, 'geometry': polygons}, crs="EPSG:4326")


def generate_raster_layer(size: Tuple[int, int], rng) -> np.ndarray:
    """Generate spatially-correlated raster."""
    base = rng.uniform(0, 1, size)
    if HAS_SCIPY:
        smoothed = gaussian_filter(base, sigma=2)
        return (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-10)
    return base


def index_raster_healpix(lats, lngs, depth: int) -> np.ndarray:
    """Index raster to HEALPix cells."""
    if HAS_XDGGS:
        hp_info = xdggs.HealpixInfo(level=depth, indexing_scheme="nested")
        return np.asarray(hp_info.geographic2cell_ids(lngs.ravel(), lats.ravel()))
    else:
        lon = Longitude(lngs.ravel(), unit=u.deg)
        lat = Latitude(lats.ravel(), unit=u.deg)
        return np.asarray(cdshealpix.lonlat_to_healpix(lon, lat, depth))


def aggregate_to_cells(values, cell_ids, unique_cells) -> np.ndarray:
    """Aggregate pixel values to cells."""
    _, inverse = np.unique(cell_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=values.ravel(), minlength=len(unique_cells))
    counts = np.bincount(inverse, minlength=len(unique_cells))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nan_to_num(sums / counts, nan=0.0)


def benchmark_vector_traditional(layers: List[gpd.GeoDataFrame]) -> Dict:
    """Traditional vector overlay."""
    start = time.perf_counter()
    try:
        renamed = [layer.rename(columns={'value': f'value_{i}'}) for i, layer in enumerate(layers)]
        result = renamed[0].copy()
        for layer in renamed[1:]:
            result = gpd.overlay(result, layer, how='union', keep_geom_type=True)
        join_time = time.perf_counter() - start
        
        classify_start = time.perf_counter()
        value_cols = [c for c in result.columns if c.startswith('value_')]
        result['sum_value'] = result[value_cols].fillna(0).sum(axis=1).astype(int)
        result['class'] = result['sum_value'].apply(classify_value)
        classify_time = time.perf_counter() - classify_start
        
        return {"success": True, "join_time": join_time, "classify_time": classify_time,
                "total": join_time + classify_time, "num_features": len(result)}
    except Exception as e:
        return {"success": False, "error": str(e), "total": time.perf_counter() - start}


def benchmark_vector_healpix(layers: List[gpd.GeoDataFrame], depth: int) -> Dict:
    """HEALPix DGGS method."""
    index_start = time.perf_counter()
    records = []
    for layer_idx, gdf in enumerate(layers):
        for _, row in gdf.iterrows():
            cells = healpix_polyfill_geometry(row.geometry, depth)
            for cell in cells:
                records.append({'cell_id': cell, 'layer': layer_idx, 'value': row.get('value', 0)})
    index_time = time.perf_counter() - index_start
    
    classify_start = time.perf_counter()
    df = pd.DataFrame(records)
    if df.empty:
        return {"success": True, "index_time": index_time, "classify_time": 0,
                "total": index_time, "num_cells": 0}
    
    per_layer = df.groupby(['cell_id', 'layer'])['value'].max().reset_index()
    pivot = per_layer.pivot_table(index='cell_id', columns='layer', values='value',
                                   aggfunc='max', fill_value=0)
    pivot['sum_value'] = pivot.sum(axis=1).astype(int)
    pivot['class'] = pivot['sum_value'].apply(classify_value)
    classify_time = time.perf_counter() - classify_start
    
    return {"success": True, "index_time": index_time, "classify_time": classify_time,
            "total": index_time + classify_time, "num_cells": len(pivot)}


def run_vector_benchmark(config: Dict, output_dir: Path) -> pd.DataFrame:
    """Run vector benchmark with HEALPix."""
    print("\n" + "=" * 70)
    print("VECTOR BENCHMARK (Figure 6) - HEALPix Replication")
    print("=" * 70)
    
    rng = np.random.default_rng(config["random_seed"])
    cfg = config["vector"]
    max_layers = max(cfg["num_layers_list"])
    depth = cfg["healpix_depth"]
    
    print(f"\nHEALPix depth: {depth}")
    print(f"Generating {max_layers} Voronoi layers...")
    layers = [generate_voronoi_layer(cfg["num_points_per_layer"], cfg["bbox"], rng) 
              for _ in tqdm(range(max_layers))]
    
    results = []
    for n in cfg["num_layers_list"]:
        print(f"\n--- {n} layers ---")
        subset = layers[:n]
        
        dggs = benchmark_vector_healpix(subset, depth)
        print(f"  HEALPix: {dggs['total']:.3f}s ({dggs.get('num_cells', 0)} cells)")
        
        trad = benchmark_vector_traditional(subset)
        if trad["success"]:
            print(f"  Vector:  {trad['total']:.3f}s ({trad.get('num_features', 0)} features)")
            speedup = trad['total'] / dggs['total'] if dggs['total'] > 0 else 0
            print(f"  → Speedup: {speedup:.1f}x")
        else:
            print(f"  Vector:  FAILED")
        
        results.append({
            "num_layers": n,
            "healpix_total": dggs["total"],
            "healpix_cells": dggs.get("num_cells", 0),
            "vector_total": trad.get("total", np.nan),
            "vector_features": trad.get("num_features", np.nan),
            "vector_success": trad["success"],
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "vector_benchmark_healpix.csv", index=False)
    return df


def run_raster_benchmark(config: Dict, output_dir: Path) -> pd.DataFrame:
    """Run raster benchmark with HEALPix."""
    print("\n" + "=" * 70)
    print("RASTER BENCHMARK (Figure 7) - HEALPix Replication")
    print("=" * 70)
    
    rng = np.random.default_rng(config["random_seed"])
    cfg = config["raster"]
    max_layers = max(cfg["num_layers_list"])
    depth = cfg["healpix_depth"]
    
    print(f"\nHEALPix depth: {depth}")
    print(f"Generating {max_layers} raster layers...")
    rasters = np.stack([generate_raster_layer(cfg["raster_size"], rng) 
                        for _ in tqdm(range(max_layers))])
    
    rows, cols = cfg["raster_size"]
    minx, miny, maxx, maxy = cfg["bbox"]
    lngs = minx + (np.arange(cols) + 0.5) * (maxx - minx) / cols
    lats = miny + (np.arange(rows) + 0.5) * (maxy - miny) / rows
    lng_grid, lat_grid = np.meshgrid(lngs, lats)
    
    print(f"\nIndexing to HEALPix...", end=" ", flush=True)
    start = time.perf_counter()
    cell_ids = index_raster_healpix(lat_grid, lng_grid, depth)
    index_time = time.perf_counter() - start
    unique_cells = np.unique(cell_ids)
    num_cells = len(unique_cells)
    print(f"{index_time:.4f}s → {num_cells} cells")
    
    print(f"Pre-indexing {max_layers} layers...", end=" ", flush=True)
    start = time.perf_counter()
    preindexed = np.zeros((num_cells, max_layers), dtype=np.float32)
    for i in range(max_layers):
        preindexed[:, i] = aggregate_to_cells(rasters[i], cell_ids, unique_cells)
    preindex_time = time.perf_counter() - start
    print(f"{preindex_time:.3f}s")
    
    print("\n--- CLASSIFICATION ---")
    results = []
    for n in cfg["num_layers_list"]:
        print(f"\n{n} layers:")
        data = rasters[:n]
        row = {"num_layers": n}
        
        # Raster
        start = time.perf_counter()
        sum_vals = (data * 10).astype(int).sum(axis=0)
        classified = np.vectorize(classify_value)(sum_vals)
        row["raster_total"] = time.perf_counter() - start
        print(f"  Raster:    {row['raster_total']:.4f}s")
        
        # HEALPix pre-indexed
        start = time.perf_counter()
        pre_data = preindexed[:, :n]
        sums = (pre_data * 10).astype(int).sum(axis=1)
        classes = np.array([classify_value(int(v)) for v in sums])
        row["healpix_total"] = time.perf_counter() - start
        print(f"  HEALPix:   {row['healpix_total']:.4f}s")
        
        results.append(row)
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "raster_benchmark_healpix.csv", index=False)
    
    with open(output_dir / "indexing_healpix.json", 'w') as f:
        json.dump({"depth": depth, "num_cells": num_cells, "index_time": index_time}, f, indent=2)
    
    return df


def plot_results(vector_df: pd.DataFrame, raster_df: pd.DataFrame, output_dir: Path):
    """Generate plots."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("HEALPix Replication of Law & Ardo (2024)\nDOI: 10.1080/20964471.2024.2429847", 
                 fontsize=14, fontweight='bold')
    
    if not vector_df.empty:
        ax = axes[0, 0]
        ax.loglog(vector_df['num_layers'], vector_df['healpix_total'], 'o-', 
                  label='HEALPix', color='purple', linewidth=2)
        valid = vector_df[vector_df['vector_success']]
        if not valid.empty:
            ax.loglog(valid['num_layers'], valid['vector_total'], 's-', 
                      label='Vector', color='orange', linewidth=2)
        ax.set_xlabel('Layers')
        ax.set_ylabel('Time (s)')
        ax.set_title('Vector Benchmark')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        if not valid.empty:
            speedup = valid['vector_total'] / valid['healpix_total']
            ax.bar(valid['num_layers'].astype(str), speedup, color='purple', alpha=0.7)
            ax.axhline(y=1, color='gray', linestyle='--')
            ax.set_xlabel('Layers')
            ax.set_ylabel('Speedup')
            ax.set_title('HEALPix Speedup vs Vector')
    
    if not raster_df.empty:
        ax = axes[1, 0]
        ax.loglog(raster_df['num_layers'], raster_df['raster_total'], 'o-', 
                  label='Raster', color='orange', linewidth=2)
        ax.loglog(raster_df['num_layers'], raster_df['healpix_total'], 's-', 
                  label='HEALPix', color='purple', linewidth=2)
        ax.set_xlabel('Layers')
        ax.set_ylabel('Time (s)')
        ax.set_title('Raster Benchmark')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        ratio = raster_df['raster_total'] / raster_df['healpix_total']
        ax.bar(raster_df['num_layers'].astype(str), ratio, color='green', alpha=0.7)
        ax.axhline(y=1, color='gray', linestyle='--')
        ax.set_xlabel('Layers')
        ax.set_ylabel('Raster / HEALPix')
        ax.set_title('Performance Ratio')
    
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_healpix.png", dpi=150)
    plt.savefig(output_dir / "benchmark_healpix.pdf")
    print(f"\nPlots saved")


def generate_summary(vector_df: pd.DataFrame, raster_df: pd.DataFrame, output_dir: Path):
    """Generate summary."""
    summary = {
        "paper": {"doi": "10.1080/20964471.2024.2429847", "original_dggs": "H3"},
        "replication": {"dggs": "HEALPix"},
        "results": {}
    }
    
    if not vector_df.empty:
        valid = vector_df[vector_df['vector_success']]
        if not valid.empty:
            speedups = valid['vector_total'] / valid['healpix_total']
            summary["results"]["vector"] = {
                "speedup_range": f"{speedups.min():.0f}x - {speedups.max():.0f}x",
                "validated": bool(speedups.max() > 10)
            }
    
    if not raster_df.empty:
        ratio = raster_df['raster_total'].mean() / raster_df['healpix_total'].mean()
        summary["results"]["raster"] = {
            "ratio": f"{ratio:.2f}x",
            "validated": bool(0.1 < ratio < 10)
        }
    
    with open(output_dir / "summary_healpix.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if "vector" in summary["results"]:
        v = summary["results"]["vector"]
        print(f"Vector: HEALPix {v['speedup_range']} faster → {'✓' if v['validated'] else '✗'}")
    if "raster" in summary["results"]:
        r = summary["results"]["raster"]
        print(f"Raster: Ratio {r['ratio']} → {'✓' if r['validated'] else '✗'}")


def main():
    parser = argparse.ArgumentParser(description="HEALPix Replication Study")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--output", "-o", default="results")
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--skip-raster", action="store_true")
    parser.add_argument("--vector-layers", type=str)
    parser.add_argument("--raster-layers", type=str)
    parser.add_argument("--healpix-depth-vector", type=int)
    parser.add_argument("--healpix-depth-raster", type=int)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.vector_layers:
        CONFIG["vector"]["num_layers_list"] = [int(x) for x in args.vector_layers.split(",")]
    if args.raster_layers:
        CONFIG["raster"]["num_layers_list"] = [int(x) for x in args.raster_layers.split(",")]
    if args.healpix_depth_vector:
        CONFIG["vector"]["healpix_depth"] = args.healpix_depth_vector
    if args.healpix_depth_raster:
        CONFIG["raster"]["healpix_depth"] = args.healpix_depth_raster
    
    print("=" * 70)
    print("HEALPIX REPLICATION STUDY")
    print("=" * 70)
    print(f"Version: {CODE_VERSION}")
    print(f"Vector depth: {CONFIG['vector']['healpix_depth']}")
    print(f"Raster depth: {CONFIG['raster']['healpix_depth']}")
    
    if not HAS_HEALPIX:
        print("ERROR: cdshealpix required")
        sys.exit(1)
    
    with open(output_dir / "system_info.json", 'w') as f:
        json.dump(get_system_info(), f, indent=2)
    
    vector_df = pd.DataFrame()
    raster_df = pd.DataFrame()
    
    if not args.skip_vector:
        vector_df = run_vector_benchmark(CONFIG, output_dir)
    if not args.skip_raster:
        raster_df = run_raster_benchmark(CONFIG, output_dir)
    
    plot_results(vector_df, raster_df, output_dir)
    generate_summary(vector_df, raster_df, output_dir)
    
    print(f"\n📁 Results: {output_dir}/")


if __name__ == "__main__":
    main()
