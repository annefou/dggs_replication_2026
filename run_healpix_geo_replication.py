#!/usr/bin/env python3
"""
DGGS Benchmark: HEALPix-geo Replication Study
Sphere vs. WGS84 Ellipsoid Extension

REPLICATES the benchmarks from Law & Ardo (2024) using healpix-geo:
"Using a discrete global grid system for a scalable, interoperable,
and reproducible system of land-use mapping"
DOI: 10.1080/20964471.2024.2429847

This is a TRUE REPLICATION: same methodology, different DGGS implementation
AND a GEODETIC EXTENSION: compares sphere vs. WGS84 ellipsoid indexing.

Key contribution vs. run_healpix_replication.py:
  - Uses healpix-geo instead of cdshealpix + astropy (no astropy dependency)
  - Runs ALL benchmarks twice: once on sphere, once on WGS84 ellipsoid
  - Adds an ellipsoid comparison analysis quantifying geodetic differences
  - Particularly relevant for high-latitude EO data (Sentinel, Copernicus)

Author: Anne Fouilloux
Date: 2026-03-07
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
from tqdm import tqdm

# healpix-geo: no astropy required
try:
    import healpix_geo.nested as healpix_nested
    HAS_HEALPIX_GEO = True
    HEALPIX_GEO_VERSION = "installed"
    try:
        import importlib.metadata
        HEALPIX_GEO_VERSION = importlib.metadata.version("healpix-geo")
    except Exception:
        pass
except ImportError:
    HAS_HEALPIX_GEO = False
    HEALPIX_GEO_VERSION = "NOT INSTALLED"

try:
    from scipy.spatial import Voronoi
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# =============================================================================
# Configuration
# =============================================================================

CODE_VERSION = "2026-03-07-healpix-geo-v1"

# Ellipsoids to compare — the central scientific contribution
ELLIPSOIDS = ["sphere", "WGS84"]

CONFIG = {
    "random_seed": 42,
    "vector": {
        "healpix_depth": 9,        # depth=9 ≈ H3 resolution 9 (~5 km cells)
        "num_layers_list": [5, 10, 20, 50],
        "num_points_per_layer": 30,
        "bbox": (-1.0, -1.0, 1.0, 1.0),   # near-equatorial (low ellipsoid effect)
    },
    "raster": {
        "healpix_depth": 9,
        "num_layers_list": [10, 50, 100, 500],
        "raster_size": (100, 100),
        "bbox": (-0.5, -0.5, 0.5, 0.5),   # equatorial
    },
    # Extra bboxes to show latitude effect in the ellipsoid analysis
    "ellipsoid_analysis": {
        "healpix_depth": 9,
        "raster_size": (50, 50),
        "regions": {
            "equatorial":    (-10.0, -10.0,  10.0,  10.0),
            "mid_latitude":  ( -5.0,  40.0,  25.0,  55.0),   # Mediterranean
            "high_latitude": (  5.0,  55.0,  30.0,  70.0),   # Scandinavia
            "arctic":        (-20.0,  70.0,  30.0,  85.0),
        },
    },
}


# =============================================================================
# Classification (unchanged from paper)
# =============================================================================

@lru_cache(maxsize=100000)
def classify_value(sum_value: int) -> int:
    """7-bit classification based on number properties (Law & Ardo 2024)."""
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


# =============================================================================
# System Information
# =============================================================================

def get_system_info() -> Dict:
    info = {
        "timestamp": datetime.now().isoformat(),
        "code_version": CODE_VERSION,
        "python_version": sys.version.split()[0],
        "paper": {"doi": "10.1080/20964471.2024.2429847"},
        "replication": {
            "dggs_library": "healpix-geo",
            "dggs_library_version": HEALPIX_GEO_VERSION,
            "original_dggs": "H3",
            "ellipsoids_compared": ELLIPSOIDS,
        },
        "dependencies": {
            "healpix_geo": HEALPIX_GEO_VERSION,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "geopandas": gpd.__version__,
        },
    }
    if HAS_PSUTIL:
        info["system"] = {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        }
    return info


# =============================================================================
# HEALPix-geo helpers (sphere AND WGS84)
# =============================================================================

def healpix_polyfill_geometry(geom, depth: int, ellipsoid: str) -> np.ndarray:
    """
    Fill a shapely geometry with HEALPix cells using healpix-geo.
    Supports both 'sphere' and 'WGS84' ellipsoid.
    Returns array of uint64 cell IDs.
    """
    try:
        if geom is None or geom.is_empty:
            return np.array([], dtype=np.uint64)
        if hasattr(geom, "geoms"):
            cells = set()
            for part in geom.geoms:
                cells.update(healpix_polyfill_geometry(part, depth, ellipsoid).tolist())
            return np.array(list(cells), dtype=np.uint64)

        coords = np.array(geom.exterior.coords)
        cell_ids, _, _ = healpix_nested.polygon_coverage(
            coords, depth, ellipsoid=ellipsoid, flat=True
        )
        return cell_ids
    except Exception:
        # Fallback: use cell centroid
        try:
            c = geom.centroid
            cid = healpix_nested.lonlat_to_healpix(
                np.array([c.x]), np.array([c.y]), depth, ellipsoid=ellipsoid
            )
            return cid
        except Exception:
            return np.array([], dtype=np.uint64)


def index_raster_healpix(lats: np.ndarray, lngs: np.ndarray,
                          depth: int, ellipsoid: str) -> np.ndarray:
    """
    Index a raster grid to HEALPix cells using healpix-geo.
    Ellipsoid can be 'sphere' or 'WGS84'.
    """
    return healpix_nested.lonlat_to_healpix(
        lngs.ravel(), lats.ravel(), depth, ellipsoid=ellipsoid
    )


def aggregate_to_cells(values: np.ndarray, cell_ids: np.ndarray,
                        unique_cells: np.ndarray) -> np.ndarray:
    """Aggregate raster pixel values to unique HEALPix cells (mean)."""
    _, inverse = np.unique(cell_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=values.ravel(), minlength=len(unique_cells))
    counts = np.bincount(inverse, minlength=len(unique_cells))
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nan_to_num(sums / counts, nan=0.0)


# =============================================================================
# Data Generation
# =============================================================================

def generate_voronoi_layer(num_points: int, bbox: Tuple,
                            rng: np.random.Generator) -> gpd.GeoDataFrame:
    """Generate a Voronoi polygon layer (deterministic via rng)."""
    minx, miny, maxx, maxy = bbox
    points = rng.uniform([minx, miny], [maxx, maxy], size=(num_points, 2))

    if HAS_SCIPY:
        try:
            boundary = np.array([
                [minx - 10, miny - 10], [maxx + 10, miny - 10],
                [minx - 10, maxy + 10], [maxx + 10, maxy + 10],
            ])
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
                return gpd.GeoDataFrame(
                    {'value': values, 'geometry': polygons}, crs="EPSG:4326"
                )
        except Exception:
            pass

    # Fallback: random rectangles
    polygons, values = [], []
    for cx, cy in points:
        w, h = rng.uniform(0.05, 0.2, 2)
        poly = box(cx - w/2, cy - h/2, cx + w/2, cy + h/2)
        poly = poly.intersection(box(minx, miny, maxx, maxy))
        if not poly.is_empty:
            polygons.append(poly)
            values.append(rng.integers(0, 2))
    return gpd.GeoDataFrame({'value': values, 'geometry': polygons}, crs="EPSG:4326")


def generate_raster_layer(size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Generate a spatially-correlated raster layer."""
    base = rng.uniform(0, 1, size)
    if HAS_SCIPY:
        smoothed = gaussian_filter(base, sigma=2)
        return (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-10)
    return base


# =============================================================================
# Vector Benchmark
# =============================================================================

def benchmark_vector_traditional(layers: List[gpd.GeoDataFrame]) -> Dict:
    """Traditional vector overlay (same for both ellipsoids — serves as baseline)."""
    start = time.perf_counter()
    try:
        renamed = [layer.rename(columns={'value': f'value_{i}'})
                   for i, layer in enumerate(layers)]
        result = renamed[0].copy()
        for layer in renamed[1:]:
            result = gpd.overlay(result, layer, how='union', keep_geom_type=True)
        join_time = time.perf_counter() - start

        classify_start = time.perf_counter()
        value_cols = [c for c in result.columns if c.startswith('value_')]
        result['sum_value'] = result[value_cols].fillna(0).sum(axis=1).astype(int)
        result['class'] = result['sum_value'].apply(classify_value)
        classify_time = time.perf_counter() - classify_start

        return {
            "success": True,
            "join_time": join_time,
            "classify_time": classify_time,
            "total": join_time + classify_time,
            "num_features": len(result),
        }
    except Exception as e:
        return {"success": False, "error": str(e),
                "total": time.perf_counter() - start, "num_features": 0}


def benchmark_vector_healpix(layers: List[gpd.GeoDataFrame],
                               depth: int, ellipsoid: str) -> Dict:
    """HEALPix DGGS vector method using healpix-geo with chosen ellipsoid."""
    index_start = time.perf_counter()
    records = []
    for layer_idx, gdf in enumerate(layers):
        for _, row in gdf.iterrows():
            cells = healpix_polyfill_geometry(row.geometry, depth, ellipsoid)
            for cell in cells:
                records.append({
                    'cell_id': int(cell),
                    'layer': layer_idx,
                    'value': int(row.get('value', 0)),
                })
    index_time = time.perf_counter() - index_start

    classify_start = time.perf_counter()
    df = pd.DataFrame(records)
    if df.empty:
        return {"success": True, "ellipsoid": ellipsoid,
                "index_time": index_time, "classify_time": 0.0,
                "total": index_time, "num_cells": 0}

    per_layer = df.groupby(['cell_id', 'layer'])['value'].max().reset_index()
    pivot = per_layer.pivot_table(
        index='cell_id', columns='layer', values='value',
        aggfunc='max', fill_value=0
    )
    pivot['sum_value'] = pivot.sum(axis=1).astype(int)
    pivot['class'] = pivot['sum_value'].apply(classify_value)
    classify_time = time.perf_counter() - classify_start

    return {
        "success": True,
        "ellipsoid": ellipsoid,
        "index_time": index_time,
        "classify_time": classify_time,
        "total": index_time + classify_time,
        "num_cells": len(pivot),
    }


def run_vector_benchmark(config: Dict, output_dir: Path) -> pd.DataFrame:
    """Run vector benchmark for sphere AND WGS84 ellipsoids."""
    print("\n" + "=" * 70)
    print("VECTOR BENCHMARK (Figure 6) — healpix-geo: sphere vs WGS84")
    print("=" * 70)

    rng = np.random.default_rng(config["random_seed"])
    cfg = config["vector"]
    max_layers = max(cfg["num_layers_list"])
    depth = cfg["healpix_depth"]

    print(f"\nHEALPix depth: {depth}  |  bbox: {cfg['bbox']}")
    print(f"Generating {max_layers} Voronoi layers...")
    layers = [
        generate_voronoi_layer(cfg["num_points_per_layer"], cfg["bbox"], rng)
        for _ in tqdm(range(max_layers))
    ]

    results = []
    for n in cfg["num_layers_list"]:
        print(f"\n--- {n} layers ---")
        subset = layers[:n]
        row = {"num_layers": n}

        # Traditional vector (reference)
        trad = benchmark_vector_traditional(subset)
        row["vector_total"] = trad.get("total", np.nan)
        row["vector_features"] = trad.get("num_features", 0)
        row["vector_success"] = trad["success"]
        if trad["success"]:
            print(f"  Vector:       {trad['total']:.3f}s  ({trad['num_features']} features)")

        # HEALPix — sphere
        hp_s = benchmark_vector_healpix(subset, depth, "sphere")
        row["healpix_sphere_total"] = hp_s["total"]
        row["healpix_sphere_cells"] = hp_s["num_cells"]
        sp_s = trad["total"] / hp_s["total"] if hp_s["total"] > 0 and trad["success"] else np.nan
        print(f"  HEALPix/sphere: {hp_s['total']:.3f}s  ({hp_s['num_cells']} cells)  speedup={sp_s:.1f}x")

        # HEALPix — WGS84
        hp_e = benchmark_vector_healpix(subset, depth, "WGS84")
        row["healpix_wgs84_total"] = hp_e["total"]
        row["healpix_wgs84_cells"] = hp_e["num_cells"]
        sp_e = trad["total"] / hp_e["total"] if hp_e["total"] > 0 and trad["success"] else np.nan
        print(f"  HEALPix/WGS84:  {hp_e['total']:.3f}s  ({hp_e['num_cells']} cells)  speedup={sp_e:.1f}x")

        # Cell set difference
        cell_diff_pct = abs(hp_s["num_cells"] - hp_e["num_cells"]) / max(hp_s["num_cells"], 1) * 100
        row["cell_count_diff_pct"] = cell_diff_pct
        print(f"  Cell count diff: {cell_diff_pct:.1f}%")

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "vector_benchmark_healpix_geo.csv", index=False)
    print(f"\nSaved: vector_benchmark_healpix_geo.csv")
    return df


# =============================================================================
# Raster Benchmark
# =============================================================================

def run_raster_benchmark(config: Dict, output_dir: Path) -> pd.DataFrame:
    """Run raster benchmark for sphere AND WGS84 ellipsoids."""
    print("\n" + "=" * 70)
    print("RASTER BENCHMARK (Figure 7) — healpix-geo: sphere vs WGS84")
    print("=" * 70)

    rng = np.random.default_rng(config["random_seed"])
    cfg = config["raster"]
    max_layers = max(cfg["num_layers_list"])
    depth = cfg["healpix_depth"]
    rows_px, cols_px = cfg["raster_size"]
    minx, miny, maxx, maxy = cfg["bbox"]

    print(f"\nHEALPix depth: {depth}  |  raster: {cfg['raster_size']}  |  bbox: {cfg['bbox']}")
    print(f"Generating {max_layers} raster layers...")
    rasters = np.stack([
        generate_raster_layer(cfg["raster_size"], rng)
        for _ in tqdm(range(max_layers))
    ])

    lngs = minx + (np.arange(cols_px) + 0.5) * (maxx - minx) / cols_px
    lats = miny + (np.arange(rows_px) + 0.5) * (maxy - miny) / rows_px
    lng_grid, lat_grid = np.meshgrid(lngs, lats)

    # Pre-index both ellipsoids
    index_data = {}
    for ellipsoid in ELLIPSOIDS:
        t0 = time.perf_counter()
        cell_ids = index_raster_healpix(lat_grid, lng_grid, depth, ellipsoid)
        index_time = time.perf_counter() - t0
        unique_cells = np.unique(cell_ids)
        num_cells = len(unique_cells)
        print(f"\nIndexing [{ellipsoid:6s}]: {index_time:.4f}s → {num_cells} unique cells")

        t0 = time.perf_counter()
        preindexed = np.zeros((num_cells, max_layers), dtype=np.float32)
        for i in range(max_layers):
            preindexed[:, i] = aggregate_to_cells(rasters[i], cell_ids, unique_cells)
        preindex_time = time.perf_counter() - t0
        print(f"  Pre-indexing {max_layers} layers: {preindex_time:.3f}s")

        index_data[ellipsoid] = {
            "cell_ids": cell_ids,
            "unique_cells": unique_cells,
            "num_cells": num_cells,
            "index_time": index_time,
            "preindexed": preindexed,
        }

    # Pixel agreement between sphere and WGS84
    cs = index_data["sphere"]["cell_ids"]
    ce = index_data["WGS84"]["cell_ids"]
    pct_diff = 100 * np.sum(cs != ce) / len(cs)
    print(f"\nPixel cell assignment difference (sphere vs WGS84): {pct_diff:.1f}%")

    # Classification loop
    print("\n--- CLASSIFICATION ---")
    results = []
    for n in cfg["num_layers_list"]:
        print(f"\n{n} layers:")
        data = rasters[:n]
        row = {"num_layers": n}

        # Pure raster baseline
        t0 = time.perf_counter()
        sum_vals = (data * 10).astype(int).sum(axis=0)
        np.vectorize(classify_value)(sum_vals)
        row["raster_total"] = time.perf_counter() - t0
        print(f"  Raster:         {row['raster_total']:.4f}s")

        for ellipsoid in ELLIPSOIDS:
            pre = index_data[ellipsoid]["preindexed"]
            t0 = time.perf_counter()
            sums = (pre[:, :n] * 10).astype(int).sum(axis=1)
            np.array([classify_value(int(v)) for v in sums])
            elapsed = time.perf_counter() - t0
            row[f"healpix_{ellipsoid.lower()}_total"] = elapsed
            print(f"  HEALPix/{ellipsoid:6s}: {elapsed:.4f}s  ({index_data[ellipsoid]['num_cells']} cells)")

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_dir / "raster_benchmark_healpix_geo.csv", index=False)

    # Save indexing metadata
    meta = {
        "depth": depth,
        "bbox": cfg["bbox"],
        "raster_size": list(cfg["raster_size"]),
        "pixel_cell_assignment_diff_pct": float(pct_diff),
        "ellipsoids": {
            e: {
                "num_cells": int(index_data[e]["num_cells"]),
                "index_time_s": float(index_data[e]["index_time"]),
            }
            for e in ELLIPSOIDS
        },
    }
    with open(output_dir / "indexing_healpix_geo.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved: raster_benchmark_healpix_geo.csv, indexing_healpix_geo.json")
    return df


# =============================================================================
# Ellipsoid Geodetic Analysis
# =============================================================================

def run_ellipsoid_analysis(config: Dict, output_dir: Path) -> Dict:
    """
    Quantify the geodetic impact of sphere vs WGS84 across latitude bands.
    This is the core scientific contribution of the healpix-geo extension.

    For each test region (equatorial → arctic), we compute:
    - % of raster pixels assigned to different cells
    - Jaccard similarity of polygon coverage
    - Number of unique cells (sphere vs WGS84)
    """
    print("\n" + "=" * 70)
    print("ELLIPSOID ANALYSIS — sphere vs WGS84 across latitude bands")
    print("=" * 70)

    cfg = config["ellipsoid_analysis"]
    depth = cfg["healpix_depth"]
    raster_size = cfg["raster_size"]
    rng = np.random.default_rng(config["random_seed"])

    analysis_results = {}

    for region_name, bbox in cfg["regions"].items():
        minx, miny, maxx, maxy = bbox
        rows_px, cols_px = raster_size
        center_lat = (miny + maxy) / 2
        center_lon = (minx + maxx) / 2

        lngs = np.linspace(minx, maxx, cols_px)
        lats = np.linspace(miny, maxy, rows_px)
        lng_g, lat_g = np.meshgrid(lngs, lats)

        cs = healpix_nested.lonlat_to_healpix(
            lng_g.ravel(), lat_g.ravel(), depth, ellipsoid="sphere"
        )
        ce = healpix_nested.lonlat_to_healpix(
            lng_g.ravel(), lat_g.ravel(), depth, ellipsoid="WGS84"
        )

        n_diff = int(np.sum(cs != ce))
        pct_diff = 100 * n_diff / len(cs)
        n_cells_s = len(np.unique(cs))
        n_cells_e = len(np.unique(ce))

        # Polygon coverage (simple bbox polygon)
        polygon_verts = np.array([
            [minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]
        ])
        try:
            pids_s, _, _ = healpix_nested.polygon_coverage(
                polygon_verts, depth, ellipsoid="sphere", flat=True
            )
            pids_e, _, _ = healpix_nested.polygon_coverage(
                polygon_verts, depth, ellipsoid="WGS84", flat=True
            )
            set_s, set_e = set(pids_s.tolist()), set(pids_e.tolist())
            jaccard = len(set_s & set_e) / len(set_s | set_e) if (set_s | set_e) else 1.0
            cells_only_sphere = len(set_s - set_e)
            cells_only_wgs84 = len(set_e - set_s)
        except Exception:
            jaccard = np.nan
            cells_only_sphere = cells_only_wgs84 = -1

        res = {
            "region": region_name,
            "bbox": list(bbox),
            "center_lat": center_lat,
            "center_lon": center_lon,
            "healpix_depth": depth,
            "raster_pixels": int(rows_px * cols_px),
            "pixels_different_cell": n_diff,
            "pixels_different_pct": round(pct_diff, 2),
            "unique_cells_sphere": n_cells_s,
            "unique_cells_wgs84": n_cells_e,
            "polygon_jaccard_similarity": round(jaccard, 4) if not np.isnan(jaccard) else None,
            "polygon_cells_only_sphere": cells_only_sphere,
            "polygon_cells_only_wgs84": cells_only_wgs84,
        }
        analysis_results[region_name] = res

        print(f"\n{region_name:14s} (center lat {center_lat:+.0f}°):")
        print(f"  Pixel assignment diff:   {pct_diff:.1f}%  ({n_diff}/{rows_px*cols_px})")
        print(f"  Unique cells — sphere: {n_cells_s}, WGS84: {n_cells_e}")
        if not np.isnan(jaccard):
            print(f"  Polygon Jaccard:         {jaccard:.4f}  "
                  f"(+{cells_only_wgs84} WGS84-only, -{cells_only_sphere} sphere-only cells)")

    # Save
    with open(output_dir / "ellipsoid_analysis.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    print(f"\nSaved: ellipsoid_analysis.json")
    return analysis_results


# =============================================================================
# Plotting
# =============================================================================

def plot_results(vector_df: pd.DataFrame, raster_df: pd.DataFrame,
                  ellipsoid_analysis: Dict, output_dir: Path):
    """Generate comparison plots: benchmarks + ellipsoid analysis."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available — skipping plots")
        return

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "HEALPix-geo Replication of Law & Ardo (2024)\n"
        "sphere vs. WGS84 Ellipsoid  |  DOI: 10.1080/20964471.2024.2429847",
        fontsize=13, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    COLORS = {"sphere": "#5B8DB8", "WGS84": "#E07B39", "vector": "#888888", "raster": "#A0C878"}

    # --- Vector timing ---
    ax = fig.add_subplot(gs[0, 0])
    if not vector_df.empty:
        valid = vector_df[vector_df.get('vector_success', True)]
        if not valid.empty and 'vector_total' in valid.columns:
            ax.loglog(valid['num_layers'], valid['vector_total'], 's--',
                      color=COLORS["vector"], label='Vector overlay', linewidth=2)
        ax.loglog(vector_df['num_layers'], vector_df['healpix_sphere_total'], 'o-',
                  color=COLORS["sphere"], label='HEALPix/sphere', linewidth=2)
        ax.loglog(vector_df['num_layers'], vector_df['healpix_wgs84_total'], 'D-',
                  color=COLORS["WGS84"], label='HEALPix/WGS84', linewidth=2)
    ax.set_xlabel('Number of layers')
    ax.set_ylabel('Time (s)')
    ax.set_title('Vector Benchmark')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Vector speedup ---
    ax = fig.add_subplot(gs[0, 1])
    if not vector_df.empty and 'vector_total' in vector_df.columns:
        valid = vector_df.dropna(subset=['vector_total'])
        if not valid.empty:
            x = np.arange(len(valid))
            w = 0.35
            sp_s = valid['vector_total'] / valid['healpix_sphere_total']
            sp_e = valid['vector_total'] / valid['healpix_wgs84_total']
            ax.bar(x - w/2, sp_s, w, label='sphere', color=COLORS["sphere"], alpha=0.85)
            ax.bar(x + w/2, sp_e, w, label='WGS84', color=COLORS["WGS84"], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(valid['num_layers'].astype(str))
            ax.axhline(y=1, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Number of layers')
    ax.set_ylabel('Speedup vs vector')
    ax.set_title('Vector Speedup')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # --- Raster timing ---
    ax = fig.add_subplot(gs[0, 2])
    if not raster_df.empty:
        ax.loglog(raster_df['num_layers'], raster_df['raster_total'], 's--',
                  color=COLORS["raster"], label='Raster (numpy)', linewidth=2)
        ax.loglog(raster_df['num_layers'], raster_df['healpix_sphere_total'], 'o-',
                  color=COLORS["sphere"], label='HEALPix/sphere', linewidth=2)
        ax.loglog(raster_df['num_layers'], raster_df['healpix_wgs84_total'], 'D-',
                  color=COLORS["WGS84"], label='HEALPix/WGS84', linewidth=2)
    ax.set_xlabel('Number of layers')
    ax.set_ylabel('Time (s)')
    ax.set_title('Raster Benchmark')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Ellipsoid: pixel diff by latitude ---
    ax = fig.add_subplot(gs[1, 0:2])
    if ellipsoid_analysis:
        regions = list(ellipsoid_analysis.values())
        lats = [r["center_lat"] for r in regions]
        diffs = [r["pixels_different_pct"] for r in regions]
        names = [r["region"].replace("_", "\n") for r in regions]
        bars = ax.bar(names, diffs, color=COLORS["WGS84"], alpha=0.85, edgecolor='white')
        for bar, d in zip(bars, diffs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{d:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_ylabel('Pixels assigned to\ndifferent cell (%)', fontsize=10)
        ax.set_title(f'Sphere vs WGS84 Indexing Difference by Region\n(HEALPix depth={CONFIG["ellipsoid_analysis"]["healpix_depth"]})',
                     fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis='y')
        # Add center latitude as secondary label
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(len(lats)))
        ax2.set_xticklabels([f'{l:+.0f}°' for l in lats], fontsize=8, color='gray')
        ax2.set_xlabel('Center latitude', fontsize=8, color='gray')

    # --- Ellipsoid: Jaccard similarity ---
    ax = fig.add_subplot(gs[1, 2])
    if ellipsoid_analysis:
        regions = list(ellipsoid_analysis.values())
        lats = [r["center_lat"] for r in regions]
        jacc = [r["polygon_jaccard_similarity"] for r in regions if r["polygon_jaccard_similarity"] is not None]
        lat_j = [r["center_lat"] for r in regions if r["polygon_jaccard_similarity"] is not None]
        if jacc:
            ax.scatter(lat_j, jacc, s=80, zorder=5, color=COLORS["sphere"])
            ax.plot(lat_j, jacc, '-', color=COLORS["sphere"], alpha=0.6)
            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
            ax.set_xlim(-15, 90)
            ax.set_ylim(0.97, 1.005)
    ax.set_xlabel('Center latitude (°)')
    ax.set_ylabel('Jaccard similarity')
    ax.set_title('Polygon Coverage\nSimilarity (sphere vs WGS84)')
    ax.grid(True, alpha=0.3)

    plt.savefig(output_dir / "benchmark_healpix_geo.png", dpi=150, bbox_inches='tight')
    plt.savefig(output_dir / "benchmark_healpix_geo.pdf", bbox_inches='tight')
    print(f"\nPlots saved: benchmark_healpix_geo.png / .pdf")
    plt.close()


# =============================================================================
# Summary
# =============================================================================

def generate_summary(vector_df: pd.DataFrame, raster_df: pd.DataFrame,
                      ellipsoid_analysis: Dict, output_dir: Path) -> Dict:
    """Generate JSON + console summary."""

    summary = {
        "paper": {"doi": "10.1080/20964471.2024.2429847", "original_dggs": "H3"},
        "replication": {
            "library": "healpix-geo",
            "version": HEALPIX_GEO_VERSION,
            "ellipsoids": ELLIPSOIDS,
        },
        "results": {},
    }

    # Vector
    if not vector_df.empty and 'vector_total' in vector_df.columns:
        valid = vector_df.dropna(subset=['vector_total'])
        if not valid.empty:
            for ell in ["sphere", "wgs84"]:
                col = f"healpix_{ell}_total"
                if col in valid.columns:
                    sp = valid['vector_total'] / valid[col]
                    summary["results"][f"vector_{ell}"] = {
                        "speedup_min": f"{sp.min():.1f}x",
                        "speedup_max": f"{sp.max():.1f}x",
                        "validated": bool(sp.max() > 1.0),
                    }

    # Raster
    if not raster_df.empty:
        for ell in ["sphere", "wgs84"]:
            col = f"healpix_{ell}_total"
            if col in raster_df.columns:
                ratio = raster_df['raster_total'].mean() / raster_df[col].mean()
                summary["results"][f"raster_{ell}"] = {
                    "raster_healpix_ratio": f"{ratio:.2f}x",
                    "validated": bool(0.1 < ratio < 10),
                }

    # Ellipsoid analysis
    if ellipsoid_analysis:
        summary["results"]["ellipsoid_analysis"] = {
            r: {
                "center_lat": v["center_lat"],
                "pixels_different_pct": v["pixels_different_pct"],
                "polygon_jaccard": v["polygon_jaccard_similarity"],
            }
            for r, v in ellipsoid_analysis.items()
        }

    with open(output_dir / "summary_healpix_geo.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Console output
    print("\n" + "=" * 70)
    print("SUMMARY — healpix-geo (sphere vs WGS84)")
    print("=" * 70)
    if "vector_sphere" in summary["results"]:
        v = summary["results"]["vector_sphere"]
        print(f"  Vector / sphere : speedup {v['speedup_min']} – {v['speedup_max']}  "
              f"{'✓' if v['validated'] else '✗'}")
    if "vector_wgs84" in summary["results"]:
        v = summary["results"]["vector_wgs84"]
        print(f"  Vector / WGS84  : speedup {v['speedup_min']} – {v['speedup_max']}  "
              f"{'✓' if v['validated'] else '✗'}")
    if "raster_sphere" in summary["results"]:
        r = summary["results"]["raster_sphere"]
        print(f"  Raster / sphere : ratio {r['raster_healpix_ratio']}  "
              f"{'✓' if r['validated'] else '✗'}")
    if "raster_wgs84" in summary["results"]:
        r = summary["results"]["raster_wgs84"]
        print(f"  Raster / WGS84  : ratio {r['raster_healpix_ratio']}  "
              f"{'✓' if r['validated'] else '✗'}")

    print("\n  Ellipsoid pixel-assignment difference:")
    for name, v in ellipsoid_analysis.items():
        print(f"    {name:14s} (lat {v['center_lat']:+.0f}°): "
              f"{v['pixels_different_pct']:5.1f}% pixels differ  "
              f"Jaccard={v['polygon_jaccard_similarity']}")

    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HEALPix-geo Replication Study — sphere vs WGS84",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Replication of: Law & Ardo (2024)
DOI: 10.1080/20964471.2024.2429847

Extension: compares sphere vs WGS84 ellipsoid using healpix-geo.
Particularly relevant for high-latitude EO data (Copernicus / ESA).

Examples:
  python run_healpix_geo_replication.py --all
  python run_healpix_geo_replication.py --skip-vector --raster-layers 10,50,100
  python run_healpix_geo_replication.py --ellipsoid-only
        """,
    )
    parser.add_argument("--all", action="store_true",
                        help="Run all benchmarks + ellipsoid analysis")
    parser.add_argument("--ellipsoid-only", action="store_true",
                        help="Run only the ellipsoid geodetic analysis")
    parser.add_argument("--output", "-o", default="results_healpix_geo",
                        help="Output directory (default: results_healpix_geo)")
    parser.add_argument("--vector-layers", type=str,
                        help="Comma-separated layer counts for vector benchmark")
    parser.add_argument("--raster-layers", type=str,
                        help="Comma-separated layer counts for raster benchmark")
    parser.add_argument("--healpix-depth", type=int,
                        help="HEALPix depth for both benchmarks (default: 9)")
    parser.add_argument("--skip-vector", action="store_true")
    parser.add_argument("--skip-raster", action="store_true")
    parser.add_argument("--skip-ellipsoid-analysis", action="store_true")
    parser.add_argument("--random-seed", type=int)
    args = parser.parse_args()

    # Apply overrides
    if args.vector_layers:
        CONFIG["vector"]["num_layers_list"] = [int(x) for x in args.vector_layers.split(",")]
    if args.raster_layers:
        CONFIG["raster"]["num_layers_list"] = [int(x) for x in args.raster_layers.split(",")]
    if args.healpix_depth:
        CONFIG["vector"]["healpix_depth"] = args.healpix_depth
        CONFIG["raster"]["healpix_depth"] = args.healpix_depth
        CONFIG["ellipsoid_analysis"]["healpix_depth"] = args.healpix_depth
    if args.random_seed:
        CONFIG["random_seed"] = args.random_seed

    # Environment variable overrides (Docker compatibility)
    for env_var, cfg_path in [
        ("VECTOR_LAYERS", ("vector", "num_layers_list")),
        ("RASTER_LAYERS", ("raster", "num_layers_list")),
        ("HEALPIX_DEPTH",  None),
        ("RANDOM_SEED",    None),
    ]:
        val = os.environ.get(env_var)
        if val and cfg_path:
            a, b = cfg_path
            CONFIG[a][b] = [int(x) for x in val.split(",")]
        elif val and env_var == "HEALPIX_DEPTH":
            CONFIG["vector"]["healpix_depth"] = int(val)
            CONFIG["raster"]["healpix_depth"] = int(val)
        elif val and env_var == "RANDOM_SEED":
            CONFIG["random_seed"] = int(val)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Header
    print("=" * 70)
    print("HEALPIX-GEO REPLICATION STUDY — sphere vs WGS84 ellipsoid")
    print("=" * 70)
    print(f"Version:         {CODE_VERSION}")
    print(f"healpix-geo:     {HEALPIX_GEO_VERSION}")
    print(f"Output:          {output_dir}")
    print(f"Random seed:     {CONFIG['random_seed']}")
    print(f"Ellipsoids:      {ELLIPSOIDS}")
    print(f"HEALPix depth:   vector={CONFIG['vector']['healpix_depth']}, "
          f"raster={CONFIG['raster']['healpix_depth']}, "
          f"ellipsoid_analysis={CONFIG['ellipsoid_analysis']['healpix_depth']}")
    print(f"Vector layers:   {CONFIG['vector']['num_layers_list']}")
    print(f"Raster layers:   {CONFIG['raster']['num_layers_list']}")
    print(f"\nhealpix-geo:  {'✅ Available' if HAS_HEALPIX_GEO else '❌ NOT INSTALLED — exiting'}")

    if not HAS_HEALPIX_GEO:
        print("\nInstall with: pip install healpix-geo")
        sys.exit(1)

    with open(output_dir / "system_info.json", 'w') as f:
        json.dump(get_system_info(), f, indent=2)

    # Run benchmarks
    vector_df = pd.DataFrame()
    raster_df = pd.DataFrame()
    ellipsoid_results = {}

    if args.ellipsoid_only:
        ellipsoid_results = run_ellipsoid_analysis(CONFIG, output_dir)
    else:
        if not args.skip_vector:
            vector_df = run_vector_benchmark(CONFIG, output_dir)
        elif (output_dir / "vector_benchmark_healpix_geo.csv").exists():
            vector_df = pd.read_csv(output_dir / "vector_benchmark_healpix_geo.csv")

        if not args.skip_raster:
            raster_df = run_raster_benchmark(CONFIG, output_dir)
        elif (output_dir / "raster_benchmark_healpix_geo.csv").exists():
            raster_df = pd.read_csv(output_dir / "raster_benchmark_healpix_geo.csv")

        if not args.skip_ellipsoid_analysis:
            ellipsoid_results = run_ellipsoid_analysis(CONFIG, output_dir)

    # Plots & summary
    if not vector_df.empty or not raster_df.empty or ellipsoid_results:
        plot_results(vector_df, raster_df, ellipsoid_results, output_dir)
        generate_summary(vector_df, raster_df, ellipsoid_results, output_dir)

    print(f"\n📁 Results saved to: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        print(f"   {f.name}")


if __name__ == "__main__":
    main()
