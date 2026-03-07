#!/usr/bin/env python3
"""
DGGS Cross-Implementation Comparison

Reads results produced by the three benchmark scripts and produces a
unified cross-DGGS comparison. Run each benchmark script first:

  python run_replication.py          --output results_h3
  python run_healpix_replication.py  --output results_healpix
  python run_healpix_geo_replication.py --output results_healpix_geo

Then run this script:

  python run_comparison.py \\
      --h3       results_h3 \\
      --healpix  results_healpix \\
      --healpix-geo results_healpix_geo \\
      --output   results_comparison

Scientific question answered:
  Do all DGGS implementations validate Law & Ardo (2024)?
  Does the sphere/WGS84 choice affect performance or crossover point?

Author: Anne Fouilloux
Date: 2026-03-07
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

CODE_VERSION = "2026-03-07-comparison-v1"

# Colours consistent across all plots
COLORS = {
    "h3":             "#E07B39",   # orange
    "healpix_sphere": "#5B8DB8",   # blue
    "healpix_wgs84":  "#8B5BA8",   # purple
    "vector":         "#AAAAAA",   # grey
    "raster":         "#A0C878",   # green
}
LABELS = {
    "h3":             "H3 (sphere)",
    "healpix_sphere": "HEALPix / sphere",
    "healpix_wgs84":  "HEALPix / WGS84",
    "vector":         "Vector overlay",
    "raster":         "Raster (numpy)",
}


# =============================================================================
# Loaders — each normalises its source CSV into a common schema
# =============================================================================

def load_h3(results_dir: Path) -> dict:
    """
    Load results from run_replication.py (H3).

    Vector CSV columns:
        num_layers, dggs_total_time, vector_total_time, vector_success, ...
    Raster CSV columns:
        num_layers, raster_total, dggs_preindex_total, ...
    """
    out = {}
    vec_path = results_dir / "vector_benchmark.csv"
    if vec_path.exists():
        df = pd.read_csv(vec_path)
        out["vector"] = pd.DataFrame({
            "num_layers":  df["num_layers"],
            "dggs_total":  df["dggs_total_time"],
            "vector_total": df["vector_total_time"],
            "vector_success": df["vector_success"],
        })

    ras_path = results_dir / "raster_benchmark.csv"
    if ras_path.exists():
        df = pd.read_csv(ras_path)
        out["raster"] = pd.DataFrame({
            "num_layers":  df["num_layers"],
            "dggs_total":  df["dggs_preindex_total"],   # pre-indexed is the fair comparison
            "raster_total": df["raster_total"],
        })
    return out


def load_healpix(results_dir: Path) -> dict:
    """
    Load results from run_healpix_replication.py (cdshealpix).

    Vector CSV columns:
        num_layers, healpix_total, healpix_cells, vector_total, vector_success
    Raster CSV columns:
        num_layers, raster_total, healpix_total
    """
    out = {}
    vec_path = results_dir / "vector_benchmark_healpix.csv"
    if vec_path.exists():
        df = pd.read_csv(vec_path)
        out["vector"] = pd.DataFrame({
            "num_layers":    df["num_layers"],
            "dggs_total":    df["healpix_total"],
            "vector_total":  df["vector_total"],
            "vector_success": df["vector_success"],
        })

    ras_path = results_dir / "raster_benchmark_healpix.csv"
    if ras_path.exists():
        df = pd.read_csv(ras_path)
        out["raster"] = pd.DataFrame({
            "num_layers":   df["num_layers"],
            "dggs_total":   df["healpix_total"],
            "raster_total": df["raster_total"],
        })
    return out


def load_healpix_geo(results_dir: Path) -> dict:
    """
    Load results from run_healpix_geo_replication.py (healpix-geo).

    Vector CSV columns:
        num_layers, healpix_sphere_total, healpix_wgs84_total,
        vector_total, vector_success, ...
    Raster CSV columns:
        num_layers, raster_total, healpix_sphere_total, healpix_wgs84_total
    """
    out = {}
    vec_path = results_dir / "vector_benchmark_healpix_geo.csv"
    if vec_path.exists():
        df = pd.read_csv(vec_path)
        # Two DGGS methods from same script
        out["vector_sphere"] = pd.DataFrame({
            "num_layers":    df["num_layers"],
            "dggs_total":    df["healpix_sphere_total"],
            "vector_total":  df["vector_total"],
            "vector_success": df["vector_success"],
        })
        out["vector_wgs84"] = pd.DataFrame({
            "num_layers":    df["num_layers"],
            "dggs_total":    df["healpix_wgs84_total"],
            "vector_total":  df["vector_total"],
            "vector_success": df["vector_success"],
        })

    ras_path = results_dir / "raster_benchmark_healpix_geo.csv"
    if ras_path.exists():
        df = pd.read_csv(ras_path)
        out["raster_sphere"] = pd.DataFrame({
            "num_layers":   df["num_layers"],
            "dggs_total":   df["healpix_sphere_total"],
            "raster_total": df["raster_total"],
        })
        out["raster_wgs84"] = pd.DataFrame({
            "num_layers":   df["num_layers"],
            "dggs_total":   df["healpix_wgs84_total"],
            "raster_total": df["raster_total"],
        })

    # Ellipsoid analysis
    ea_path = results_dir / "ellipsoid_analysis.json"
    if ea_path.exists():
        with open(ea_path) as f:
            out["ellipsoid_analysis"] = json.load(f)

    return out


# =============================================================================
# Crossover analysis
# =============================================================================

def find_crossover(layers: np.ndarray, dggs_times: np.ndarray,
                    vector_times: np.ndarray) -> Optional[float]:
    """
    Find the number of layers at which DGGS becomes faster than vector overlay.
    Returns None if DGGS is always faster or always slower in the tested range.
    """
    speedups = vector_times / dggs_times
    # Find where speedup crosses 1.0 (DGGS becomes faster)
    for i in range(len(speedups) - 1):
        if speedups[i] < 1.0 and speedups[i + 1] >= 1.0:
            # Linear interpolation
            x0, x1 = layers[i], layers[i + 1]
            y0, y1 = speedups[i], speedups[i + 1]
            return float(x0 + (1.0 - y0) * (x1 - x0) / (y1 - y0))
        if speedups[i] >= 1.0:
            return float(layers[i])   # Already faster from the start of tested range
    return None


# =============================================================================
# Build unified comparison table
# =============================================================================

def build_comparison_table(h3: dict, healpix: dict, geo: dict) -> pd.DataFrame:
    """
    Merge all vector results into a single wide table indexed by num_layers.
    Columns: num_layers, vector_total, h3_total, healpix_sphere_total,
             healpix_wgs84_total, h3_speedup, healpix_sphere_speedup,
             healpix_wgs84_speedup
    """
    frames = {}
    if "vector" in h3:
        frames["h3"] = h3["vector"][["num_layers", "dggs_total", "vector_total"]]
    if "vector" in healpix:
        frames["healpix_sphere_cdsh"] = healpix["vector"][["num_layers", "dggs_total"]]
    if "vector_sphere" in geo:
        frames["healpix_sphere"] = geo["vector_sphere"][["num_layers", "dggs_total"]]
    if "vector_wgs84" in geo:
        frames["healpix_wgs84"] = geo["vector_wgs84"][["num_layers", "dggs_total"]]

    if not frames:
        return pd.DataFrame()

    # Start from H3 if available, else first available
    base_key = "h3" if "h3" in frames else next(iter(frames))
    result = frames[base_key].rename(columns={"dggs_total": "h3_total"})

    for key, df in frames.items():
        if key == base_key:
            continue
        col = f"{key}_total"
        result = result.merge(
            df.rename(columns={"dggs_total": col}),
            on="num_layers", how="outer"
        )

    result = result.sort_values("num_layers").reset_index(drop=True)

    # Speedup columns
    for col in ["h3_total", "healpix_sphere_total", "healpix_wgs84_total",
                "healpix_sphere_cdsh_total"]:
        sp_col = col.replace("_total", "_speedup")
        if col in result.columns and "vector_total" in result.columns:
            result[sp_col] = result["vector_total"] / result[col]

    return result


# =============================================================================
# Plotting
# =============================================================================

def plot_comparison(h3: dict, healpix: dict, geo: dict,
                     table: pd.DataFrame, output_dir: Path):
    """Produce 6-panel comparison figure."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available — skipping plots")
        return

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Cross-DGGS Comparison — Replication of Law & Ardo (2024)\n"
        "H3 vs HEALPix (sphere) vs HEALPix (WGS84) | "
        "DOI: 10.1080/20964471.2024.2429847",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Vector timing (log-log) ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    plotted_vector = False
    for method, data_key, color in [
        ("h3",             "vector",        COLORS["h3"]),
        ("healpix_sphere", "vector",        COLORS["healpix_sphere"]),
        ("healpix_wgs84",  "vector_wgs84",  COLORS["healpix_wgs84"]),
    ]:
        src = {"h3": h3, "healpix_sphere": healpix,
               "healpix_wgs84": geo}.get(
                   "h3" if method == "h3" else
                   "healpix_sphere" if method == "healpix_sphere" else "healpix_wgs84"
               )
        key = {"h3": "vector", "healpix_sphere": "vector",
               "healpix_wgs84": "vector_wgs84"}[method]
        # Use geo for sphere too if healpix not loaded
        if method == "healpix_sphere" and "vector" not in healpix and "vector_sphere" in geo:
            src, key = geo, "vector_sphere"
        if src and key in src:
            df = src[key]
            valid = df[df["vector_success"] == True] if "vector_success" in df.columns else df
            ax.loglog(valid["num_layers"], valid["dggs_total"], "o-",
                      color=color, label=LABELS[method], linewidth=2, markersize=6)
            if not plotted_vector and "vector_total" in valid.columns:
                ax.loglog(valid["num_layers"], valid["vector_total"], "s--",
                          color=COLORS["vector"], label=LABELS["vector"],
                          linewidth=1.5, markersize=5, alpha=0.7)
                plotted_vector = True
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Time (s)")
    ax.set_title("Vector Benchmark — Timing")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Vector speedup vs vector overlay ─────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for col, color, label in [
        ("h3_speedup",             COLORS["h3"],             LABELS["h3"]),
        ("healpix_sphere_speedup", COLORS["healpix_sphere"], LABELS["healpix_sphere"]),
        ("healpix_wgs84_speedup",  COLORS["healpix_wgs84"],  LABELS["healpix_wgs84"]),
    ]:
        if col in table.columns:
            valid = table.dropna(subset=[col])
            ax.semilogy(valid["num_layers"], valid[col], "o-",
                        color=color, label=label, linewidth=2, markersize=6)
    ax.axhline(y=1, color="gray", linestyle="--", linewidth=1, label="Break-even")
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Speedup vs vector overlay")
    ax.set_title("Vector Speedup")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Raster timing ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    plotted_raster = False
    for method, src, key, color in [
        ("h3",             h3,      "raster",        COLORS["h3"]),
        ("healpix_sphere", healpix, "raster",        COLORS["healpix_sphere"]),
        ("healpix_wgs84",  geo,     "raster_wgs84",  COLORS["healpix_wgs84"]),
    ]:
        # fallback: use geo sphere if healpix not loaded
        if method == "healpix_sphere" and "raster" not in healpix and "raster_sphere" in geo:
            src, key = geo, "raster_sphere"
        if key in src:
            df = src[key]
            ax.loglog(df["num_layers"], df["dggs_total"], "o-",
                      color=color, label=LABELS[method], linewidth=2, markersize=6)
            if not plotted_raster and "raster_total" in df.columns:
                ax.loglog(df["num_layers"], df["raster_total"], "s--",
                          color=COLORS["raster"], label=LABELS["raster"],
                          linewidth=1.5, markersize=5, alpha=0.7)
                plotted_raster = True
    ax.set_xlabel("Number of layers")
    ax.set_ylabel("Time (s)")
    ax.set_title("Raster Benchmark — Timing")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Crossover comparison bar chart ───────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    crossovers = {}
    for method, src, key, color in [
        ("h3",             h3,      "vector",        COLORS["h3"]),
        ("healpix_sphere", healpix, "vector",        COLORS["healpix_sphere"]),
        ("healpix_wgs84",  geo,     "vector_wgs84",  COLORS["healpix_wgs84"]),
    ]:
        if method == "healpix_sphere" and "vector" not in healpix and "vector_sphere" in geo:
            src, key = geo, "vector_sphere"
        if key in src:
            df = src[key].dropna(subset=["dggs_total", "vector_total"])
            if not df.empty:
                co = find_crossover(
                    df["num_layers"].values,
                    df["dggs_total"].values,
                    df["vector_total"].values,
                )
                crossovers[method] = co
    if crossovers:
        names = [LABELS[m] for m in crossovers]
        vals = [v if v is not None else 0 for v in crossovers.values()]
        colors = [COLORS[m] for m in crossovers]
        bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        for bar, val in zip(bars, crossovers.values()):
            label = f"~{val:.0f}" if val else "N/A"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    label, ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Layers at crossover point")
    ax.set_title("Crossover: DGGS becomes\nfaster than vector overlay")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 5: Speedup at max tested layers ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    max_speedups = {}
    for col, method in [
        ("h3_speedup",             "h3"),
        ("healpix_sphere_speedup", "healpix_sphere"),
        ("healpix_wgs84_speedup",  "healpix_wgs84"),
    ]:
        if col in table.columns:
            valid = table.dropna(subset=[col])
            if not valid.empty:
                max_speedups[method] = valid[col].max()
    if max_speedups:
        names = [LABELS[m] for m in max_speedups]
        vals = list(max_speedups.values())
        colors = [COLORS[m] for m in max_speedups]
        bars = ax.bar(range(len(names)), vals, color=colors, alpha=0.85, edgecolor="white")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}x", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax.set_ylabel("Max speedup vs vector overlay")
    ax.set_title("Peak Speedup (all tested layers)")
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 6: Ellipsoid pixel-assignment difference ────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ea = geo.get("ellipsoid_analysis", {})
    if ea:
        regions = list(ea.values())
        names = [r["region"].replace("_", "\n") for r in regions]
        diffs = [r["pixels_different_pct"] for r in regions]
        lats  = [r["center_lat"] for r in regions]
        bars  = ax.bar(names, diffs, color=COLORS["healpix_wgs84"],
                       alpha=0.85, edgecolor="white")
        for bar, d in zip(bars, diffs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{d:.0f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Pixels in different cell (%)")
        ax.set_title("Sphere vs WGS84 Indexing\nDifference by Region")
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(len(lats)))
        ax2.set_xticklabels([f"{l:+.0f}°" for l in lats],
                             fontsize=8, color="gray")
        ax2.set_xlabel("Center latitude", fontsize=8, color="gray")
    else:
        ax.text(0.5, 0.5, "No ellipsoid analysis\ndata available",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        ax.set_title("Sphere vs WGS84 Indexing\nDifference by Region")
    ax.grid(True, alpha=0.3, axis="y")

    plt.savefig(output_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_dir / "comparison.pdf", bbox_inches="tight")
    print(f"Plots saved: comparison.png / .pdf")
    plt.close()


# =============================================================================
# Summary
# =============================================================================

def generate_summary(table: pd.DataFrame, h3: dict, healpix: dict,
                      geo: dict, output_dir: Path) -> dict:
    """Print summary table and save JSON."""

    summary = {
        "generated": datetime.now().isoformat(),
        "code_version": CODE_VERSION,
        "paper": {"doi": "10.1080/20964471.2024.2429847"},
        "methods": {},
    }

    print("\n" + "=" * 70)
    print("CROSS-DGGS COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<22} {'Max speedup':>12} {'Crossover':>12} {'Claim validated':>16}")
    print("-" * 64)

    for method, sp_col, src, key, color in [
        ("H3 (sphere)",       "h3_speedup",             h3,      "vector",       None),
        ("HEALPix/sphere",    "healpix_sphere_speedup",  healpix, "vector",       None),
        ("HEALPix/WGS84",     "healpix_wgs84_speedup",   geo,     "vector_wgs84", None),
    ]:
        # Fallback for healpix_sphere
        if method == "HEALPix/sphere" and "vector" not in healpix and "vector_sphere" in geo:
            src, key = geo, "vector_sphere"

        max_sp = "—"
        crossover = "—"
        validated = "—"

        if sp_col in table.columns:
            valid_sp = table.dropna(subset=[sp_col])
            if not valid_sp.empty:
                ms = valid_sp[sp_col].max()
                max_sp = f"{ms:.0f}x"
                validated = "✅" if ms > 1.0 else "❌"
                summary["methods"][method] = {"max_speedup": round(ms, 1)}

        if key in src:
            df = src[key].dropna(subset=["dggs_total", "vector_total"])
            if not df.empty:
                co = find_crossover(
                    df["num_layers"].values,
                    df["dggs_total"].values,
                    df["vector_total"].values,
                )
                crossover = f"~{co:.0f} layers" if co else "not reached"
                if method in summary["methods"]:
                    summary["methods"][method]["crossover_layers"] = co

        print(f"{method:<22} {max_sp:>12} {crossover:>12} {validated:>16}")

    # Ellipsoid impact
    ea = geo.get("ellipsoid_analysis", {})
    if ea:
        print("\n  Sphere vs WGS84 pixel-assignment difference:")
        print(f"  {'Region':<16} {'Center lat':>12} {'Pixels differ':>14} {'Jaccard':>10}")
        print("  " + "-" * 54)
        for name, v in ea.items():
            jac = v.get("polygon_jaccard_similarity")
            jac_str = f"{jac:.4f}" if jac is not None else "—"
            print(f"  {name:<16} {v['center_lat']:>+10.0f}°  "
                  f"{v['pixels_different_pct']:>11.1f}%  {jac_str:>10}")
        summary["ellipsoid_analysis"] = {
            name: {
                "center_lat": v["center_lat"],
                "pixels_different_pct": v["pixels_different_pct"],
                "jaccard": v.get("polygon_jaccard_similarity"),
            }
            for name, v in ea.items()
        }

    # Save
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save comparison table
    if not table.empty:
        table.to_csv(output_dir / "comparison_table.csv", index=False)
        print(f"\n  Comparison table → comparison_table.csv")

    print(f"\n  Full summary     → comparison_summary.json")
    return summary


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-DGGS comparison: H3 vs HEALPix/sphere vs HEALPix/WGS84",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Run the three benchmark scripts first, then compare:

  python run_replication.py            --output results_h3
  python run_healpix_replication.py    --output results_healpix
  python run_healpix_geo_replication.py --output results_healpix_geo
  python run_comparison.py \\
      --h3 results_h3 \\
      --healpix results_healpix \\
      --healpix-geo results_healpix_geo

Any subset of the three inputs is accepted — missing ones are silently skipped.
        """,
    )
    parser.add_argument("--h3", type=Path, default=None,
                        help="Directory with run_replication.py results")
    parser.add_argument("--healpix", type=Path, default=None,
                        help="Directory with run_healpix_replication.py results")
    parser.add_argument("--healpix-geo", type=Path, default=None,
                        help="Directory with run_healpix_geo_replication.py results")
    parser.add_argument("--output", "-o", type=Path, default=Path("results_comparison"),
                        help="Output directory (default: results_comparison)")
    args = parser.parse_args()

    # Require at least one input
    if not any([args.h3, args.healpix, args.healpix_geo]):
        parser.error("Provide at least one of --h3, --healpix, --healpix-geo")

    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CROSS-DGGS COMPARISON")
    print("=" * 70)
    print(f"Version: {CODE_VERSION}")
    print(f"Output:  {args.output}")

    # Load results
    h3_data, hp_data, geo_data = {}, {}, {}

    if args.h3:
        if not args.h3.exists():
            print(f"WARNING: --h3 path not found: {args.h3}")
        else:
            h3_data = load_h3(args.h3)
            print(f"H3            : loaded from {args.h3}  "
                  f"(vector={'vector' in h3_data}, raster={'raster' in h3_data})")

    if args.healpix:
        if not args.healpix.exists():
            print(f"WARNING: --healpix path not found: {args.healpix}")
        else:
            hp_data = load_healpix(args.healpix)
            print(f"HEALPix       : loaded from {args.healpix}  "
                  f"(vector={'vector' in hp_data}, raster={'raster' in hp_data})")

    if args.healpix_geo:
        if not args.healpix_geo.exists():
            print(f"WARNING: --healpix-geo path not found: {args.healpix_geo}")
        else:
            geo_data = load_healpix_geo(args.healpix_geo)
            print(f"HEALPix-geo   : loaded from {args.healpix_geo}  "
                  f"(sphere={'vector_sphere' in geo_data}, "
                  f"wgs84={'vector_wgs84' in geo_data}, "
                  f"ellipsoid_analysis={'ellipsoid_analysis' in geo_data})")

    # Build unified table
    table = build_comparison_table(h3_data, hp_data, geo_data)

    # Plot + summary
    plot_comparison(h3_data, hp_data, geo_data, table, args.output)
    generate_summary(table, h3_data, hp_data, geo_data, args.output)

    print(f"\n📁 Results saved to: {args.output}/")
    for f in sorted(args.output.iterdir()):
        print(f"   {f.name}")


if __name__ == "__main__":
    main()
