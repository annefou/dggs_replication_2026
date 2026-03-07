# DGGS Benchmark Replication Environment

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18339339.png)](https://doi.org/10.5281/zenodo.18339339)
![Build Docker Image](https://github.com/annefou/dggs_replication_2026/actions/workflows/docker-build.yml/badge.svg)
![Run Replication](https://github.com/annefou/dggs_replication_2026/actions/workflows/run-replication.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Replication study** of the benchmarks from Law & Ardo (2024), extended in v3.0.0 with HEALPix benchmarks using the `healpix-geo` library (sphere and WGS84 ellipsoid).

This repository provides a **reproducible environment** for replicating the benchmarks from:

> Law, R.M. & Ardo, J. (2024). "Using a discrete global grid system for a scalable, interoperable, and reproducible system of land-use mapping"
> *Big Earth Data*, DOI: [10.1080/20964471.2024.2429847](https://doi.org/10.1080/20964471.2024.2429847)

**Original benchmark code:** [dggsBenchmarks v1.1.1](https://github.com/manaakiwhenua/dggsBenchmarks/releases/tag/v1.1.1)

---

## Key Findings

### v2.0.0 — H3 / xdggs replication

Our replication validates the paper's central claims:

| Benchmark | Paper Claim | Our Result | Status |
|-----------|-------------|------------|--------|
| **Vector (Figure 6)** | DGGS >> Vector performance | DGGS 2.5x faster at 20 layers, grows with scale | Validated |
| **Vector scaling** | Vector fails at ~500 layers | Feature count explodes exponentially | Validated |
| **Raster (Figure 7)** | DGGS ≈ Raster performance | Equivalent within 2x | Validated |

#### Vector Benchmark Scaling (H3, depth 9)

| Layers | DGGS Time | Vector Time | Vector Features | Speedup |
|--------|-----------|-------------|-----------------|---------|
| 5 | 2.6s | 0.2s | 53 | 0.1x |
| 10 | 4.9s | 1.3s | 616 | 0.3x |
| 15 | 7.4s | 5.9s | 1,737 | 0.8x |
| **20** | **10.1s** | **25.0s** | **3,362** | **2.5x** |

**Key insight:** DGGS scales linearly O(n), while vector overlay creates exponentially more features with each layer. The crossover point occurs around 15–20 layers.

### v3.0.0 — HEALPix / healpix-geo extension (new)

All three DGGS implementations (H3, HEALPix/sphere, HEALPix/WGS84) validate the paper's claims with consistent results:

| Method | Max speedup vs vector | Crossover point |
|--------|----------------------|-----------------|
| H3 (sphere) | ~5,800× | ~5 layers |
| HEALPix / sphere | ~5,691× | ~5 layers |
| HEALPix / WGS84 | ~5,603× | ~5 layers |

> All benchmarks use **HEALPix depth 9** (~1 km² cells), chosen to match H3 resolution 9 for a like-for-like comparison.

#### Sphere vs WGS84 Ellipsoid Indexing Difference (HEALPix depth 9)

A key new finding in v3.0.0 is the impact of the reference surface on cell assignment:

| Region | Center latitude | Pixels in different cell | Jaccard similarity |
|--------|----------------|--------------------------|-------------------|
| Equatorial | 0° | 27% | 0.9951 |
| Mid-latitude (Mediterranean) | +48° | **98%** | 0.9843 |
| High-latitude (Scandinavia) | +62° | **91%** | 0.9868 |
| Arctic | +78° | 53% | 0.9908 |

**Key insight:** For European EO data (Copernicus/Sentinel, 45–65°N), sphere-based HEALPix indexing assigns almost every pixel to the wrong cell. WGS84 indexing via `healpix-geo` is strongly recommended for production workflows.

---

## Purpose

The original benchmark code ([dggsBenchmarks v1.1.1](https://github.com/manaakiwhenua/dggsBenchmarks/releases/tag/v1.1.1)) does not include a containerized or fully reproducible environment. This replication study provides:

1. **Docker container** with all dependencies pinned
2. **Synthetic data generation** scripts (deterministic with seeded RNG)
3. **Benchmark scripts** that measure the same metrics as the paper
4. **Comparison analysis** to verify replication success
5. **Cross-DGGS comparison** (H3 vs HEALPix/sphere vs HEALPix/WGS84) *(new in v3.0.0)*
6. **GitHub Actions** for automated CI/CD and continuous verification
7. **Zenodo DOI** for persistent, citable archival
8. **Documentation** of the replication process

---

## Scripts

| Script | Description |
|--------|-------------|
| `run_replication.py` | H3 reproduction + xdggs replication (v2.0.0) |
| `run_healpix_replication.py` | HEALPix benchmarks via cdshealpix (v2.0.0) |
| `run_healpix_geo_replication.py` | HEALPix benchmarks via healpix-geo, sphere + WGS84 *(new in v3.0.0)* |
| `run_comparison.py` | Cross-DGGS unified comparison (reads all result CSVs) *(new in v3.0.0)* |

---

## Methodology

### Vector Benchmark (Figure 6)

Following the paper's Section 3.2.1 methodology:

1. **Data Generation**: Random points → Voronoi polygons (`scipy.spatial.Voronoi`)
2. **Values**: Each polygon assigned 0 or 1 randomly
3. **Dissolve**: Polygons dissolved by value before overlay (as per paper)
4. **Traditional Method**: Unary union (spatial overlay) of all dissolved layers
5. **DGGS Method**: **Polyfill** polygons to H3/HEALPix cells → join on cell ID
6. **Classification**: 7 functions (prime, perfect, triangular, square, pentagonal, hexagonal, Fibonacci) → 7-bit class

> **Note on Polyfill**: The paper explicitly states: *"A polygon filling algorithm is implemented through the H3 Python bindings, which we used through H3-Pandas, where it is termed 'polyfilling'."* This fills entire polygons with H3 cells, not just centroids.

### Raster Benchmark (Figure 7)

1. **Data Generation**: Spatially-correlated rasters (Gaussian smoothing)
2. **Traditional Method**: NumPy array stacking and classification
3. **DGGS Method**: Index raster cells to H3/HEALPix → aggregate → classify
4. **Replication**: Also includes xdggs vectorized indexing comparison

### Ellipsoid Analysis (new in v3.0.0)

HEALPix benchmarks are run twice — once with `ellipsoid='sphere'` and once with `ellipsoid='WGS84'` — using `healpix-geo` v0.0.11. The ellipsoid analysis measures the percentage of pixels assigned to a different cell depending on the reference surface, across four latitude bands (equatorial, mid-latitude, high-latitude, arctic).

### Reproduction vs Replication

| Term | Definition | Implementation |
|------|------------|----------------|
| **Reproduction** | Same methodology, same tools | H3 library + Pandas (as in paper) |
| **Replication** | Same methodology, different tools | xdggs for vectorized H3 indexing (v2.0.0); healpix-geo for HEALPix sphere+WGS84 (v3.0.0) |

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull pre-built image from GitHub Container Registry
docker pull ghcr.io/annefou/dggs_replication_2026:latest

# Run all benchmarks (H3 + HEALPix/healpix-geo + comparison)
docker run -v $(pwd)/results:/app/results ghcr.io/annefou/dggs_replication_2026:latest

# Or build locally
docker build -t dggs-benchmark-replication .
docker run -v $(pwd)/results:/app/results dggs-benchmark-replication
```

### Option 2: Local Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# H3 replication
python run_replication.py --all --output results_h3

# HEALPix/healpix-geo replication (sphere + WGS84)
python run_healpix_geo_replication.py --all --output results_healpix_geo

# Cross-DGGS comparison
python run_comparison.py \
    --h3 results_h3 \
    --healpix-geo results_healpix_geo \
    --output results_comparison
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_LAYERS` | `5,10,20,50,100` | Comma-separated layer counts for vector benchmark |
| `RASTER_LAYERS` | `10,50,100,500` | Comma-separated layer counts for raster benchmark |
| `H3_RESOLUTION` | `9` | H3 resolution for raster benchmark |
| `VECTOR_H3_RESOLUTION` | `9` | H3 resolution for vector polyfill (paper used 14) |
| `HEALPIX_DEPTH` | `9` | HEALPix depth for all HEALPix benchmarks *(new in v3.0.0)* |
| `POINTS_PER_LAYER` | `30` | Points per Voronoi layer |
| `RANDOM_SEED` | `42` | Random seed for reproducibility |

### CLI Arguments

```bash
# H3 replication
python run_replication.py --all
python run_replication.py --skip-vector
python run_replication.py --skip-raster
python run_replication.py --vector-layers 5,10,20
python run_replication.py --raster-layers 10,50
python run_replication.py --output my_results

# HEALPix/healpix-geo replication (new in v3.0.0)
python run_healpix_geo_replication.py --all
python run_healpix_geo_replication.py --ellipsoid-only
python run_healpix_geo_replication.py --vector-layers 5,10,20,50 --healpix-depth 9

# Cross-DGGS comparison (new in v3.0.0)
python run_comparison.py \
    --h3 results_h3 \
    --healpix results_healpix \        # optional
    --healpix-geo results_healpix_geo \
    --output results_comparison
```

### Note on H3 Resolution and HEALPix Depth

The paper used H3 resolution 14 for vector polyfill, which creates billions of cells for realistic polygons. We default to resolution/depth 9 for practical CI runs:

| Resolution/Depth | Approx. cell area | Cells per 0.1° × 0.1° polygon |
|-----------------|-------------------|-------------------------------|
| H3 resolution 9 | ~0.1 km² | ~1,500 |
| HEALPix depth 9 | ~0.013 deg² (~1 km²) | comparable |
| H3 resolution 14 | ~0.0006 km² | ~26,000,000 |

To match the paper exactly for H3:
```bash
docker run -e VECTOR_H3_RESOLUTION=14 ...
```

---

## Resource Requirements

| Benchmark Mode | Vector Layers | Est. RAM | Est. Time |
|---------------|---------------|----------|-----------|
| `quick-test` | 5, 10 | ~2 GB | ~2 min |
| `ci-test` | 5–100 | ~4 GB | ~10 min |
| `full` | 10–1000 | ~16+ GB | ~1 hour |

**GitHub Actions runners have ~7GB RAM**, so `ci-test` mode is used by default.

### Running with Different Modes

```bash
# CI test (default - fits in GitHub Actions)
docker run -v $(pwd)/results:/app/results \
    -e VECTOR_LAYERS=5,10,20,50,100 \
    ghcr.io/annefou/dggs_replication_2026:latest

# Full benchmark (paper values - requires ~16GB+ RAM)
docker run -v $(pwd)/results:/app/results \
    -e VECTOR_LAYERS=10,20,50,100,200,500,1000 \
    -e RASTER_LAYERS=10,50,100,500,1000,5000,10000 \
    -e VECTOR_H3_RESOLUTION=14 \
    ghcr.io/annefou/dggs_replication_2026:latest
```

### Reproducing / Replicating the original paper

```bash
python run_replication.py --output results_more \
    --vector-layers "5,10,20,50" \
    --raster-layers "10,50,100,500,1000,5000,10000"
```

> Note: Larger configurations cannot be executed with GitHub Actions due to memory constraints.

---

## GitHub Actions: Automated Replication

### Triggering a Replication Run

#### Via GitHub UI

1. Go to **Actions** → **Run Replication**
2. Click **Run workflow**
3. Choose options:
   - `benchmark_type`: `ci-test`, `quick-test`, or `full`
   - `random_seed`: For reproducibility (default: 42)

#### Via GitHub CLI

```bash
# Run CI test (default)
gh workflow run run-replication.yml -f benchmark_type=ci-test

# Run quick test
gh workflow run run-replication.yml -f benchmark_type=quick-test
```

### Scheduled Runs

The replication runs automatically every Sunday at 00:00 UTC to continuously verify the results remain reproducible.

### Viewing Results

- **Workflow summary**: Shows comparison with paper claims
- **Artifacts**: Download full results (CSV, JSON, plots)
- **Releases**: Benchmark results attached to releases

---

## Output Files

After running the replication, you'll find:

```
results/
├── system_info.json                        # Hardware/software environment
├── vector_benchmark.csv                    # H3 vector benchmark timings
├── raster_benchmark.csv                    # H3 raster benchmark timings
├── indexing_benchmark.json                 # H3 vs xdggs comparison
├── summary.json                            # Structured results for CI
├── benchmark_unified.png                   # H3 benchmark plots (PNG)
├── benchmark_unified.pdf                   # H3 benchmark plots (PDF)
│
├── vector_benchmark_healpix_geo.csv        # HEALPix sphere+WGS84 vector timings (v3.0.0)
├── raster_benchmark_healpix_geo.csv        # HEALPix sphere+WGS84 raster timings (v3.0.0)
├── ellipsoid_analysis.json                 # Sphere vs WGS84 indexing difference (v3.0.0)
├── summary_healpix_geo.json                # HEALPix structured summary (v3.0.0)
├── benchmark_healpix_geo.png               # HEALPix benchmark plots (v3.0.0)
│
├── comparison_table.csv                    # Cross-DGGS unified table (v3.0.0)
├── comparison_summary.json                 # Cross-DGGS structured summary (v3.0.0)
├── comparison.png                          # Cross-DGGS comparison plot (v3.0.0)
└── comparison.pdf                          # Cross-DGGS comparison plot (v3.0.0)
```

---

## Zenodo Integration

This repository is linked to Zenodo for persistent archival and DOI assignment.

### Citing This Replication

```bibtex
@software{dggs_replication_2026,
  author       = {Fouilloux, Anne},
  title        = {DGGS Benchmark Replication Environment},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18339339},
  url          = {https://doi.org/10.5281/zenodo.18339339}
}
```

---

## Differences from Original

| Aspect | Original | This Replication |
|--------|----------|------------------|
| Environment | None provided | Docker + pip |
| Dependencies | Unpinned | All versions pinned |
| H3 Resolution | 14 | 9 (default), configurable |
| Polyfill | H3-Pandas | H3 v4 `polygon_to_cells()` |
| Reproducibility | Seeded RNG | Same + containerized |
| HEALPix support | None | healpix-geo (sphere + WGS84) *(v3.0.0)* |
| Cross-DGGS comparison | None | H3 vs HEALPix/sphere vs HEALPix/WGS84 *(v3.0.0)* |

---

## Citation

If you use this replication environment, please cite both:

```bibtex
@article{law2024dggs,
  title={Using a discrete global grid system for a scalable, interoperable, 
         and reproducible system of land-use mapping},
  author={Law, Richard M and Ardo, James},
  journal={Big Earth Data},
  volume={9},
  number={1},
  pages={29--46},
  year={2024},
  doi={10.1080/20964471.2024.2429847}
}

@software{dggs_replication_2026,
  title={DGGS Benchmark Replication Environment},
  author={Fouilloux, Anne},
  year={2026},
  url={https://github.com/annefou/dggs_replication_2026},
  note={Replication of Law \& Ardo (2024)}
}
```

---

## License

This replication code is released under the MIT License.
The original benchmark code is subject to its own license terms.

---

## Contact

- **Replication author:** Anne Fouilloux (ORCID: [0000-0002-1784-2920](https://orcid.org/0000-0002-1784-2920))
- **Original paper authors:**
  - Richard M. Law (ORCID: [0000-0002-7400-2530](https://orcid.org/0000-0002-7400-2530))
  - James Ardo (ORCID: [0009-0008-1201-9733](https://orcid.org/0009-0008-1201-9733))

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **3.0.0** | **2026-03-07** | **Added HEALPix benchmarks via healpix-geo (sphere + WGS84); cross-DGGS comparison script; ellipsoid indexing analysis** |
| 2.0.0 | 2026-01-21 | Updated methodology: polyfill + dissolve matching paper |
| 1.0.0 | 2026-01-17 | Initial replication environment |

---

## Acknowledgments

- Original research by Richard M. Law and James Ardo at Manaaki Whenua – Landcare Research
- H3 library by Uber Technologies
- xdggs library for vectorized DGGS operations
- healpix-geo library for WGS84-aware HEALPix indexing *(new in v3.0.0)*
- This replication follows the framework from the [Replication Handbook](https://forrt.org/replication_handbook/)
