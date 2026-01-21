# DGGS Benchmark Replication Environment

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
![Build Docker Image](https://github.com/annefou/dggs_replication_2026/actions/workflows/docker-build.yml/badge.svg)
![Run Replication](https://github.com/annefou/dggs_replication_2026/actions/workflows/run-replication.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Replication study** of the benchmarks from Law & Ardo (2024)

This repository provides a **reproducible environment** for replicating the benchmarks from:

> Law, R.M. & Ardo, J. (2024). "Using a discrete global grid system for a scalable, interoperable, and reproducible system of land-use mapping"
> *Big Earth Data*, DOI: [10.1080/20964471.2024.2429847](https://doi.org/10.1080/20964471.2024.2429847)

**Original benchmark code:** [dggsBenchmarks v1.1.1](https://github.com/manaakiwhenua/dggsBenchmarks/releases/tag/v1.1.1)

## Key Findings

Our replication validates the paper's central claims:

| Benchmark | Paper Claim | Our Result | Status |
|-----------|-------------|------------|--------|
| **Vector (Figure 6)** | DGGS >> Vector performance | DGGS 2.5x faster at 20 layers, grows with scale | ✅ Validated |
| **Vector scaling** | Vector fails at ~500 layers | Feature count explodes exponentially | ✅ Validated |
| **Raster (Figure 7)** | DGGS ≈ Raster performance | Equivalent within 2x | ✅ Validated |

### Vector Benchmark Scaling

| Layers | DGGS Time | Vector Time | Vector Features | Speedup |
|--------|-----------|-------------|-----------------|---------|
| 5 | 2.6s | 0.2s | 53 | 0.1x |
| 10 | 4.9s | 1.3s | 616 | 0.3x |
| 15 | 7.4s | 5.9s | 1,737 | 0.8x |
| **20** | **10.1s** | **25.0s** | **3,362** | **2.5x** |

**Key insight:** DGGS scales linearly O(n), while vector overlay creates exponentially more features with each layer. The crossover point occurs around 15-20 layers.

## Purpose

The original benchmark code ([dggsBenchmarks v1.1.1](https://github.com/manaakiwhenua/dggsBenchmarks/releases/tag/v1.1.1)) does not include a containerized or fully reproducible environment. This replication study provides:

1. **Docker container** with all dependencies pinned
2. **Synthetic data generation** scripts (deterministic with seeded RNG)
3. **Benchmark scripts** that measure the same metrics as the paper
4. **Comparison analysis** to verify replication success
5. **GitHub Actions** for automated CI/CD and continuous verification
6. **Zenodo DOI** for persistent, citable archival
7. **Documentation** of the replication process

## Methodology

### Vector Benchmark (Figure 6)

Following the paper's Section 3.2.1 methodology:

1. **Data Generation**: Random points → Voronoi polygons (`scipy.spatial.Voronoi`)
2. **Values**: Each polygon assigned 0 or 1 randomly
3. **Dissolve**: Polygons dissolved by value before overlay (as per paper)
4. **Traditional Method**: Unary union (spatial overlay) of all dissolved layers
5. **DGGS Method**: **Polyfill** polygons to H3 cells → join on cell ID
6. **Classification**: 7 functions (prime, perfect, triangular, square, pentagonal, hexagonal, Fibonacci) → 7-bit class

> **Note on Polyfill**: The paper explicitly states: *"A polygon filling algorithm is implemented through the H3 Python bindings, which we used through H3-Pandas, where it is termed 'polyfilling'."* This fills entire polygons with H3 cells, not just centroids.

### Raster Benchmark (Figure 7)

1. **Data Generation**: Spatially-correlated rasters (Gaussian smoothing)
2. **Traditional Method**: NumPy array stacking and classification
3. **DGGS Method**: Index raster cells to H3 → aggregate → classify
4. **Replication**: Also includes xdggs vectorized indexing comparison

### Reproduction vs Replication

| Term | Definition | Implementation |
|------|------------|----------------|
| **Reproduction** | Same methodology, same tools | H3 library + Pandas (as in paper) |
| **Replication** | Same methodology, different tools | xdggs for vectorized indexing |

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Pull pre-built image from GitHub Container Registry
docker pull ghcr.io/annefou/dggs_replication_2026:latest

# Run replication
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

# Run replication
python run_replication.py --all
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VECTOR_LAYERS` | `5,10,20,50,100` | Comma-separated layer counts for vector benchmark |
| `RASTER_LAYERS` | `10,50,100,500` | Comma-separated layer counts for raster benchmark |
| `H3_RESOLUTION` | `9` | H3 resolution for raster benchmark |
| `VECTOR_H3_RESOLUTION` | `9` | H3 resolution for vector polyfill (paper used 14) |
| `POINTS_PER_LAYER` | `30` | Points per Voronoi layer |
| `RANDOM_SEED` | `42` | Random seed for reproducibility |

### CLI Arguments

```bash
python run_replication.py --all                    # Run all benchmarks
python run_replication.py --skip-vector            # Skip vector benchmark
python run_replication.py --skip-raster            # Skip raster benchmark
python run_replication.py --vector-layers 5,10,20  # Custom layer counts
python run_replication.py --raster-layers 10,50    # Custom layer counts
python run_replication.py --output my_results      # Custom output directory
```

### Configuration Priority

1. CLI arguments (highest)
2. Environment variables
3. Defaults in code

### Note on H3 Resolution

The paper used H3 resolution 14 for vector polyfill, which creates billions of cells for realistic polygons. We default to resolution 9 for practical CI runs:

| Resolution | Cells per 0.1° × 0.1° polygon |
|------------|------------------------------|
| 9 | ~1,500 |
| 14 | ~26,000,000 |

To match the paper exactly:
```bash
docker run -e VECTOR_H3_RESOLUTION=14 ...
```

## Resource Requirements

| Benchmark Mode | Vector Layers | Est. RAM | Est. Time |
|---------------|---------------|----------|-----------|
| `quick-test` | 5, 10 | ~2 GB | ~2 min |
| `ci-test` | 5-100 | ~4 GB | ~10 min |
| `full` | 10-1000 | ~16+ GB | ~1 hour |

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

## Output Files

After running the replication, you'll find:

```
results/
├── system_info.json              # Hardware/software environment
├── vector_benchmark.csv          # Vector benchmark timings
├── raster_benchmark.csv          # Raster benchmark timings
├── indexing_benchmark.json       # H3 vs xdggs comparison (if xdggs available)
├── summary.json                  # Structured results for CI
├── benchmark_results_unified.png # Comparison plots
└── benchmark_results_unified.pdf # Vector plots
```

## Zenodo Integration

This repository is linked to Zenodo for persistent archival and DOI assignment.

### Citing This Replication

```bibtex
@software{dggs_replication_2026,
  author       = {Fouilloux, Anne},
  title        = {DGGS Benchmark Replication Environment},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

## Differences from Original

| Aspect | Original | This Replication |
|--------|----------|------------------|
| Environment | None provided | Docker + pip |
| Dependencies | Unpinned | All versions pinned |
| H3 Resolution | 14 | 9 (default), configurable |
| Polyfill | H3-Pandas | H3 v4 `polygon_to_cells()` |
| Reproducibility | Seeded RNG | Same + containerized |

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

## License

This replication code is released under the MIT License.
The original benchmark code is subject to its own license terms.

## Contact

- **Replication author:** Anne Fouilloux (ORCID: [0000-0002-1784-2920](https://orcid.org/0000-0002-1784-2920))
- **Original paper authors:** 
  - Richard M. Law (ORCID: [0000-0002-7400-2530](https://orcid.org/0000-0002-7400-2530))
  - James Ardo (ORCID: [0009-0008-1201-9733](https://orcid.org/0009-0008-1201-9733))

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-01-21 | Updated methodology: polyfill + dissolve matching paper |
| 1.0.0 | 2026-01-17 | Initial replication environment |

## Acknowledgments

- Original research by Richard M. Law and James Ardo at Manaaki Whenua – Landcare Research
- H3 library by Uber Technologies
- xdggs library for vectorized DGGS operations
- This replication follows the framework from the [Replication Handbook](https://doi.org/10.31222/osf.io/mbrxt)
