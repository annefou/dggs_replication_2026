# DGGS Benchmark Replication Environment

<!-- 
  BADGE SETUP INSTRUCTIONS:
  1. Go to your repo > Actions > select workflow > click "..." > "Create status badge"
  2. Copy the markdown and paste below
  3. Replace the placeholder badges below with your actual badges
-->

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
![Build Docker Image](https://github.com/annefou/dggs_replication_2026/actions/workflows/docker-build.yml/badge.svg)
![Run Replication](https://github.com/annefou/dggs_replication_2026/workflows/run-replication.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!--
  If the above badges don't work, generate them from GitHub:
  1. Go to: https://github.com/annefou/dggs_replication_2026/actions
  2. Click on a workflow (e.g., "Build Docker Image")
  3. Click the "..." button (top right, next to search)
  4. Click "Create status badge"
  5. Copy the Markdown
-->

> **Replication study** of the benchmarks from Law & Ardo (2024)

This repository provides a **reproducible environment** for replicating the benchmarks from:

> Law, R.M. & Ardo, J. (2024). "Using a discrete global grid system for a scalable, interoperable, and reproducible system of land-use mapping"
> *Big Earth Data*, DOI: [10.1080/20964471.2024.2429847](https://doi.org/10.1080/20964471.2024.2429847)

**Original benchmark code:** [dggsBenchmarks v1.1.1](https://github.com/manaakiwhenua/dggsBenchmarks/releases/tag/v1.1.1)

## Purpose

The original benchmark code ([dggsBenchmarks v1.1.1](https://github.com/manaakiwhenua/dggsBenchmarks/releases/tag/v1.1.1)) does not include a containerized or fully reproducible environment. This replication study provides:

1. **Docker container** with all dependencies pinned
2. **Synthetic data generation** scripts (deterministic with seeded RNG)
3. **Benchmark scripts** that measure the same metrics as the paper
4. **Comparison analysis** to verify replication success
5. **GitHub Actions** for automated CI/CD and continuous verification
6. **Zenodo DOI** for persistent, citable archival
7. **Documentation** of the replication process

## GitHub Actions: Automated Replication

This repository uses GitHub Actions to automatically build the Docker image and run the replication benchmarks.

### Triggering a Replication Run

#### Via GitHub UI

1. Go to **Actions** → **Run Replication**
2. Click **Run workflow**
3. Choose options:
   - `benchmark_type`: `all`, `vector`, `raster`, or `quick-test`
   - `random_seed`: For reproducibility (default: 42)
   - `upload_to_release`: Attach results to a release

#### Via GitHub CLI

```bash
# Run full replication
gh workflow run run-replication.yml -f benchmark_type=all

# Run quick test
gh workflow run run-replication.yml -f benchmark_type=quick-test

# Run with custom seed
gh workflow run run-replication.yml -f benchmark_type=all -f random_seed=12345
```

### Scheduled Runs

The replication runs automatically every Sunday at 00:00 UTC to continuously verify the results remain reproducible.

### Viewing Results

- **Workflow summary**: Shows comparison with paper claims
- **Artifacts**: Download full results (CSV, JSON, plots)
- **Releases**: Benchmark results attached to releases

## Zenodo Integration

This repository is linked to Zenodo for persistent archival and DOI assignment.

### Setup Zenodo (First Time)

1. Go to [Zenodo](https://zenodo.org/) and log in with GitHub
2. Navigate to [GitHub settings](https://zenodo.org/account/settings/github/)
3. Enable this repository for Zenodo
4. Create a release on GitHub
5. Zenodo automatically archives the release and assigns a DOI

### Citing This Replication

The `.zenodo.json` file contains metadata for proper citation. After the first release, update the DOI badge in this README.

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

## Resource Requirements

The benchmarks require significant memory, especially for large datasets:

| Benchmark Mode | Vector Layers | Polygons | Traditional Methods | DGGS Methods | Est. RAM |
|---------------|---------------|----------|---------------------|--------------|----------|
| `quick-test` | 5, 10 | 10 | All layers | All layers | ~2 GB |
| `ci-test` | 5-100 | 30 | ≤20 layers only | All layers | ~4 GB |
| `full` | 10-1000 | 100 | ≤500 layers | All layers | ~16+ GB |

**GitHub Actions runners have ~7GB RAM**, so `ci-test` mode limits traditional methods to small layer counts while running DGGS on the full range.

### How Comparison Works

The benchmark runs:
- **DGGS methods** on ALL layer counts (they scale well)
- **Traditional methods** only up to `MAX_TRADITIONAL_LAYERS` (they can OOM)

This still demonstrates the paper's key finding: DGGS scales linearly while traditional methods fail at higher layer counts.

Example CI output:
```
Benchmarking with 5 layers...
  Running traditional vector method...
  DGGS: 0.15s
  Vector: 2.34s      ← Traditional runs (5 ≤ 20)

Benchmarking with 10 layers...
  Running traditional vector method...
  DGGS: 0.28s
  Vector: 8.92s      ← Traditional runs (10 ≤ 20)

Benchmarking with 50 layers...
  Skipping traditional method (layers 50 > max 20)
  DGGS: 1.42s        ← Only DGGS runs (50 > 20)
```

### Running with Different Modes

```bash
# CI test (compare at small scale, DGGS on full range)
docker run -v $(pwd)/results:/app/results \
    -e VECTOR_LAYERS=5,10,20,50,100 \
    -e POLYGONS_PER_LAYER=30 \
    -e MAX_TRADITIONAL_LAYERS=20 \
    ghcr.io/annefou/dggs_replication_2026:latest

# Full benchmark (paper values - requires ~16GB+ RAM)
docker run -v $(pwd)/results:/app/results \
    ghcr.io/annefou/dggs_replication_2026:latest
```

## Quick Start

### Configuration

**`config.env`** is the **single source of truth** for all configuration:

```bash
# config.env - EDIT THIS FILE TO CHANGE SETTINGS
DGGS_BENCHMARKS_REPO=https://github.com/manaakiwhenua/dggsBenchmarks
DGGS_BENCHMARKS_VERSION=v1.1.1
PYTHON_VERSION=3.11
H3_RESOLUTION=9
RANDOM_SEED=42

# Dataset sizes (paper values by default)
VECTOR_LAYERS=10,20,50,100,200,500,1000
RASTER_LAYERS=10,50,100,500,1000,5000,10000
```

**Configuration priority** (highest to lowest):
1. CLI arguments (`--vector-layers`, `--raster-layers`, `--seed`)
2. Environment variables (`VECTOR_LAYERS`, `RASTER_LAYERS`, `RANDOM_SEED`)
3. `config.env` file

All components read from `config.env`:

| File | How it reads config.env |
|------|------------------------|
| `Dockerfile` | Sources at build time |
| `run_replication.py` | `load_config_env()` function |
| `Makefile` | `include config.env` |
| GitHub Actions | Sources in workflow steps |

**Python dependencies** are defined in `requirements.txt` (single source of truth):

| File | Purpose |
|------|---------|
| `requirements.txt` | All Python packages with pinned versions |
| `environment.yml` | Conda wrapper that references `requirements.txt` |

**To replicate a different version:**

1. Edit `config.env`:
   ```bash
   DGGS_BENCHMARKS_VERSION=v2.0.0
   ```

2. Rebuild:
   ```bash
   make docker-build
   ```

3. When releasing, also update static metadata:
   - `.zenodo.json` (for Zenodo DOI)
   - `CITATION.cff` (for GitHub citation)

### Option 1: Docker (Recommended)

The Docker image is automatically built and pushed to **GitHub Container Registry** (ghcr.io) on every push to main and on releases.

```bash
# Pull pre-built image from GitHub Container Registry
docker pull ghcr.io/annefou/dggs_replication_2026:latest

# Run replication
docker run -v $(pwd)/results:/app/results ghcr.io/annefou/dggs_replication_2026:latest

# Or build locally
docker build -t dggs-benchmark-replication .
docker run -v $(pwd)/results:/app/results dggs-benchmark-replication
```

**Available image tags:**

| Tag | Description |
|-----|-------------|
| `latest` | Latest build from main branch |
| `v1.0.0` | Specific release version |
| `sha-abc1234` | Specific commit |

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

### Option 3: Conda Environment

Conda is useful if you have trouble installing GDAL/GEOS system libraries.

```bash
# Create environment (uses requirements.txt for Python packages)
conda env create -f environment.yml
conda activate dggs-replication

# Run replication
python run_replication.py --all
```

### Dependency Files

| File | Purpose | Used by |
|------|---------|---------|
| `requirements.txt` | **Single source of truth** for Python packages | Docker, pip, conda |
| `environment.yml` | Conda wrapper + system libs (GDAL) | Conda only |

> **Note:** `environment.yml` references `requirements.txt` via `pip: -r requirements.txt`. 
> Only edit `requirements.txt` to change Python package versions.

## Usage

```bash
# Generate synthetic data only
python run_replication.py --generate-data

# Run vector benchmark (Figure 6)
python run_replication.py --vector

# Run raster benchmark (Figure 7)
python run_replication.py --raster

# Run all benchmarks and compare with paper
python run_replication.py --all

# Generate plots
python run_replication.py --plot

# Custom random seed
python run_replication.py --all --seed 12345

# Custom output directory
python run_replication.py --all --output my_results
```

## Expected Results

### Figure 6: Vector Input Benchmark

| Claim | Expected | Verification |
|-------|----------|--------------|
| DGGS >> Vector performance | DGGS should be orders of magnitude faster | Compare timing curves |
| Vector method fails | Should fail at ~500 layers due to memory | Check failure point |

### Figure 7: Raster Input Benchmark

| Claim | Expected | Verification |
|-------|----------|--------------|
| DGGS ≈ Raster performance | Roughly equivalent timing curves | Compare on log-log plot |

## Output Files

After running the replication, you'll find:

```
results/
├── system_info.json           # Hardware/software environment
├── vector_benchmark.csv       # Vector benchmark timings
├── raster_benchmark.csv       # Raster benchmark timings
├── comparison_with_paper.json # Automated comparison
├── benchmark_results.png      # Plots (cf. Figures 6 & 7)
└── benchmark_results.pdf      # Vector plots
```

## Replication vs Reproduction

This is a **reproduction study** according to the [Replication Handbook](https://doi.org/10.31222/osf.io/mbrxt):

| Aspect | Original | This Study |
|--------|----------|------------|
| Data | Synthetic (seeded RNG) | Same (regenerated) |
| Analysis | Benchmark scripts | Same methodology |
| Type | - | **Reproduction** |

## Differences from Original

1. **Environment**: We provide Docker/conda environments (original had none)
2. **Dependencies**: All packages pinned to specific versions
3. **Data generation**: Simplified NLM implementation (nlmpy optional)
4. **Vector benchmark**: Uses geopandas overlay (original may have used different method)
5. **Hardware**: Results will differ in absolute timing (patterns should match)

## Methodology Notes

### Synthetic Data Generation

- **Vector data**: Random polygons simulating Voronoi regions
- **Raster data**: Neutral Landscape Models (NLM) using mid-point displacement
- **Determinism**: All random generators seeded for reproducibility

### Benchmark Methodology

The paper describes:

> "We designed a benchmarking experiment to compare a land-use mapping workflow using either vector data, or data indexed to a DGGS."

Key operations measured:
- **DGGS method**: Indexing (convert to H3) + Classification (join on cell ID)
- **Vector method**: Spatial union/overlay operations
- **Raster method**: Warp (align grids) + Classification (stack and compute)

## Citation

If you use this replication environment, please cite:

```bibtex
@article{law2024dggs,
  title={Using a discrete global grid system for a scalable, interoperable, 
         and reproducible system of land-use mapping},
  author={Law, Richard M and Ardo, James},
  journal={Big Earth Data},
  year={2024},
  doi={10.1080/20964471.2024.2429847}
}

@software{dggs_replication_2026,
  title={DGGS Benchmark Replication Environment},
  author={[Your Name]},
  year={2026},
  url={[Your Repository URL]},
  note={Replication of Law \& Ardo (2024)}
}
```

## License

This replication code is released under the MIT License.
The original benchmark code is subject to its own license terms.

## Contact

- Replication author: Anne Fouilloux (ORCID: [0000-0002-1784-2920](https://orcid.org/0000-0002-1784-2920))
- Original paper authors: Richard M. Law (ORCID: [0000-0002-7400-2530](https://orcid.org/0000-0002-7400-2530)), James Ardo (ORCID: [0009-0008-1201-9733](https://orcid.org/0009-0008-1201-9733))

## Repository Setup Guide

If you're using this as a template for your own replication study:

### 1. Create GitHub Repository

```bash
# Clone this template
git clone https://github.com/annefou/dggs_replication_2026.git
cd dggs-benchmark-replication

# Or use GitHub's "Use this template" feature
```

### 2. Update Metadata

Edit these files with your information:
- `CITATION.cff`: Update author, repository URL
- `.zenodo.json`: Update creator, affiliation
- `README.md`: Update badges with your username

### 3. Enable GitHub Actions

1. Go to repository **Settings** → **Actions** → **General**
2. Enable "Read and write permissions" for GITHUB_TOKEN
3. Allow actions to create and approve pull requests

### 4. Enable GitHub Container Registry

1. Go to **Settings** → **Packages**
2. Ensure packages are enabled
3. First push will create the container package

### 5. Link to Zenodo

1. Go to [zenodo.org/account/settings/github/](https://zenodo.org/account/settings/github/)
2. Toggle ON for your repository
3. Create a release → Zenodo automatically archives

### 6. Update DOI Badge

After first Zenodo release:
1. Copy the DOI badge markdown from Zenodo
2. Replace the placeholder in README.md

### 7. Run First Replication

```bash
# Via GitHub Actions
gh workflow run run-replication.yml -f benchmark_type=all

# Or locally
make docker-run
```

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-17 | Initial replication environment |

## Acknowledgments

- Original research by Richard M. Law and James Ardo at Manaaki Whenua – Landcare Research
- H3 library by Uber Technologies
- This replication follows the framework from the [Replication Handbook](https://doi.org/10.31222/osf.io/mbrxt)
