# DGGS Benchmark Replication

A reproducible **replication study** of the benchmarks from Law & Ardo (2024),
*Using a discrete global grid system for a scalable, interoperable, and
reproducible system of land-use mapping* ([Big Earth Data,
10.1080/20964471.2024.2429847](https://doi.org/10.1080/20964471.2024.2429847)).

This book presents the results. It is built from the committed result artifacts
in `results_*/` — the benchmarks themselves live in the `run_*.py` scripts and
are documented in the [README](https://github.com/annefou/dggs_replication_2026).

## Contents

- **[01 — Overview](notebooks/01_overview.ipynb)** — paper claims, run
  environment, and benchmark configuration.
- **[02 — H3 replication](notebooks/02_h3_replication.ipynb)** — Figures 6 & 7:
  DGGS vs vector overlay and vs raster.
- **[03 — HEALPix benchmarks](notebooks/03_healpix_benchmarks.ipynb)** — the
  v3.0.0 extension: HEALPix on the sphere vs the WGS84 ellipsoid.
- **[04 — Cross-method comparison](notebooks/04_comparison.ipynb)** — all
  methods side by side.

## How to reproduce the benchmarks

The book loads pre-computed results. To regenerate them, run the benchmark
scripts (see the README and `Makefile`):

```bash
pip install -r requirements.txt
make            # or: python run_comparison.py
```

```{note}
Benchmark timings are hardware-dependent. The committed results were produced on
the machine recorded in `results_h3/system_info.json`; your numbers will differ
but the qualitative conclusions should hold.
```
