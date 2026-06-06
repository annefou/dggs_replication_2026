# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Shared helpers
#
# This module is imported by the result notebooks. It locates the repository
# root (so the notebooks run both from the repo root in CI and from the
# `notebooks/` directory locally) and provides small loaders for the committed
# result artifacts.
#
# **Important:** these notebooks do **not** re-run the benchmarks. The
# benchmarks are timing-sensitive and live in the `run_*.py` scripts. The book
# loads the already-committed CSV/JSON artifacts under `results_*/` so the build
# is fast, deterministic, and the reported wall-clock numbers stay meaningful.

# %%
import json
from pathlib import Path

import pandas as pd


def find_root(start: Path | None = None) -> Path:
    """Return the repository root (the directory containing ``results_h3/``)."""
    start = start or Path.cwd()
    for p in [start, *start.parents]:
        if (p / "results_h3").exists() and (p / "results_comparison").exists():
            return p
    raise FileNotFoundError(
        "Could not locate the repository root (no results_h3/ found). "
        "Run from the repo root or the notebooks/ directory."
    )


ROOT = find_root()


def load_csv(rel: str) -> pd.DataFrame:
    """Load a committed result CSV by path relative to the repo root."""
    return pd.read_csv(ROOT / rel)


def load_json(rel: str) -> dict:
    """Load a committed result JSON by path relative to the repo root."""
    with open(ROOT / rel) as fh:
        return json.load(fh)
