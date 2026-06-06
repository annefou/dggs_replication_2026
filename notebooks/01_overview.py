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
# # 01 — Overview
#
# This Jupyter Book presents the results of a **replication study** of the
# benchmarks from:
#
# > Law, R.M. & Ardo, J. (2024). *Using a discrete global grid system for a
# > scalable, interoperable, and reproducible system of land-use mapping.*
# > **Big Earth Data**, DOI:
# > [10.1080/20964471.2024.2429847](https://doi.org/10.1080/20964471.2024.2429847)
#
# Original benchmark code:
# [dggsBenchmarks v1.1.1](https://github.com/manaakiwhenua/dggsBenchmarks/releases/tag/v1.1.1).
#
# The replication has three layers:
#
# * **Reproduction** — same methodology, same tools (H3 + Polars).
# * **Replication** — same methodology, alternative tools (`xdggs`).
# * **Extension (v3.0.0)** — HEALPix benchmarks on the sphere and on the WGS84
#   ellipsoid via the `healpix-geo` library.
#
# ```{note}
# These notebooks **load committed results** from the `results_*/` directories
# rather than re-running the benchmarks. The benchmarks are timing-sensitive and
# live in the `run_*.py` scripts; re-running them on a shared CI runner would
# produce meaningless wall-clock numbers. See `notebooks/_helpers.py`.
# ```

# %%
from _helpers import ROOT, load_json

print(f"Repository root: {ROOT}")

# %% [markdown]
# ## Run environment
#
# The committed results record the exact environment they were produced in.

# %%
info = load_json("results_h3/system_info.json")

print(f"Code version : {info['code_version']}")
print(f"Timestamp    : {info['timestamp']}")
print(f"Python       : {info['python_version']}")
print(f"CPU count    : {info['system']['cpu_count']}")
print(f"Memory (GB)  : {info['system']['memory_gb']}")
print()
print("Pinned dependencies:")
for pkg, ver in info["dependencies"].items():
    print(f"  {pkg:<12} {ver}")

# %% [markdown]
# ## Benchmark configuration

# %%
cfg = info["configuration"]
for key, val in cfg.items():
    print(f"{key:<16}: {val}")

# %% [markdown]
# ## Paper claims under test
#
# The two central performance claims of Law & Ardo (2024):

# %%
summary = load_json("results_h3/summary.json")
for name, claim in summary["paper"]["claims"].items():
    print(f"- {name.capitalize()}: {claim}")
