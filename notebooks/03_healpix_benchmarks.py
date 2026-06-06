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
# # 03 — HEALPix benchmarks (v3.0.0 extension)
#
# Extends the replication beyond H3 to **HEALPix**, using the `healpix-geo`
# library, on two geometries:
#
# * **sphere** — HEALPix on a perfect sphere;
# * **WGS84** — HEALPix on the WGS84 ellipsoid.
#
# This lets us ask two questions the original paper did not: does the DGGS
# speed-up hold for a second grid system, and does accounting for the
# ellipsoid change which cells a region maps to?

# %%
import matplotlib.pyplot as plt

from _helpers import load_csv, load_json

# %% [markdown]
# ## HEALPix vector benchmark (sphere vs WGS84)

# %%
geo = load_csv("results_healpix_geo/vector_benchmark_healpix_geo.csv")
geo[
    [
        "num_layers",
        "vector_total",
        "healpix_sphere_total",
        "healpix_wgs84_total",
    ]
]

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(geo["num_layers"], geo["vector_total"], "s-", label="Vector overlay")
ax.plot(geo["num_layers"], geo["healpix_sphere_total"], "o-", label="HEALPix (sphere)")
ax.plot(geo["num_layers"], geo["healpix_wgs84_total"], "^-", label="HEALPix (WGS84)")
ax.set_xlabel("Number of layers")
ax.set_ylabel("Total time (s)")
ax.set_yscale("log")
ax.set_title("HEALPix vector benchmark — sphere vs WGS84 vs vector overlay")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# Sphere and WGS84 HEALPix have essentially identical timing — the ellipsoid
# correction does not cost performance. Both beat vector overlay by the same
# orders of magnitude seen for H3.

# %% [markdown]
# ## Does the ellipsoid change the cells?
#
# Timing is one thing; **correctness** is another. `ellipsoid_analysis.json`
# records, for four latitude bands, how many raster pixels are assigned to a
# *different* HEALPix cell when the WGS84 ellipsoid is used instead of a sphere.

# %%
ell = load_json("results_healpix_geo/ellipsoid_analysis.json")

rows = []
for region, d in ell.items():
    rows.append(
        {
            "region": region,
            "center_lat": d["center_lat"],
            "pixels_different_pct": d["pixels_different_pct"],
            "polygon_jaccard": d["polygon_jaccard_similarity"],
        }
    )

import pandas as pd

ell_df = pd.DataFrame(rows)
ell_df

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(ell_df["region"], ell_df["pixels_different_pct"], color="#c44e52")
ax.set_ylabel("Pixels assigned to a different cell (%)")
ax.set_title("Sphere vs WGS84: cell reassignment by latitude band")
for i, v in enumerate(ell_df["pixels_different_pct"]):
    ax.text(i, v + 1, f"{v:.1f}%", ha="center")
ax.set_ylim(0, 110)
fig.tight_layout()
plt.show()

# %% [markdown]
# ```{important}
# The ellipsoid matters. At mid- and high latitudes, **most** raster pixels map
# to a different HEALPix cell on WGS84 than on a sphere (up to 98% at ~47°N),
# even though the polygon-level Jaccard similarity stays high (≈0.98–0.99).
# Treating a real-world ellipsoidal Earth as a sphere is cheap to compute but
# changes which cell a location belongs to — a correctness concern for any
# land-use mapping built on the sphere assumption.
# ```
