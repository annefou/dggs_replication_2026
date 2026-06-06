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
# # 02 — H3 replication (v2.0.0)
#
# Reproduces Figures 6 and 7 of Law & Ardo (2024) using H3, comparing a DGGS
# workflow against the classic vector-overlay and raster workflows.

# %%
import matplotlib.pyplot as plt

from _helpers import load_csv, load_json

# %% [markdown]
# ## Vector benchmark (Figure 6)
#
# Claim: *DGGS provides orders-of-magnitude performance improvement over vector
# overlay as the number of layers grows.*

# %%
vec = load_csv("results_h3/vector_benchmark.csv")
vec[["num_layers", "dggs_total_time", "vector_total_time"]]

# %% [markdown]
# The vector-overlay cost explodes super-linearly with the number of layers,
# while the DGGS cost grows roughly linearly.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(vec["num_layers"], vec["dggs_total_time"], "o-", label="DGGS (H3)")
ax.plot(vec["num_layers"], vec["vector_total_time"], "s-", label="Vector overlay")
ax.set_xlabel("Number of layers")
ax.set_ylabel("Total time (s)")
ax.set_yscale("log")
ax.set_title("Vector benchmark — DGGS vs vector overlay (H3, depth 9)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
plt.show()

# %%
vec["speedup"] = vec["vector_total_time"] / vec["dggs_total_time"]
vec[["num_layers", "speedup"]].round(1)

# %% [markdown]
# ## Raster benchmark (Figure 7)
#
# Claim: *DGGS and raster methods show roughly equivalent performance.*

# %%
ras = load_csv("results_h3/raster_benchmark.csv")
ras[["num_layers", "raster_total", "dggs_xdggs_total", "dggs_preindex_total"]]

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(ras["num_layers"], ras["raster_total"], "o-", label="Raster")
ax.plot(ras["num_layers"], ras["dggs_xdggs_total"], "s-", label="DGGS (xdggs index)")
ax.plot(ras["num_layers"], ras["dggs_preindex_total"], "^-", label="DGGS (pre-indexed)")
ax.set_xlabel("Number of layers")
ax.set_ylabel("Total time (s)")
ax.set_title("Raster benchmark — DGGS vs raster (H3, depth 9)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Verdict

# %%
results = load_json("results_h3/summary.json")["results"]
for bench, r in results.items():
    validated = r.get("paper_claim_validated")
    mark = "validated" if validated else "not validated (equivalent, as expected)"
    print(f"{bench:<22}: {mark}")
    print(f"  → {r['conclusion']}")
