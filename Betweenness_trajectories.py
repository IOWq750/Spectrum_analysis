# -*- coding: utf-8 -*-
"""
Plot typical temporal trajectories of betweenness centrality
for BC-based evolutionary types.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# PATHS
# ==================================================
os.chdir(r"d:\YandexDisk\Projects\MES_evolution\Spectrum_analysis\MES_Out_Points")

PANEL_CSV = "betweenness_panel.csv"
TYPES_CSV = "betweenness_evolution_types.csv"

SEP = ";"

# plot settings
BAND_Q_LOW = 0.10
BAND_Q_HIGH = 0.90
N_REPRESENTATIVES = 5
MIN_YEARS = 10

# ==================================================
# LOAD
# ==================================================
panel = pd.read_csv(PANEL_CSV, sep=SEP)
types = pd.read_csv(TYPES_CSV, sep=SEP)[["x", "y", "BC_EvolutionType"]]

# years
panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
panel = panel.dropna(subset=["year"])
panel["year"] = panel["year"].round().astype(int)

# coords
for df in (panel, types):
    df["x"] = pd.to_numeric(df["x"], errors="coerce").round(0).astype(int)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").round(0).astype(int)

# merge
df = panel.merge(types, on=["x", "y"], how="inner")
df = df.dropna(subset=["Betweenness", "BC_EvolutionType"])
df["BC_EvolutionType"] = df["BC_EvolutionType"].astype(int)

print("Rows:", len(df))
print("Types:", df["BC_EvolutionType"].value_counts().sort_index().to_dict())

# ==================================================
# AGGREGATE TYPICAL CURVES
# ==================================================
def qlow(x): return np.nanquantile(x, BAND_Q_LOW)
def qhigh(x): return np.nanquantile(x, BAND_Q_HIGH)

stats = (
    df.groupby(["BC_EvolutionType", "year"])["Betweenness"]
      .agg(mean="mean", qlow=qlow, qhigh=qhigh, n="count")
      .reset_index()
)

years_all = sorted(df["year"].unique())
types_all = sorted(df["BC_EvolutionType"].unique())

# ==================================================
# REPRESENTATIVE NODES
# ==================================================
def pick_representatives(df_type):
    piv = df_type.pivot(index=["x", "y"], columns="year", values="Betweenness")
    piv = piv.loc[piv.notna().sum(axis=1) >= MIN_YEARS]

    if piv.empty:
        return []

    piv = piv.reindex(columns=years_all)
    mean_traj = np.nanmean(piv.values, axis=0)

    dists = []
    for idx, row in piv.iterrows():
        x = row.values
        mask = ~np.isnan(x) & ~np.isnan(mean_traj)
        if mask.sum() < MIN_YEARS:
            continue
        d = np.linalg.norm(x[mask] - mean_traj[mask]) / np.sqrt(mask.sum())
        dists.append((d, idx))

    dists.sort()
    return [idx for _, idx in dists[:N_REPRESENTATIVES]]

# ==================================================
# PLOTTING
# ==================================================
# --- all types together
plt.figure(figsize=(11, 6))
for t in types_all:
    s = stats[stats["BC_EvolutionType"] == t]
    plt.plot(s["year"], s["mean"], linewidth=2, label=f"Type {t}")
    plt.fill_between(s["year"], s["qlow"], s["qhigh"], alpha=0.15)

plt.xlabel("Year")
plt.ylabel("Betweenness centrality (normalized)")
plt.title("Typical betweenness trajectories by evolutionary type")
plt.legend()
plt.tight_layout()
plt.show()

# --- per-type detailed plots
for t in types_all:
    df_t = df[df["BC_EvolutionType"] == t]
    s = stats[stats["BC_EvolutionType"] == t]

    plt.figure(figsize=(11, 6))
    plt.plot(s["year"], s["mean"], linewidth=3, label="Mean trajectory")
    plt.fill_between(s["year"], s["qlow"], s["qhigh"], alpha=0.15)

    reps = pick_representatives(df_t)
    for (x, y) in reps:
        g = df_t[(df_t["x"] == x) & (df_t["y"] == y)]
        plt.plot(g["year"], g["Betweenness"], linewidth=1, alpha=0.7)

    plt.xlabel("Year")
    plt.ylabel("Betweenness centrality (normalized)")
    plt.title(f"BC EvolutionType {t}")
    plt.legend()
    plt.tight_layout()
    plt.show()
