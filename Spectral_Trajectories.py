# -*- coding: utf-8 -*-
"""
Plot typical temporal trajectories of spectral activity (SpectralNorm)
for each EvolutionType.

FIXED VERSION:
- handles fractional years correctly
- robust to missing data
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================================================
# WORKING DIRECTORY
# ==================================================
os.chdir(r"d:\YandexDisk\Projects\MES_evolution\Spectrum_analysis\MES_Out_Points")

# ==================================================
# CONFIG
# ==================================================
PANEL_CSV = "spectral_panel.csv"
TYPES_CSV = "node_evolution_types.csv"
SEP = ";"

K_EIG_MAX = 20

PLOT_BANDS = True
BAND_Q_LOW = 0.10
BAND_Q_HIGH = 0.90

PLOT_INDIVIDUALS = True
N_REPRESENTATIVES = 5
MIN_YEARS_PER_NODE = 10

# ==================================================
# HELPERS
# ==================================================
def get_eig_cols(df, kmax=20):
    eig_cols = [c for c in df.columns if str(c).startswith("eig_")]
    def eig_num(c):
        try:
            return int(str(c).split("_")[1])
        except Exception:
            return 999
    return sorted(eig_cols, key=eig_num)[:kmax]


def compute_spectral_norm(df, eig_cols):
    for c in eig_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X = df[eig_cols].to_numpy(dtype=float)
    sn = np.sqrt(np.nansum(X * X, axis=1))
    sn[np.all(np.isnan(X), axis=1)] = np.nan
    df["SpectralNorm"] = sn
    return df


# ==================================================
# DATA PREPARATION
# ==================================================
def prepare_data():
    panel = pd.read_csv(PANEL_CSV, sep=SEP)
    types = pd.read_csv(TYPES_CSV, sep=SEP)

    # ---- ensure coordinate columns exist
    for df, name in [(panel, "panel"), (types, "types")]:
        if not {"x", "y"}.issubset(df.columns):
            raise ValueError(f"{name} must contain 'x' and 'y' columns")

    # ---- normalize coordinates (IMPORTANT)
    for df in (panel, types):
        df["x"] = pd.to_numeric(df["x"], errors="coerce").round(0).astype("Int64")
        df["y"] = pd.to_numeric(df["y"], errors="coerce").round(0).astype("Int64")

    # ---- year handling
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel = panel.dropna(subset=["year"])
    panel["year"] = panel["year"].round().astype(int)

    # ---- merge by spatial identity
    df = panel.merge(
        types[["x", "y", "EvolutionType"]],
        on=["x", "y"],
        how="inner"
    )

    # ---- SpectralNorm
    if "SpectralNorm" in df.columns:
        df["SpectralNorm"] = pd.to_numeric(df["SpectralNorm"], errors="coerce")

    if "SpectralNorm" not in df.columns or df["SpectralNorm"].notna().sum() == 0:
        eig_cols = get_eig_cols(df, K_EIG_MAX)
        if not eig_cols:
            raise ValueError("No SpectralNorm or eig_* columns found.")
        df = compute_spectral_norm(df, eig_cols)

    df = df.dropna(subset=["SpectralNorm", "EvolutionType"])
    df["EvolutionType"] = df["EvolutionType"].astype(int)

    # ---- diagnostics
    print("=== DATA CHECK ===")
    print("Rows:", len(df))
    print("Years:", df["year"].min(), "-", df["year"].max())
    print("Evolution types:", df["EvolutionType"].value_counts().sort_index().to_dict())
    print("==================")

    return df.sort_values(["EvolutionType", "x", "y", "year"])



# ==================================================
# COMPUTE TYPICAL CURVES
# ==================================================
def compute_typical_curves(df):
    def qlow(x): return np.nanquantile(x, BAND_Q_LOW)
    def qhigh(x): return np.nanquantile(x, BAND_Q_HIGH)

    return (
        df.groupby(["EvolutionType", "year"])["SpectralNorm"]
          .agg(mean="mean", qlow=qlow, qhigh=qhigh, n="count")
          .reset_index()
    )


def pick_representatives(df_type, all_years):
    piv = df_type.pivot(index="node_key", columns="year", values="SpectralNorm")
    piv = piv.loc[piv.notna().sum(axis=1) >= MIN_YEARS_PER_NODE]

    if piv.empty:
        return []

    piv = piv.reindex(columns=all_years)
    mean_traj = np.nanmean(piv.values, axis=0)

    dists = []
    for node, row in piv.iterrows():
        x = row.values
        mask = ~np.isnan(x) & ~np.isnan(mean_traj)
        if mask.sum() < MIN_YEARS_PER_NODE:
            continue
        d = np.linalg.norm(x[mask] - mean_traj[mask]) / np.sqrt(mask.sum())
        dists.append((d, node))

    dists.sort()
    return [n for _, n in dists[:N_REPRESENTATIVES]]


# ==================================================
# PLOTTING
# ==================================================
def plot_trajectories(df, stats):
    types = sorted(df["EvolutionType"].unique())
    years = sorted(df["year"].unique())

    # ---- all types in one plot
    plt.figure(figsize=(11, 6))
    for t in types:
        s = stats[stats["EvolutionType"] == t]
        if s.empty:
            continue
        plt.plot(s["year"], s["mean"], linewidth=2, label=f"Type {t}")
        if PLOT_BANDS:
            plt.fill_between(s["year"], s["qlow"], s["qhigh"], alpha=0.15)

    plt.xlabel("Year")
    plt.ylabel("Spectral activity (SpectralNorm)")
    plt.title("Typical trajectories of spectral activity by evolution type")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- per-type detailed plots
    if not PLOT_INDIVIDUALS:
        return

    for t in types:
        df_t = df[df["EvolutionType"] == t]
        s = stats[stats["EvolutionType"] == t]
        if s.empty:
            continue

        plt.figure(figsize=(11, 6))
        plt.plot(s["year"], s["mean"], linewidth=3, label="Mean trajectory")
        if PLOT_BANDS:
            plt.fill_between(s["year"], s["qlow"], s["qhigh"], alpha=0.15)

        reps = pick_representatives(df_t, years)
        for nk in reps:
            g = df_t[df_t["node_key"] == nk]
            plt.plot(g["year"], g["SpectralNorm"], linewidth=1, alpha=0.7)

        plt.xlabel("Year")
        plt.ylabel("Spectral activity (SpectralNorm)")
        plt.title(f"EvolutionType {t}: typical and representative trajectories")
        plt.legend()
        plt.tight_layout()
        plt.show()


# ==================================================
# MAIN
# ==================================================
def main():
    df = prepare_data()
    stats = compute_typical_curves(df)
    plot_trajectories(df, stats)


if __name__ == "__main__":
    main()
