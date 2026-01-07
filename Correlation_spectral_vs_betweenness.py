# -*- coding: utf-8 -*-
"""
Correlation analysis between spectral activity and betweenness centrality
at node-level and panel-level (robust version).
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# ==================================================
# PATHS
# ==================================================
os.chdir(r"d:\YandexDisk\Projects\MES_evolution\Spectrum_analysis\MES_Out_Points")

SEP = ";"

# node-level profiles
SPECTRAL_PROFILES = "node_evolution_types.csv"
BC_PROFILES = "betweenness_node_profiles.csv"

# panel-level
SPECTRAL_PANEL = "spectral_panel.csv"
BC_PANEL = "betweenness_panel.csv"

# optional: types
SPECTRAL_TYPES = "node_evolution_types.csv"
BC_TYPES = "betweenness_evolution_types.csv"

# ==================================================
# UTILITIES
# ==================================================
def corr_report(x, y, label):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10:
        return None

    s = spearmanr(x[mask], y[mask])
    p = pearsonr(x[mask], y[mask])

    return {
        "Metric": label,
        "Spearman_r": s.statistic,
        "Spearman_p": s.pvalue,
        "Pearson_r": p.statistic,
        "Pearson_p": p.pvalue,
        "N": int(mask.sum())
    }


def compute_spectral_norm_from_eigs(df, kmax=20):
    eig_cols = [c for c in df.columns if str(c).startswith("eig_")]
    if not eig_cols:
        raise ValueError("No SpectralNorm and no eig_* columns found.")

    # sort eig columns by index
    def idx(c):
        try:
            return int(c.split("_")[1])
        except Exception:
            return 999

    eig_cols = sorted(eig_cols, key=idx)[:kmax]

    X = df[eig_cols].apply(pd.to_numeric, errors="coerce").values
    sn = np.sqrt(np.nansum(X * X, axis=1))
    sn[np.all(np.isnan(X), axis=1)] = np.nan
    df["SpectralNorm"] = sn
    return df


# ==================================================
# 1️⃣ NODE-LEVEL CORRELATIONS
# ==================================================
print("\n=== NODE-LEVEL CORRELATIONS ===")

spec = pd.read_csv(SPECTRAL_PROFILES, sep=SEP)
bc = pd.read_csv(BC_PROFILES, sep=SEP)

nodes = spec.merge(bc, on=["x", "y"], how="inner")

results = []

results.append(
    corr_report(
        nodes["mean_SpectralNorm"].values,
        nodes["mean_BC"].values,
        "mean_SpectralNorm vs mean_BC"
    )
)

results.append(
    corr_report(
        nodes["trend_SpectralNorm"].values,
        nodes["trend_BC"].values,
        "trend_SpectralNorm vs trend_BC"
    )
)

if "BC_EuclideanNorm" in nodes.columns:
    results.append(
        corr_report(
            nodes["mean_SpectralNorm"].values,
            nodes["BC_EuclideanNorm"].values,
            "mean_SpectralNorm vs BC_EuclideanNorm"
        )
    )

df_node_corr = pd.DataFrame([r for r in results if r is not None])
print(df_node_corr)

df_node_corr.to_csv("correlation_node_level.csv", sep=SEP, index=False)

# ==================================================
# 2️⃣ PANEL-LEVEL (YEAR-NORMALIZED) CORRELATION
# ==================================================
print("\n=== PANEL-LEVEL YEAR-NORMALIZED CORRELATIONS ===")

sp = pd.read_csv(SPECTRAL_PANEL, sep=SEP)
bc_p = pd.read_csv(BC_PANEL, sep=SEP)

# harmonize year and coords
for df in (sp, bc_p):
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["year"] = df["year"].round().astype(int)

    df["x"] = pd.to_numeric(df["x"], errors="coerce").round(0).astype(int)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").round(0).astype(int)

# --- compute SpectralNorm if missing
if "SpectralNorm" not in sp.columns:
    print("SpectralNorm not found — computing from eig_*")
    sp = compute_spectral_norm_from_eigs(sp)

panel = sp.merge(
    bc_p,
    on=["year", "x", "y"],
    how="inner"
)

panel = panel.dropna(subset=["SpectralNorm", "Betweenness"])

# z-score per year
panel["Spectral_z"] = (
    panel["SpectralNorm"]
    - panel.groupby("year")["SpectralNorm"].transform("mean")
) / panel.groupby("year")["SpectralNorm"].transform("std")

panel["BC_z"] = (
    panel["Betweenness"]
    - panel.groupby("year")["Betweenness"].transform("mean")
) / panel.groupby("year")["Betweenness"].transform("std")

year_corr = []

for year, g in panel.groupby("year"):
    mask = np.isfinite(g["Spectral_z"]) & np.isfinite(g["BC_z"])
    if mask.sum() < 10:
        continue

    r = spearmanr(g.loc[mask, "Spectral_z"], g.loc[mask, "BC_z"])
    year_corr.append({
        "year": year,
        "Spearman_r": r.statistic,
        "p_value": r.pvalue,
        "N": int(mask.sum())
    })

df_year_corr = pd.DataFrame(year_corr)
print(df_year_corr.head())

df_year_corr.to_csv("correlation_panel_by_year.csv", sep=SEP, index=False)

# ==================================================
# 3️⃣ TYPE-LEVEL CORRESPONDENCE
# ==================================================
print("\n=== TYPE-LEVEL CORRESPONDENCE ===")

if os.path.exists(SPECTRAL_TYPES) and os.path.exists(BC_TYPES):
    st = pd.read_csv(SPECTRAL_TYPES, sep=SEP)[["x", "y", "EvolutionType"]]
    bt = pd.read_csv(BC_TYPES, sep=SEP)[["x", "y", "BC_EvolutionType"]]

    types = st.merge(bt, on=["x", "y"], how="inner")

    cross = pd.crosstab(
        types["EvolutionType"],
        types["BC_EvolutionType"],
        normalize="index"
    )

    print(cross)
    cross.to_csv("correlation_type_cross.csv", sep=SEP)

else:
    print("Type files not found — skipping type-level analysis")

print("\nDONE.")
