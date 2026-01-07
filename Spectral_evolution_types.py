# -*- coding: utf-8 -*-
"""
Identify evolutionary node types from spectral trajectories,
attach real substation names, and export results as CSV and Shapefile.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# WORKING DIRECTORY
# ==================================================
os.chdir(r"d:\YandexDisk\Projects\MES_evolution\Spectrum_analysis\MES_Out_Points")

# ==================================================
# INPUT / OUTPUT
# ==================================================
PANEL_CSV = "spectral_panel.csv"
POINTS_SHP = "Initial_Points.shp"

OUT_CSV = "node_evolution_types.csv"
OUT_SHP = "node_evolution_types.shp"

SEP = ";"
N_CLUSTERS = 5
RANDOM_STATE = 42
K_EIG_MAX = 20

# ==================================================
# HELPERS
# ==================================================
def get_eig_cols(df, kmax=20):
    cols = [c for c in df.columns if str(c).startswith("eig_")]

    def idx(c):
        try:
            return int(c.split("_")[1])
        except Exception:
            return 999

    return sorted(cols, key=idx)[:kmax]


def compute_spectral_norm(df, eig_cols):
    for c in eig_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    X = df[eig_cols].to_numpy(dtype=float)
    sn = np.sqrt(np.nansum(X * X, axis=1))
    sn[np.all(np.isnan(X), axis=1)] = np.nan
    df["SpectralNorm"] = sn
    return df


# ==================================================
# AGGREGATION PER NODE
# ==================================================
def aggregate_per_node(df):
    rows = []

    for (x, y), g in df.groupby(["x", "y"]):
        g = g.sort_values("year")

        sn = g["SpectralNorm"].values
        years = g["year"].values

        if len(sn) < 3:
            continue

        try:
            slope = np.polyfit(years, sn, 1)[0]
        except Exception:
            slope = np.nan

        rows.append({
            "x": int(x),
            "y": int(y),
            "life_span": len(sn),
            "mean_SpectralNorm": np.nanmean(sn),
            "max_SpectralNorm": np.nanmax(sn),
            "trend_SpectralNorm": slope,
            "mean_DeltaSpec": np.nanmean(np.abs(np.diff(sn))),
            "max_DeltaSpec": np.nanmax(np.abs(np.diff(sn))),
            "n_role_changes": int(np.sum(np.sign(sn[:-1]) != np.sign(sn[1:])))
        })

    return pd.DataFrame(rows)


# ==================================================
# MAIN
# ==================================================
def main():

    # ---------- Load spectral panel
    panel = pd.read_csv(PANEL_CSV, sep=SEP)

    # year
    panel["year"] = pd.to_numeric(panel["year"], errors="coerce")
    panel = panel.dropna(subset=["year"])
    panel["year"] = panel["year"].round().astype(int)

    # coordinates
    panel["x"] = pd.to_numeric(panel["x"], errors="coerce").round(0).astype("Int64")
    panel["y"] = pd.to_numeric(panel["y"], errors="coerce").round(0).astype("Int64")
    panel = panel.dropna(subset=["x", "y"])

    # spectral norm
    if "SpectralNorm" not in panel.columns:
        eig_cols = get_eig_cols(panel, K_EIG_MAX)
        panel = compute_spectral_norm(panel, eig_cols)

    panel = panel.dropna(subset=["SpectralNorm"])

    # ---------- Aggregate per node
    nodes = aggregate_per_node(panel)
    print(f"Aggregated nodes: {len(nodes)}")

    # ---------- Clustering
    feat_cols = [
        "mean_SpectralNorm",
        "max_SpectralNorm",
        "trend_SpectralNorm",
        "mean_DeltaSpec",
        "max_DeltaSpec",
        "n_role_changes",
        "life_span"
    ]

    X = (
        nodes[feat_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .values
    )

    Xs = StandardScaler().fit_transform(X)

    km = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=20
    )

    nodes["EvolutionType"] = km.fit_predict(Xs)

    # ---------- Attach real substation names
    print("Attaching substation names...")
    pts = gpd.read_file(POINTS_SHP)

    g_nodes = gpd.GeoDataFrame(
        nodes,
        geometry=[Point(xy) for xy in zip(nodes["x"], nodes["y"])],
        crs=pts.crs
    )

    g_joined = gpd.sjoin_nearest(
        g_nodes,
        pts[["Name", "Name_en", "geometry"]],
        how="left",
        distance_col="dist_to_station"
    )

    # ---------- Save CSV
    df_out = pd.DataFrame(g_joined.drop(columns="geometry"))
    df_out.to_csv(OUT_CSV, sep=SEP, index=False)
    print(f"Saved table: {OUT_CSV}")

    # ---------- Save Shapefile
    g_joined.to_file(OUT_SHP, encoding="utf-8")
    print(f"Saved shapefile: {OUT_SHP}")

    # ---------- Diagnostics
    print("\nEvolutionType counts:")
    print(df_out["EvolutionType"].value_counts().sort_index())

    print("\nLargest distances to named stations:")
    print(
        df_out[["Name", "EvolutionType", "dist_to_station"]]
        .sort_values("dist_to_station", ascending=False)
        .head()
    )


if __name__ == "__main__":
    main()
