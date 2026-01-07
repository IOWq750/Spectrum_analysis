# -*- coding: utf-8 -*-
"""
Compute temporal evolution of node betweenness centrality
from yearly line shapefiles and aggregate trajectory descriptors.

Analogue to spectral evolution analysis.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point

# ==================================================
# PATHS
# ==================================================
BASE_PATH = r"d:\YandexDisk\Projects\MES_evolution\BackUp251124\SHP"
OUT_PATH = r"d:\YandexDisk\Projects\MES_evolution\Spectrum_analysis\MES_Out_Points"

os.chdir(OUT_PATH)

YEARS = range(1933, 2021)

SEP = ";"
SNAP_TOL = 1.0        # meters
USE_LENGTH = True     # edge weight

# ==================================================
# HELPERS
# ==================================================
def quantize(val, tol):
    return round(val / tol) * tol


def node_key(pt, tol):
    return (quantize(pt.x, tol), quantize(pt.y, tol))


def build_graph(edges_gdf):
    node_map = {}
    next_id = 0
    G = nx.Graph()

    def get_node(pt):
        nonlocal next_id
        k = node_key(pt, SNAP_TOL)
        if k not in node_map:
            node_map[k] = next_id
            next_id += 1
        return node_map[k], k

    for _, row in edges_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if isinstance(geom, LineString):
            lines = [geom]
        else:
            lines = [g for g in geom.geoms if isinstance(g, LineString)]

        for ls in lines:
            p0 = Point(ls.coords[0])
            p1 = Point(ls.coords[-1])

            u, ku = get_node(p0)
            v, kv = get_node(p1)

            if u == v:
                continue

            w = float(ls.length) if USE_LENGTH else 1.0

            if G.has_edge(ku, kv):
                G[ku][kv]["weight"] += w
            else:
                G.add_edge(ku, kv, weight=w)

    return G


# ==================================================
# MAIN LOOP: YEARLY BC
# ==================================================
records = []

for year in YEARS:
    shp = os.path.join(BASE_PATH, f"TL_{year}.shp")
    if not os.path.exists(shp):
        continue

    print(f"Processing year {year}")

    gdf = gpd.read_file(shp)

    if gdf.crs is None or gdf.crs.is_geographic:
        raise ValueError("Projected CRS required")

    G = build_graph(gdf)

    if G.number_of_nodes() == 0:
        continue

    bc = nx.betweenness_centrality(
        G,
        normalized=True,
        weight="weight"
    )

    for (x, y), val in bc.items():
        records.append({
            "year": year,
            "x": int(x),
            "y": int(y),
            "Betweenness": float(val)
        })

panel = pd.DataFrame(records)
panel.to_csv("betweenness_panel.csv", sep=SEP, index=False)
print("Saved betweenness_panel.csv")

# ==================================================
# AGGREGATE PER NODE
# ==================================================
out = []

for (x, y), g in panel.groupby(["x", "y"]):
    g = g.sort_values("year")
    bc = g["Betweenness"].values
    years = g["year"].values

    if len(bc) < 3:
        continue

    # trend
    try:
        slope = np.polyfit(years, bc, 1)[0]
    except Exception:
        slope = np.nan

    # deltas
    deltas = np.abs(np.diff(bc))

    # Euclidean norm
    bc_norm = np.sqrt(np.sum(bc ** 2))

    # deviation from mean trajectory
    mean_bc = np.mean(bc)
    deviation = np.sqrt(np.mean((bc - mean_bc) ** 2))

    out.append({
        "x": x,
        "y": y,
        "life_span": len(bc),
        "mean_BC": mean_bc,
        "max_BC": np.max(bc),
        "trend_BC": slope,
        "mean_DeltaBC": np.mean(deltas),
        "max_DeltaBC": np.max(deltas),
        "BC_EuclideanNorm": bc_norm,
        "BC_Deviation": deviation
    })

nodes = pd.DataFrame(out)
nodes.to_csv("betweenness_node_profiles.csv", sep=SEP, index=False)
print("Saved betweenness_node_profiles.csv")
