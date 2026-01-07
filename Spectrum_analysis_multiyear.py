# -*- coding: utf-8 -*-
"""
Year-by-year Laplacian eigenvectors for a changing, possibly disconnected power grid,
with Procrustes alignment of spectral embeddings across years using stable node keys.

Key ideas:
- For a graph with c connected components, Laplacian has c zero eigenvalues.
  We skip the first c eigenvectors and export the next k (non-trivial).
- Use sparse eigensolver (scipy.sparse.linalg.eigsh) for scalability.
- Align embeddings between consecutive years using Orthogonal Procrustes on common nodes.

Outputs per year:
- Point layer (shp or gpkg) with node_id, node_key, comp_id, degree_w, eig_1..eig_k, etc.
- Optional: one CSV table with all years stacked.

Notes:
- Shapefile has limitations (field name length 10, numeric precision). Prefer GPKG if possible.
"""

import os
import re
import math
import numpy as np
import geopandas as gpd
import networkx as nx

from shapely.geometry import Point, LineString
from scipy.sparse.linalg import eigsh

# -----------------------------
# Graph building
# -----------------------------

def build_graph_from_lines(
    edges_gdf: gpd.GeoDataFrame,
    weight_field: str | None = None,
    use_length: bool = True,
    snap_tol: float = 0.0,
    collapse_parallel: bool = True,
):
    """
    Build an undirected weighted graph from line geometries.
    Nodes are created from line endpoints. Node key is (x, y) after optional quantization.

    Returns:
        G: nx.Graph
        nodes_gdf: GeoDataFrame with columns: node_id, node_key, x, y, geometry
        node_order: list of node_id in stable order for matrix layout
        node_id_to_key: dict[node_id] -> tuple(x, y)
    """
    if edges_gdf.crs is None:
        raise ValueError("Input layer has no CRS. Use a projected CRS (meters) for meaningful lengths/tolerance.")

    def quantize(val: float, tol: float) -> float:
        if tol is None or tol <= 0:
            return float(val)
        return round(val / tol) * tol

    def make_node_key(x: float, y: float, tol: float):
        return (quantize(x, tol), quantize(y, tol))

    node_key_to_id: dict[tuple[float, float], int] = {}
    node_id_to_key: dict[int, tuple[float, float]] = {}
    next_id = 0

    def get_or_create_node_id(x: float, y: float) -> int:
        nonlocal next_id
        key = make_node_key(x, y, snap_tol)
        if key not in node_key_to_id:
            node_key_to_id[key] = next_id
            node_id_to_key[next_id] = key
            next_id += 1
        return node_key_to_id[key]

    G = nx.Graph()

    for _, row in edges_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        if isinstance(geom, LineString):
            lines = [geom]
        else:
            # MultiLineString/GeometryCollection -> flatten
            try:
                lines = [g for g in geom.geoms if isinstance(g, LineString)]
            except Exception:
                continue

        for ls in lines:
            if ls is None or ls.is_empty:
                continue
            x1, y1 = ls.coords[0]
            x2, y2 = ls.coords[-1]

            u = get_or_create_node_id(x1, y1)
            v = get_or_create_node_id(x2, y2)
            if u == v:
                continue

            # edge weight
            if weight_field is not None and weight_field in row and row[weight_field] is not None:
                w = float(row[weight_field])
            elif use_length:
                w = float(ls.length)
            else:
                w = 1.0

            if collapse_parallel:
                if G.has_edge(u, v):
                    G[u][v]["weight"] += w
                else:
                    G.add_edge(u, v, weight=w)
            else:
                # still collapses into simple graph (spectral methods require simple matrix)
                if G.has_edge(u, v):
                    G[u][v]["weight"] += w
                else:
                    G.add_edge(u, v, weight=w)

    # Ensure nodes exist even if isolated
    for nid in node_id_to_key.keys():
        if nid not in G:
            G.add_node(nid)

    node_order = sorted(node_id_to_key.keys())

    # nodes GeoDataFrame
    xs = [node_id_to_key[nid][0] for nid in node_order]
    ys = [node_id_to_key[nid][1] for nid in node_order]
    pts = [Point(x, y) for x, y in zip(xs, ys)]
    node_keys = [node_id_to_key[nid] for nid in node_order]

    nodes_gdf = gpd.GeoDataFrame(
        {
            "node_id": node_order,
            "node_key": [str(k) for k in node_keys],  # stable string key for joins/exports
            "x": xs,
            "y": ys,
        },
        geometry=pts,
        crs=edges_gdf.crs,
    )

    return G, nodes_gdf, node_order, node_id_to_key


def add_component_ids(G: nx.Graph, node_order: list[int]) -> dict[int, int]:
    """Return dict[node_id] -> component id (0..c-1)."""
    comp_id = {}
    for i, comp in enumerate(nx.connected_components(G)):
        for nid in comp:
            comp_id[nid] = i
    # make sure every node has a comp_id
    for nid in node_order:
        comp_id.setdefault(nid, -1)
    return comp_id


# -----------------------------
# Spectral computation
# -----------------------------

def compute_laplacian_eigs(
    G: nx.Graph,
    node_order: list[int],
    k_nontrivial: int,
    use_normalized: bool = True,
    tol_eig: float = 1e-8,
    maxiter: int = 2000,
    random_state: int = 0,
):
    """
    Compute first k_nontrivial eigenvectors AFTER the zero-eigenvalue subspace.
    For a graph with c components, skip c eigenvectors.

    Returns:
        eigvals_small: array of smallest (c+k) eigenvalues (sorted)
        X: embedding matrix [N, k_nontrivial] (eigenvectors corresponding to indices c..c+k-1)
        c: number of connected components
    """
    N = len(node_order)
    if N == 0:
        return np.array([]), np.empty((0, k_nontrivial)), 0

    c = nx.number_connected_components(G)
    # need at least (c + k) eigenpairs, but cannot exceed N-1 for eigsh k parameter
    need = c + k_nontrivial
    if need >= N:
        # for tiny graphs, reduce k to feasible
        need = max(1, N - 1)
        k_nontrivial = max(0, need - c)

    if k_nontrivial <= 0:
        # no room for nontrivial vectors
        return np.array([]), np.empty((N, 0)), c

    # Build sparse Laplacian matrix in node_order
    # Use networkx to get sparse matrix quickly
    if use_normalized:
        L = nx.normalized_laplacian_matrix(G, nodelist=node_order, weight="weight")
    else:
        L = nx.laplacian_matrix(G, nodelist=node_order, weight="weight")

    # eigsh for smallest eigenvalues:
    # Robust trick for PSD matrices: use shift-invert with sigma=0 (find eigenvalues near 0)
    # which='LM' in shift-invert mode returns closest to sigma.
    rng = np.random.default_rng(random_state)
    v0 = rng.normal(size=N)

    try:
        eigvals, eigvecs = eigsh(
            L,
            k=need,
            sigma=0.0,
            which="LM",
            tol=tol_eig,
            maxiter=maxiter,
            v0=v0,
        )
    except Exception:
        # fallback: try direct smallest magnitude (can be less robust)
        eigvals, eigvecs = eigsh(
            L,
            k=need,
            which="SM",
            tol=tol_eig,
            maxiter=maxiter,
            v0=v0,
        )

    # sort ascending
    idx = np.argsort(eigvals)
    eigvals = np.asarray(eigvals)[idx]
    eigvecs = np.asarray(eigvecs)[:, idx]

    # We skip first c eigenvectors. Sometimes numerics can give tiny nonzero values;
    # but for disconnected graph multiplicity should match c.
    start = c
    end = min(c + k_nontrivial, eigvecs.shape[1])
    X = eigvecs[:, start:end].copy()

    return eigvals, X, c


# -----------------------------
# Procrustes alignment
# -----------------------------

def orthogonal_procrustes_align(X_prev: np.ndarray, X_curr: np.ndarray) -> np.ndarray:
    """
    Find orthogonal R that minimizes ||X_curr R - X_prev||_F.
    Return aligned X_curr_aligned = X_curr R.

    Assumes X_prev and X_curr have same shape [M, k].
    """
    # cross-covariance
    M = X_curr.T @ X_prev
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    return X_curr @ R


# -----------------------------
# Year processing
# -----------------------------

def process_year(
    edges_shp: str,
    weight_field: str | None,
    use_length: bool,
    snap_tol: float,
    k_eig: int,
    use_normalized: bool,
):
    edges_gdf = gpd.read_file(edges_shp)

    if edges_gdf.crs is None or edges_gdf.crs.is_geographic:
        print("⚠️  CRS geographic/None. Лучше перепроецировать в метры для корректных длин/весов/snap_tol.")

    G, nodes_gdf, node_order, node_id_to_key = build_graph_from_lines(
        edges_gdf,
        weight_field=weight_field,
        use_length=use_length,
        snap_tol=snap_tol,
        collapse_parallel=True,
    )

    comp_id = add_component_ids(G, node_order)
    nodes_gdf["comp_id"] = [comp_id[nid] for nid in node_order]

    # weighted degree
    deg_w = dict(G.degree(weight="weight"))
    nodes_gdf["degree_w"] = [float(deg_w.get(nid, 0.0)) for nid in node_order]

    eigvals, X, c = compute_laplacian_eigs(
        G,
        node_order=node_order,
        k_nontrivial=k_eig,
        use_normalized=use_normalized,
    )

    # attach
    for j in range(X.shape[1]):
        nodes_gdf[f"eig_{j+1}"] = X[:, j].astype(float)

    nodes_gdf["n_comp"] = int(c)
    if eigvals.size > 0:
        # store some leading eigenvalues for reference (after zeros)
        # (names short for shapefile)
        for t in range(min(5, len(eigvals))):
            nodes_gdf[f"lam{t}"] = float(eigvals[t])
    else:
        nodes_gdf["lam0"] = np.nan

    # Return both gdf and raw embedding with node_key index for alignment
    # Use node_key as stable join key
    node_keys = nodes_gdf["node_key"].to_numpy()
    return nodes_gdf, node_keys, X


def save_nodes(nodes_gdf: gpd.GeoDataFrame, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nodes_gdf.to_file(out_path)
    print(f"✅ Saved: {out_path}")


# -----------------------------
# Main pipeline
# -----------------------------

def run_pipeline(
    in_dir: str,
    out_dir: str,
    years: list[int],
    in_pattern: str = r"TL_{year}.shp",
    out_pattern: str = r"P_{year}_spectrum.shp",
    weight_field: str | None = None,
    use_length: bool = True,
    snap_tol: float = 0.0,
    k_eig: int = 10,
    use_normalized: bool = True,
    export_stacked_csv: bool = True,
    stacked_csv_name: str = "spectral_panel.csv",
):
    """
    Processes years in order, aligns embeddings year-to-year (Procrustes),
    saves per-year files and optionally a stacked CSV table.
    """
    all_rows = []

    prev_year = None
    prev_key_to_vec = None  # dict[node_key] -> vector (k,)
    prev_k = None

    for year in years:
        in_path = os.path.join(in_dir, in_pattern.format(year=year))
        out_path = os.path.join(out_dir, out_pattern.format(year=year))

        if not os.path.exists(in_path):
            print(f"⏭️  Missing input for {year}: {in_path}")
            continue

        print(f"\n=== Year {year} ===")
        nodes_gdf, node_keys, X = process_year(
            edges_shp=in_path,
            weight_field=weight_field,
            use_length=use_length,
            snap_tol=snap_tol,
            k_eig=k_eig,
            use_normalized=use_normalized,
        )

        # Align to previous year if possible
        if prev_key_to_vec is not None and X.shape[1] > 0:
            k_curr = X.shape[1]
            k_prev = prev_k
            k_use = min(k_curr, k_prev)

            # find common keys
            curr_key_to_idx = {k: i for i, k in enumerate(node_keys)}
            common_keys = [k for k in prev_key_to_vec.keys() if k in curr_key_to_idx]

            if len(common_keys) >= max(10, 2 * k_use):
                X_prev_common = np.vstack([prev_key_to_vec[k][:k_use] for k in common_keys])
                X_curr_common = np.vstack([X[curr_key_to_idx[k], :k_use] for k in common_keys])

                # center? (usually not needed for Laplacian eigvecs; keep as-is)
                X_curr_aligned = orthogonal_procrustes_align(X_prev_common, X_curr_common)

                # Apply found rotation to ALL current nodes (only first k_use dims)
                # Compute R using SVD from common subsets, then rotate full X
                M = X_curr_common.T @ X_prev_common
                U, _, Vt = np.linalg.svd(M, full_matrices=False)
                R = U @ Vt

                X[:, :k_use] = X[:, :k_use] @ R

                # write back aligned columns
                for j in range(k_use):
                    nodes_gdf[f"eig_{j+1}"] = X[:, j].astype(float)

                nodes_gdf["aligned_to"] = int(prev_year)
                nodes_gdf["n_common"] = int(len(common_keys))
                print(f"🔧 Aligned {year} -> {prev_year} using {len(common_keys)} common nodes (k={k_use}).")
            else:
                nodes_gdf["aligned_to"] = -1
                nodes_gdf["n_common"] = int(len(common_keys))
                print(f"⚠️  Not enough common nodes to align {year} to {prev_year}: {len(common_keys)} found.")
        else:
            nodes_gdf["aligned_to"] = -1
            nodes_gdf["n_common"] = 0

        # Save per-year
        save_nodes(nodes_gdf, out_path)

        # Update prev cache
        # cache vectors per key for next year's alignment
        prev_k = X.shape[1]
        prev_key_to_vec = {k: X[i, :].copy() for i, k in enumerate(node_keys)}
        prev_year = year

        # Collect for stacked table (CSV)
        if export_stacked_csv:
            base_cols = ["node_key", "x", "y", "comp_id", "degree_w", "n_comp", "aligned_to", "n_common"]
            eig_cols = [c for c in nodes_gdf.columns if re.fullmatch(r"eig_\d+", str(c))]

            # Make a plain (non-geometry) table
            df = nodes_gdf[base_cols + eig_cols].copy()
            df["year"] = int(year)
            all_rows.append(df)

    if export_stacked_csv and all_rows:
        import pandas as pd
        panel = pd.concat(all_rows, ignore_index=True)
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, stacked_csv_name)
        panel.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"\n📦 Stacked panel CSV saved: {csv_path}")


# -----------------------------
# Run example (edit paths)
# -----------------------------
if __name__ == "__main__":
    in_dir = r"d:\YandexDisk\Projects\MES_evolution\BackUp251124\SHP"
    out_dir = r"d:\YandexDisk\Projects\MES_evolution\Spectrum_analysis\MES_Out_Points"

    years = list(range(1933, 2021))

    run_pipeline(
        in_dir=in_dir,
        out_dir=out_dir,
        years=years,
        in_pattern=r"TL_{year}.shp",
        out_pattern=r"P_{year}_spectrum.shp",
        weight_field=None,     # e.g., "CAPACITY" if exists
        use_length=True,
        snap_tol=1.0,          # вы сказали координаты уникальны -> можно 0.0 (без квантизации)
        k_eig=20,              # попробуйте 20-50; дальше смотреть устойчивость
        use_normalized=True,   # рекомендую True для сопоставимости между годами
        export_stacked_csv=True,
        stacked_csv_name="spectral_panel.csv",
    )
