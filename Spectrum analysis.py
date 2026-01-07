# -*- coding: utf-8 -*-
"""
Build a graph from a line shapefile, compute Laplacian spectrum,
attach eigenvectors to nodes, and export nodes as a shapefile.

Usage example (edit paths at the bottom):

- edges_shp = "data/roads.shp"
- out_nodes_shp = "out/nodes_with_spectrum.shp"

Options:
- weight_field: name of attribute for edge weights (e.g., 'CAPACITY').
               If None, use length (in layer CRS) or 1 if use_length=False.
- use_length:   if True and weight_field is None, use line length as weight.
- snap_tol:     snapping tolerance in CRS units (e.g., meters).
               Endpoints whose coords are within snap_tol land on the same node.
- k_eig:        how many non-trivial eigenvectors to export (Fiedler = #1).
"""

import os
import math
import numpy as np
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

def build_graph_from_lines(
    edges_gdf: gpd.GeoDataFrame,
    weight_field: str | None = None,
    use_length: bool = True,
    snap_tol: float = 0.5,
    collapse_parallel: bool = True,
):
    """
    Returns:
        G (nx.Graph): undirected weighted graph
        nodes_gdf (GeoDataFrame): unique snapped nodes with 'node_id'
        node_index (dict): mapping node_id -> integer index (for spectra)
    """
    if edges_gdf.crs is None:
        raise ValueError("Input shapefile has no CRS. Please set a projected CRS (meters).")

    # helper: quantize a coordinate to grid size snap_tol (fast & robust)
    def quantize(val, tol):
        if tol <= 0:
            return val
        return round(val / tol) * tol

    def make_node_key(pt: Point, tol: float):
        return (quantize(pt.x, tol), quantize(pt.y, tol))

    # collect nodes
    node_key_to_id = {}
    node_id_to_point = {}
    next_id = 0

    def get_or_create_node_id(pt: Point):
        nonlocal next_id
        key = make_node_key(pt, snap_tol)
        if key not in node_key_to_id:
            node_key_to_id[key] = next_id
            node_id_to_point[next_id] = Point(key[0], key[1])
            next_id += 1
        return node_key_to_id[key]

    # build graph
    G = nx.Graph()

    # iterate edges
    for _, row in edges_gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, LineString):
            lines = [geom]
        else:
            # MultiLineString / GeometryCollection -> flatten LineStrings
            lines = [g for g in geom.geoms if isinstance(g, LineString)]

        for ls in lines:
            start_pt = Point(ls.coords[0])
            end_pt   = Point(ls.coords[-1])
            u = get_or_create_node_id(start_pt)
            v = get_or_create_node_id(end_pt)
            if u == v:
                continue  # skip zero-length / closed to same node

            # edge weight
            if weight_field is not None and weight_field in row and row[weight_field] is not None:
                w = float(row[weight_field])
            elif use_length:
                w = float(ls.length)
            else:
                w = 1.0

            # collapse parallel edges by summing weights
            if collapse_parallel:
                if G.has_edge(u, v):
                    G[u][v]['weight'] += w
                else:
                    G.add_edge(u, v, weight=w)
            else:
                # keep parallel edges separate by adding a unique key
                # (networkx.MultiGraph would be more appropriate, but
                #  for spectral analysis we need a simple weighted Graph)
                if G.has_edge(u, v):
                    G[u][v]['weight'] += w
                else:
                    G.add_edge(u, v, weight=w)

    # build nodes GeoDataFrame
    node_ids = sorted(node_id_to_point.keys())
    node_points = [node_id_to_point[nid] for nid in node_ids]
    nodes_gdf = gpd.GeoDataFrame({"node_id": node_ids}, geometry=node_points, crs=edges_gdf.crs)

    # ensure every node exists in G (isolated endpoints possible on invalid data)
    for nid in node_ids:
        if nid not in G:
            G.add_node(nid)

    # Build a stable node index mapping (0..N-1) for consistent matrix layout
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    return G, nodes_gdf, node_index


def laplacian_spectrum(G: nx.Graph, node_index: dict[str, int]):
    """
    Compute weighted Laplacian eigenpairs.
    Returns:
        eigvals (np.ndarray, shape [N])
        eigvecs (np.ndarray, shape [N, N]) columns correspond to eigenvectors
        ordered_node_ids (list) nodes ordered as in the matrix
    """
    ordered_node_ids = [nid for nid, _ in sorted(node_index.items(), key=lambda kv: kv[1])]
    # Construct weighted adjacency in the given order
    N = len(ordered_node_ids)
    A = np.zeros((N, N), dtype=float)
    for u, v, data in G.edges(data=True):
        if u not in node_index or v not in node_index:
            continue
        i = node_index[u]
        j = node_index[v]
        w = float(data.get('weight', 1.0))
        A[i, j] += w
        A[j, i] += w

    # Degree matrix
    d = A.sum(axis=1)
    L = np.diag(d) - A

    # eigen-decomposition (symmetric PSD)
    eigvals, eigvecs = np.linalg.eigh(L)
    return eigvals, eigvecs, ordered_node_ids


def attach_eigenvectors_to_nodes(nodes_gdf: gpd.GeoDataFrame,
                                 eigvals: np.ndarray,
                                 eigvecs: np.ndarray,
                                 ordered_node_ids: list[int],
                                 k_eig: int = 3):
    """
    Add k first non-trivial eigenvectors (skip λ0=0) as columns eig_1..eig_k.
    """
    # map node_id -> row index in eigvecs
    node_pos = {nid: i for i, nid in enumerate(ordered_node_ids)}
    # find indices of smallest eigenvalues sorted
    # eigh already returns sorted ascending. skip the first (≈0) if graph connected
    start_idx = 1  # Fiedler as eig_1
    end_idx = min(start_idx + k_eig, eigvecs.shape[1])

    for k, col_idx in enumerate(range(start_idx, end_idx), start=1):
        print(k)
        colname = f"eig_{k}"
        values = np.zeros(len(nodes_gdf), dtype=float)
        for r, nid in enumerate(nodes_gdf["node_id"].tolist()):
            i = node_pos[nid]
            values[r] = eigvecs[i, col_idx]
        nodes_gdf[colname] = values

    # also полезно сохранить несколько служебных значений
    nodes_gdf["lambda_1"] = eigvals[1] if eigvals.shape[0] > 1 else np.nan
    return nodes_gdf


def main(
    edges_shp: str,
    out_nodes_shp: str,
    weight_field: str | None = None,
    use_length: bool = True,
    snap_tol: float = 0.5,
    k_eig: int = 3,
):
    # 1) Read lines
    edges_gdf = gpd.read_file(edges_shp)

    # Рекомендуется ПРОЕКТИРОВАННЫЙ CRS (метры), чтобы snap_tol и длины были осмысленны
    if edges_gdf.crs is None or edges_gdf.crs.is_geographic:
        print("⚠️  Рекомендуется перепроецировать слой в метры (например, UTM), "
              "чтобы корректно работать с длинами и snap_tol.")

    # 2) Build graph
    G, nodes_gdf, node_index = build_graph_from_lines(
        edges_gdf,
        weight_field=weight_field,
        use_length=use_length,
        snap_tol=snap_tol,
        collapse_parallel=True,
    )

    # 3) Spectrum
    eigvals, eigvecs, ordered_node_ids = laplacian_spectrum(G, node_index)

    # 4) Attach k eigenvectors to nodes
    nodes_gdf = attach_eigenvectors_to_nodes(
        nodes_gdf, eigvals, eigvecs, ordered_node_ids, k_eig=k_eig
    )

    # 5) Export nodes with attributes
    #os.makedirs(os.path.dirname(out_nodes_shp), exist_ok=True)
    nodes_gdf.to_file(out_nodes_shp)
    print(f"✅ Saved: {out_nodes_shp}")
    print("Eigenvalues (first 10):", np.round(eigvals[:10], 6))


if __name__ == "__main__":
    path = r'd:\YandexDisk\Projects\MES_evolution\BackUp251124\SHP'
    out = r'd:\YandexDisk\Projects\MES_evolution\Spectrum_analysis\MES_Out_Points'
    # Вход: линейный шейпфайл (дороги/линии сети)
    for year in range(1933, 2021):
        edges_shp = f"{path}\TL_{year}.shp"
        # Выход: точечный шейпфайл с узлами и собственными векторами
        out_nodes_shp = f"{out}\P_{year}_spectrum.shp"

        # Если у рёбер есть числовой атрибут-вес (например, 'WEIGHT'/'CAPACITY'), укажите его:
        weight_field = None   # например: "CAPACITY"
        use_length = True     # если weight_field=None, использовать длину линии как вес?
        snap_tol = 1.0        # толерантность снэпа в единицах CRS (например, метры)
        k_eig = 10             # сколько нетривиальных собственных векторов сохранить

        # Запуск
        main(
            edges_shp=edges_shp,
            out_nodes_shp=out_nodes_shp,
            weight_field=weight_field,
            use_length=use_length,
            snap_tol=snap_tol,
            k_eig=k_eig,
        )
