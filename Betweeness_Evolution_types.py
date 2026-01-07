# -*- coding: utf-8 -*-
"""
Cluster nodes by betweenness centrality evolution profiles.
"""

import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# PATHS
# ==================================================
os.chdir(r"d:\YandexDisk\Projects\MES_evolution\Spectrum_analysis\MES_Out_Points")

IN_PROFILES = "betweenness_node_profiles.csv"
OUT_TYPES = "betweenness_evolution_types.csv"

SEP = ";"
N_CLUSTERS = 6
RANDOM_STATE = 42

# ==================================================
# LOAD
# ==================================================
df = pd.read_csv(IN_PROFILES, sep=SEP)

# ==================================================
# FEATURES
# ==================================================
feat_cols = [
    "mean_BC",
    "max_BC",
    "trend_BC",
    "mean_DeltaBC",
    "max_DeltaBC",
    "BC_EuclideanNorm",
    "BC_Deviation",
    "life_span"
]

X = (
    df[feat_cols]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .values
)

Xs = StandardScaler().fit_transform(X)

# ==================================================
# CLUSTERING
# ==================================================
km = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=20
)

df["BC_EvolutionType"] = km.fit_predict(Xs)

# ==================================================
# SAVE
# ==================================================
df.to_csv(OUT_TYPES, sep=SEP, index=False)

print("BC Evolution types:")
print(df["BC_EvolutionType"].value_counts().sort_index())
print(f"Saved: {OUT_TYPES}")
