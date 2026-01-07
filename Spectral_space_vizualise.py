import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os
import matplotlib as mpl
from matplotlib.gridspec import GridSpec

# =====================
# SETTINGS
# =====================
os.chdir(r'd:\YandexDisk\Projects\MES_evolution\Spectrum_analysis\MES_Out_Points')

points_shp = r"P_1970_spectrum.shp"
lines_shp = r"TL_1970.shp"   # <<< ЛИНЕЙНЫЙ ШЕЙПФАЙЛ СЕТИ

eig_cols = ["eig_1", "eig_2", "eig_3"]

out_dir = "output_viz"
os.makedirs(out_dir, exist_ok=True)

out_png = os.path.join(out_dir, "points_spatial_and_spectral.png")

# =====================
# LOAD DATA
# =====================
gdf_pts = gpd.read_file(points_shp)
gdf_lines = gpd.read_file(lines_shp)

for c in eig_cols:
    if c not in gdf_pts.columns:
        raise ValueError(f"Column '{c}' not found")

# фильтрация точек
gdf_pts = gdf_pts.dropna(subset=eig_cols)
gdf_pts = gdf_pts[(gdf_pts[eig_cols] != 0).any(axis=1)]

# =====================
# COMMON COLOR SCALE
# =====================
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(
    vmin=gdf_pts[eig_cols[0]].min(),
    vmax=gdf_pts[eig_cols[0]].max()
)

# =====================
# FIGURE LAYOUT
# =====================
fig = plt.figure(figsize=(15, 8))
gs = GridSpec(
    nrows=1,
    ncols=3,
    width_ratios=[1, 1, 0.05],
    wspace=0.25
)

# ---------- Spatial distribution with network ----------
ax1 = fig.add_subplot(gs[0, 0])

# network edges (background)
gdf_lines.plot(
    ax=ax1,
    color="lightgray",
    linewidth=0.8,
    alpha=0.7,
    zorder=1
)

# network nodes
gdf_pts.plot(
    ax=ax1,
    column=eig_cols[0],
    cmap=cmap,
    norm=norm,
    markersize=15,
    legend=False,
    zorder=2
)

ax1.set_title(
    "Spatial distribution of network nodes\n(colored by the first eigenvector)",
    fontsize=11
)
ax1.set_axis_off()
xmin, ymin, xmax, ymax = gdf_pts.total_bounds

dx = (xmax - xmin) * 0.05
dy = (ymax - ymin) * 0.05

ax1.set_xlim(xmin - dx, xmax + dx)
ax1.set_ylim(ymin - dy, ymax + dy)


# ---------- Spectral space ----------
ax2 = fig.add_subplot(gs[0, 1], projection="3d")

ax2.scatter(
    gdf_pts[eig_cols[0]],
    gdf_pts[eig_cols[1]],
    gdf_pts[eig_cols[2]],
    c=gdf_pts[eig_cols[0]],
    cmap=cmap,
    norm=norm,
    s=30
)

ax2.set_xlabel("Eigenvector 1")
ax2.set_ylabel("Eigenvector 2")
ax2.set_zlabel("Eigenvector 3")
ax2.set_title("3D spectral space", fontsize=11)

# ---------- Colorbar ----------
cax = fig.add_subplot(gs[0, 2])
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    cax=cax
)
cbar.set_label("Eigenvector 1")

# =====================
# SAVE
# =====================
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.close()

print("Done:")
print(out_png)
