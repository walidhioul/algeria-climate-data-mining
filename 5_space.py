"""
Step 5 - Exploration of the SPACE dimension (exploration only, nothing is modified).

Questions answered here:
  1. What does each variable look like on a map (average over time)?
  2. How different are winter and summer maps (t2m)?
  3. How many valid cells really lie inside Algeria?
  4. Does surface pressure behave like an altitude map, and how does it relate to temperature?
  5. How is the variance of t2m split between "where" (space), "when" (season) and the rest?
  6. Where and when do the extreme runoff and precipitation values occur?

Set BORDER_FILE to the path of the Algeria GeoJSON file (simplemaps).
"""
import json
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.path import Path

BORDER_FILE = "dz.json"        # <- change to the real name/path of your Algeria GeoJSON

ds = xr.open_dataset("era5land_algeria.nc").load()

DISPLAY = {
    "t2m":  (1, -273.15, "deg C"),
    "d2m":  (1, -273.15, "deg C"),
    "slhf": (1 / 86400, 0, "W/m2 (daily total assumed)"),
    "ssr":  (1 / 86400, 0, "W/m2 (daily total assumed)"),
    "str":  (1 / 86400, 0, "W/m2 (daily total assumed)"),
    "sshf": (1 / 86400, 0, "W/m2 (daily total assumed)"),
    "ro":   (1000, 0, "mm (per day, to confirm)"),
    "e":    (1000, 0, "mm (per day, to confirm)"),
    "tp":   (1000, 0, "mm (per day, to confirm)"),
    "u10":  (1, 0, "m/s"),
    "v10":  (1, 0, "m/s"),
    "sp":   (1 / 100, 0, "hPa"),
}
ORDER = ["t2m", "d2m", "ssr", "str", "sshf", "slhf", "e", "tp", "ro", "u10", "v10", "sp"]


def show(name, da):
    scale, offset, unit = DISPLAY[name]
    return da * scale + offset, unit


# ---------------------------------------------------------------------------
# Algeria border (used to draw the outline and to know which cells are inside)
# ---------------------------------------------------------------------------
rings = []
if os.path.exists(BORDER_FILE):
    with open(BORDER_FILE, encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            rings.append(np.array(poly[0]))        # outer ring of each polygon
else:
    print(f"WARNING: {BORDER_FILE} not found. Maps will have no outline and every cell counts as inside.")


def draw_border(ax):
    for r in rings:
        ax.plot(r[:, 0], r[:, 1], "k-", lw=0.7)


lon2d, lat2d = np.meshgrid(ds["longitude"].values, ds["latitude"].values)
if rings:
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for r in rings:
        inside |= Path(r).contains_points(pts)
    inside = inside.reshape(lat2d.shape)
else:
    inside = np.ones(lat2d.shape, dtype=bool)

inside_da = xr.DataArray(inside, coords=[ds["latitude"], ds["longitude"]], dims=["latitude", "longitude"])
valid_da = ds["t2m"].isel(valid_time=0, drop=True).notnull()

# ---------------------------------------------------------------------------
# 3. How many valid cells are inside Algeria?
# ---------------------------------------------------------------------------
n_valid = int(valid_da.sum())
n_valid_inside = int((valid_da & inside_da).sum())
n_nan_inside = int((~valid_da & inside_da).sum())
print("=== Cells ===")
print(f"Valid (land) cells in the box:               {n_valid}")
print(f"Valid cells inside the Algeria polygon:      {n_valid_inside}  ({100 * n_valid_inside / n_valid:.1f}% of valid cells)")
print(f"Valid cells OUTSIDE Algeria (neighbours):    {n_valid - n_valid_inside}")
print(f"NaN (sea) cells inside the Algeria polygon:  {n_nan_inside}")

# ---------------------------------------------------------------------------
# 1. TIME-MEAN MAPS of the 12 variables
# ---------------------------------------------------------------------------
fig1, axes = plt.subplots(3, 4, figsize=(20, 12))
for ax, name in zip(axes.ravel(), ORDER):
    m, unit = show(name, ds[name].mean("valid_time"))
    m.plot.pcolormesh(ax=ax, robust=True, cbar_kwargs={"label": unit})
    draw_border(ax)
    ax.set_title(f"{name}: mean over 2015-2026")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
fig1.tight_layout()
fig1.savefig("step5_mean_maps.png", dpi=130)

# ---------------------------------------------------------------------------
# 2. SEASONAL MAPS of t2m: January, July and the difference
# ---------------------------------------------------------------------------
tc = ds["t2m"] - 273.15
jan = tc.where(ds["valid_time.month"] == 1).mean("valid_time", skipna=True)
jul = tc.where(ds["valid_time.month"] == 7).mean("valid_time", skipna=True)
fig2, axs = plt.subplots(1, 3, figsize=(20, 6))
for ax, da, title in zip(axs, [jan, jul, jul - jan], ["t2m, all Januaries", "t2m, all Julys", "July minus January"]):
    da.plot.pcolormesh(ax=ax, cbar_kwargs={"label": "deg C"})
    draw_border(ax)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_aspect("equal")
fig2.tight_layout()
fig2.savefig("step5_t2m_seasonal_maps.png", dpi=130)

# ---------------------------------------------------------------------------
# 4. Surface pressure as an altitude proxy, and its link with temperature
#    (cells inside Algeria only)
# ---------------------------------------------------------------------------
sel = valid_da & inside_da
sp_mean = (ds["sp"].mean("valid_time") / 100).where(sel)
t_mean = (ds["t2m"].mean("valid_time") - 273.15).where(sel)
sp_flat = sp_mean.values.ravel()
t_flat = t_mean.values.ravel()
lat_flat = lat2d.ravel()
ok = ~np.isnan(sp_flat) & ~np.isnan(t_flat)
corr = np.corrcoef(sp_flat[ok], t_flat[ok])[0, 1]
print(f"\nCorrelation between time-mean sp and time-mean t2m across cells (Algeria): {corr:.2f}")

fig3 = plt.figure(figsize=(8, 6))
sc = plt.scatter(sp_flat[ok], t_flat[ok], c=lat_flat[ok], s=3, cmap="viridis")
plt.colorbar(sc, label="latitude")
plt.xlabel("time-mean surface pressure (hPa)  [low pressure = high altitude]")
plt.ylabel("time-mean t2m (deg C)")
plt.title("One point per cell inside Algeria")
fig3.savefig("step5_sp_vs_t2m.png", dpi=150)

# ---------------------------------------------------------------------------
# 5. VARIANCE DECOMPOSITION of t2m (cells inside Algeria)
#    total = spatial (cells differ) + seasonal (months differ) + residual
# ---------------------------------------------------------------------------
sub = tc.where(sel)                                          # (time, lat, lon), NaN outside
cell_mean = sub.mean("valid_time")
total_var = float(sub.var(("valid_time", "latitude", "longitude")))
spatial_var = float(cell_mean.var(("latitude", "longitude")))
temporal_var = float(sub.var("valid_time").mean(("latitude", "longitude")))
clim = sub.groupby("valid_time.month").mean("valid_time")
resid = sub.groupby("valid_time.month") - clim
resid_var = float(resid.var("valid_time").mean(("latitude", "longitude")))
seasonal_var = temporal_var - resid_var
print("\n=== Variance of t2m (cells inside Algeria) ===")
print(f"Total variance:                                 {total_var:.1f}")
print(f"  between cells (space, 'where'):               {spatial_var:.1f}  ({100 * spatial_var / total_var:.0f}%)")
print(f"  seasonal cycle (time, 'when in the year'):    {seasonal_var:.1f}  ({100 * seasonal_var / total_var:.0f}%)")
print(f"  residual (what remains after both):           {resid_var:.1f}  ({100 * resid_var / total_var:.0f}%)")

# ---------------------------------------------------------------------------
# 6. WHERE AND WHEN are the extremes of ro and tp?
# ---------------------------------------------------------------------------
print("\n=== Extreme values ===")
for name in ["ro", "tp"]:
    idx = ds[name].argmax(dim=("valid_time", "latitude", "longitude"))
    cell = ds.isel(valid_time=int(idx["valid_time"]), latitude=int(idx["latitude"]), longitude=int(idx["longitude"]))
    lat, lon = float(cell["latitude"]), float(cell["longitude"])
    ins = bool(inside_da.sel(latitude=lat, longitude=lon))
    print(f"{name}: max = {float(cell[name]) * 1000:.2f} mm at lat={lat:.1f}, lon={lon:.1f}, "
          f"month={str(cell['valid_time'].values)[:7]}, inside Algeria={ins}, "
          f"sp={float(cell['sp']) / 100:.0f} hPa")

plt.show()