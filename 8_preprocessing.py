"""
Step 8 - PREPROCESSING and FEATURE ENGINEERING

Input : era5land_algeria.nc (raw ERA5-Land download, untouched)
Output (folder data/processed):
    cells.csv                  one row per kept grid cell
    era5_dz_aggregated.csv     long table: one row per (cell, month), readable units
    supervised_dataset.csv     features + next-month t2m target + 3 classes + train/test flag
    class_edges.json           class boundaries (computed on the TRAINING targets only)
    clustering_dataset.csv     one row per (cell, year): monthly profiles + engineered features

What this script does, in order (the order matters, see the justification next to each step):
    1. Reduce spatial resolution by a factor of 4 (mean of 4x4 blocks, land cells only)
    2. Keep only cells inside Algeria (border polygon) with enough land
    3. REPAIR the inconsistent period found in the exploration (ssr, str, sshf from 2022-09 to 2024-02)
       -> done BEFORE any statistic is computed, otherwise the bad values contaminate every number
    4. Convert units to readable ones
    5. Feature diagnostics (variability, skewness, outliers) for the report
    6. Build the supervised dataset (target = t2m of the NEXT month, 3 equal-width classes)
    7. Chronological train/test split (no shuffling); class edges from the TRAINING targets only
    8. Build the clustering dataset (one instance = one cell in one year), from the repaired data
Skew correction (signed log) and standardisation are done in 9_supervised.py, fitted on the TRAIN set only.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path

# ----------------------------- settings -----------------------------------
FILE = "era5land_algeria.nc"
BORDER_FILE = "dz.json"            # Algeria GeoJSON (simplemaps)
OUT_DIR = "data/processed"
AGG = 4                            # spatial aggregation factor required by the brief
MIN_LAND_FRACTION = 0.5            # a 4x4 block needs >= 50% land cells to be kept
TRAIN_END_TARGET = "2023-12-01"    # last TARGET month inside the training set
TEST_START_FEATURE = "2024-01-01"  # first FEATURE month inside the test set
LAST_FULL_YEAR = 2025              # years with 12 months (2026 is incomplete)

# Inconsistent period found with check_inconsistent_months.py (robust z-score vs the same calendar month):
# ssr and str flagged for 18 consecutive months, sshf for 15, all cells outside their usual range.
BAD_START = "2022-09-01"
BAD_END = "2024-02-01"
BAD_VARS = ["ssr", "str", "sshf"]  # add "slhf", "e" here if the check with a lower threshold flags them too
SKEWED = ["ro", "tp", "e", "slhf"]
os.makedirs(OUT_DIR, exist_ok=True)

VARS = ["d2m", "t2m", "slhf", "ssr", "str", "sshf", "ro", "e", "u10", "v10", "sp", "tp"]
FEATURES = ["d2m", "slhf", "ssr", "str", "sshf", "ro", "e", "u10", "v10", "sp", "tp"]  # from the brief

# ----------------------------- 1. load raw ---------------------------------
ds = xr.open_dataset(FILE).load()
ds = ds.drop_vars([v for v in ("expver", "number") if v in ds.coords])
print("Raw grid:", dict(ds.sizes))

# Make the grid size a multiple of AGG by dropping the first rows/columns
# (north / west edge of the box: sea and outside Algeria, no land of interest is lost)
n_drop_lat = ds.sizes["latitude"] % AGG
n_drop_lon = ds.sizes["longitude"] % AGG
ds = ds.isel(latitude=slice(n_drop_lat, None), longitude=slice(n_drop_lon, None))
print(f"Dropped {n_drop_lat} northern rows and {n_drop_lon} western columns -> {dict(ds.sizes)}")

# ----------------------------- 2. aggregate x4 -----------------------------
land = ds["t2m"].notnull().astype("float32").isel(valid_time=0, drop=True)
land_fraction = land.coarsen(latitude=AGG, longitude=AGG).mean()      # share of land cells in each block
coarse = ds.coarsen(latitude=AGG, longitude=AGG).mean()               # NaN (sea) cells are skipped
print("Aggregated grid:", dict(coarse.sizes))

# ----------------------------- 3. Algeria mask -----------------------------
lon2d, lat2d = np.meshgrid(coarse["longitude"].values, coarse["latitude"].values)
inside = np.ones(lat2d.shape, dtype=bool)
if os.path.exists(BORDER_FILE):
    with open(BORDER_FILE, encoding="utf-8") as f:
        gj = json.load(f)
    pts = np.column_stack([lon2d.ravel(), lat2d.ravel()])
    inside = np.zeros(len(pts), dtype=bool)
    for feat in gj["features"]:
        geom = feat["geometry"]
        polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
        for poly in polys:
            inside |= Path(np.array(poly[0])).contains_points(pts)
    inside = inside.reshape(lat2d.shape)
else:
    print(f"WARNING: {BORDER_FILE} not found, no border mask applied")

inside_da = xr.DataArray(inside, coords=[coarse["latitude"], coarse["longitude"]], dims=["latitude", "longitude"])
print("Cells whose centre is inside Algeria, by minimum land fraction of the 4x4 block:")
for thr in (0.25, 0.5, 0.75, 1.0):
    print(f"  land fraction >= {thr:.2f}: {int((inside_da & (land_fraction >= thr)).sum())} cells")
keep = inside_da & (land_fraction >= MIN_LAND_FRACTION)
print(f"Cells: {keep.size} in the aggregated box, {int(keep.sum())} kept "
      f"(centre inside Algeria and >= {int(MIN_LAND_FRACTION * 100)}% land)")

# ----------------------------- 3. repair the inconsistent period ----------------
# Only ssr, str and sshf are affected (t2m = the target, d2m, winds and pressure are not).
# Repair = replace the value of each cell by the MEDIAN of the same calendar month over the clean months
# BEFORE the problem (2015-01 .. 2022-08). Those months are older than the test period: no leakage.
times = pd.DatetimeIndex(coarse["valid_time"].values)
bad_idx = np.where((times >= pd.Timestamp(BAD_START)) & (times <= pd.Timestamp(BAD_END)))[0]
ref_idx = np.where(times < pd.Timestamp(BAD_START))[0]
before = {v: coarse[v].mean(("latitude", "longitude")).values.copy() for v in BAD_VARS}
for v in BAD_VARS:
    clim = coarse[v].isel(valid_time=ref_idx).groupby("valid_time.month").median("valid_time")
    arr = coarse[v].values.copy()
    for i in bad_idx:
        arr[i] = clim.sel(month=times[i].month).values
    coarse[v] = coarse[v].copy(data=arr)
print(f"\nRepaired {BAD_VARS} for {len(bad_idx)} months ({BAD_START[:7]} .. {BAD_END[:7]}) "
      f"with the pre-problem median of each cell and calendar month")
fig, axes = plt.subplots(1, len(BAD_VARS), figsize=(6 * len(BAD_VARS), 4))
for ax, v in zip(np.atleast_1d(axes), BAD_VARS):
    ax.plot(times, before[v], label="before repair", lw=1)
    ax.plot(times, coarse[v].mean(("latitude", "longitude")).values, label="after repair", lw=1)
    ax.set_title(f"{v}: box mean per month (raw units)")
    ax.legend()
    ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("step8_repair_check.png", dpi=150)

# ----------------------------- 4. readable units ---------------------------
coarse["t2m"] = coarse["t2m"] - 273.15            # K -> deg C
coarse["d2m"] = coarse["d2m"] - 273.15
for v in ("slhf", "ssr", "str", "sshf"):
    coarse[v] = coarse[v] / 86400.0               # J/m2 per day -> W/m2 (daily total assumed)
for v in ("ro", "e", "tp"):
    coarse[v] = coarse[v] * 1000.0                # m -> mm
coarse["sp"] = coarse["sp"] / 100.0               # Pa -> hPa

# ----------------------------- 5. long table -------------------------------
df = coarse[VARS].where(keep).to_dataframe().reset_index()
df = df.rename(columns={"valid_time": "time"})
df = df.dropna(subset=["t2m"])
# a kept cell must have no missing value at all
bad = df[df[VARS].isna().any(axis=1)][["latitude", "longitude"]].drop_duplicates()
if len(bad):
    print(f"Removing {len(bad)} cells that still contain NaN")
    df = df.merge(bad.assign(_bad=1), on=["latitude", "longitude"], how="left")
    df = df[df["_bad"].isna()].drop(columns="_bad")
df["cell_id"] = df.groupby(["latitude", "longitude"]).ngroup()
df["year"] = df["time"].dt.year
df["month"] = df["time"].dt.month
df = df.sort_values(["cell_id", "time"]).reset_index(drop=True)
print(f"Long table: {len(df)} rows = {df['cell_id'].nunique()} cells x {df['time'].nunique()} months")

df["repaired"] = (df["time"] >= pd.Timestamp(BAD_START)) & (df["time"] <= pd.Timestamp(BAD_END))

cells = df[["cell_id", "latitude", "longitude"]].drop_duplicates().sort_values("cell_id")
cells.to_csv(f"{OUT_DIR}/cells.csv", index=False)
df.to_csv(f"{OUT_DIR}/era5_dz_aggregated.csv", index=False)

# ----------------------------- 5. feature diagnostics (for the report) -----
def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def tukey_share(x):
    """Share of points flagged by the 1.5 x IQR rule (it assumes a symmetric distribution)."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return float(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean())


diag = []
for v in FEATURES:
    x = df[v].values
    r = {"feature": v, "std": x.std(), "coef_variation": x.std() / abs(x.mean()) if x.mean() != 0 else np.nan,
         "zero_share": float((x == 0).mean()), "skew": pd.Series(x).skew(), "outliers_IQR_share": tukey_share(x)}
    if v in SKEWED:
        y = signed_log1p(x)
        r["skew_after_log"] = pd.Series(y).skew()
        r["outliers_IQR_share_after_log"] = tukey_share(y)
    diag.append(r)
diag = pd.DataFrame(diag).set_index("feature")
pd.set_option("display.width", 220)
print("\n=== Feature diagnostics on the repaired, aggregated data ===")
print(diag.round(3).to_string())
diag.round(4).to_csv(f"{OUT_DIR}/feature_diagnostics.csv")

# ----------------------------- 6. supervised dataset -----------------------
g = df.groupby("cell_id")
df["target_t2m"] = g["t2m"].shift(-1)          # t2m of the NEXT month, same cell
df["target_time"] = g["time"].shift(-1)
sup = df.dropna(subset=["target_t2m"]).copy()
gap = (sup["target_time"].dt.year * 12 + sup["target_time"].dt.month) - (sup["time"].dt.year * 12 + sup["time"].dt.month)
assert (gap == 1).all(), "months are not consecutive"
sup = sup.rename(columns={"t2m": "t2m_now"})   # kept ONLY for baselines, it is NOT a feature (brief)

train_mask = sup["target_time"] <= pd.Timestamp(TRAIN_END_TARGET)
test_mask = sup["time"] >= pd.Timestamp(TEST_START_FEATURE)
sup["split"] = np.where(train_mask, "train", np.where(test_mask, "test", "gap"))
sup = sup[sup["split"] != "gap"].copy()       # the 1-month gap avoids a train label living in the test period

# 3 equal-width classes, boundaries computed on the TRAINING targets only
lo = float(sup.loc[sup["split"] == "train", "target_t2m"].min())
hi = float(sup.loc[sup["split"] == "train", "target_t2m"].max())
edges = np.linspace(lo, hi, 4)
inner = edges[1:3]
sup["target_class"] = np.digitize(sup["target_t2m"], inner)     # 0 = cold, 1 = medium, 2 = hot
sup["now_class"] = np.digitize(sup["t2m_now"], inner)           # for the persistence baseline
with open(f"{OUT_DIR}/class_edges.json", "w") as f:
    json.dump({"train_min": lo, "train_max": hi, "edges": edges.tolist(),
               "class_names": ["cold", "medium", "hot"]}, f, indent=2)

cols = (["cell_id", "latitude", "longitude", "time", "target_time", "year", "month"]
        + FEATURES + ["t2m_now", "target_t2m", "target_class", "now_class", "repaired", "split"])
sup = sup[cols]
sup.to_csv(f"{OUT_DIR}/supervised_dataset.csv", index=False)

print("\n=== Supervised dataset ===")
print(f"Class edges (deg C): {np.round(edges, 2).tolist()}")
for s in ("train", "test"):
    part = sup[sup["split"] == s]
    dist = part["target_class"].value_counts(normalize=True).sort_index().round(3).to_dict()
    print(f"{s:5s}: {len(part):7d} rows, features from {part['time'].min():%Y-%m} to {part['time'].max():%Y-%m}, "
          f"class shares {dist}")

# ----------------------------- 7. clustering dataset -----------------------
d = df[df["year"] <= LAST_FULL_YEAR]
t = d.pivot_table(index=["cell_id", "year"], columns="month", values="t2m")
s = d.pivot_table(index=["cell_id", "year"], columns="month", values="str")
t.columns = [f"t2m_m{m:02d}" for m in t.columns]
s.columns = [f"str_m{m:02d}" for m in s.columns]
clu = t.join(s)

# engineered features describing the year and the long-term change
clu["t2m_mean"] = t.mean(axis=1)
clu["t2m_amp"] = t.max(axis=1) - t.min(axis=1)                       # seasonal amplitude of the year
clu["str_mean"] = s.mean(axis=1)
clu["t2m_anom"] = clu["t2m_mean"] - clu.groupby("cell_id")["t2m_mean"].transform("mean")  # vs the cell's own average year

def slope_per_decade(column):
    """Linear trend of the annual mean over the years, per cell (deg C or W/m2 per decade)."""
    y = clu[column].unstack("year")
    x = y.columns.values.astype(float)
    xc = x - x.mean()
    sl = (y.sub(y.mean(axis=1), axis=0).values @ xc) / (xc ** 2).sum() * 10
    return pd.Series(sl, index=y.index)

clu["t2m_slope"] = clu.index.get_level_values("cell_id").map(slope_per_decade("t2m_mean"))
clu["str_slope"] = clu.index.get_level_values("cell_id").map(slope_per_decade("str_mean"))
clu = clu.reset_index().merge(cells, on="cell_id")
clu.to_csv(f"{OUT_DIR}/clustering_dataset.csv", index=False)

print("\n=== Clustering dataset ===")
print(f"{len(clu)} instances = {clu['cell_id'].nunique()} cells x {clu['year'].nunique()} years "
      f"({clu['year'].min()}-{clu['year'].max()}), {clu.shape[1] - 4} feature columns")
rep_months = d[d["repaired"]].groupby("year")["time"].nunique()
print("Months per year whose str values were repaired (they enter the clustering profiles):", rep_months.to_dict())
print("Saved everything in", OUT_DIR)