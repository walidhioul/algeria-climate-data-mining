"""
Step 4 - Exploration of the TIME dimension (exploration only, nothing is modified).

Questions answered here:
  1. How do the variables evolve month after month (national mean)?
  2. How big is the seasonal cycle (climatology)?
  3. Once the seasonal cycle is removed (anomalies), is there a trend?
  4. Do very different places (coast, highlands, Sahara, Hoggar) behave differently?
  5. Do the bumps in the t2m histogram come from the seasons?
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("era5land_algeria.nc").load()   # 273 MB: fine to keep in memory

# name: (scale, offset, label)  ->  value_shown = value * scale + offset  (display only)
DISPLAY = {
    "t2m":  (1, -273.15, "deg C"),
    "d2m":  (1, -273.15, "deg C"),
    "slhf": (1 / 86400, 0, "W/m2 (assumes daily total)"),
    "ssr":  (1 / 86400, 0, "W/m2 (assumes daily total)"),
    "str":  (1 / 86400, 0, "W/m2 (assumes daily total)"),
    "sshf": (1 / 86400, 0, "W/m2 (assumes daily total)"),
    "ro":   (1000, 0, "mm (per day, to confirm)"),
    "e":    (1000, 0, "mm (per day, to confirm)"),
    "tp":   (1000, 0, "mm (per day, to confirm)"),
    "u10":  (1, 0, "m/s"),
    "v10":  (1, 0, "m/s"),
    "sp":   (1 / 100, 0, "hPa"),
}
ORDER = ["t2m", "d2m", "ssr", "str", "sshf", "slhf", "e", "tp", "ro", "u10", "v10", "sp"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def show(name, da):
    """Readable version of a DataArray (display only)."""
    scale, offset, unit = DISPLAY[name]
    return da * scale + offset, unit


# ---------------------------------------------------------------------------
# 1. NATIONAL MONTHLY MEAN of each variable
#    (mean over all valid cells of the box, NaN sea cells are skipped).
#    Caution: the box also contains parts of neighbouring countries and the
#    mean is not area-weighted: it is an exploration tool, not a final result.
# ---------------------------------------------------------------------------
nat = {name: ds[name].mean(dim=("latitude", "longitude")) for name in ORDER}

fig1, axes = plt.subplots(4, 3, figsize=(16, 13), sharex=True)
for ax, name in zip(axes.ravel(), ORDER):
    y, unit = show(name, nat[name])
    ax.plot(y["valid_time"].values, y.values, lw=1)
    ax.set_title(name)
    ax.set_ylabel(unit, fontsize=8)
    ax.grid(alpha=0.3)
fig1.suptitle("Box mean of each variable, month by month (2015 - 2026)")
fig1.tight_layout()
fig1.savefig("step4_national_monthly_means.png", dpi=150)

# ---------------------------------------------------------------------------
# 2. CLIMATOLOGY and 3. ANOMALIES for the national t2m
#    climatology = average of all Januaries, all Februaries, ...
#    anomaly     = value - climatology of its calendar month
# ---------------------------------------------------------------------------
t = nat["t2m"] - 273.15
clim = t.groupby("valid_time.month").mean()
clim_std = t.groupby("valid_time.month").std()
anom = t.groupby("valid_time.month") - clim

amplitude = float(clim.max() - clim.min())
print("=== National t2m (deg C): climatology ===")
for m in range(12):
    print(f"{MONTHS[m]}: mean={float(clim[m]):6.2f}   std across years={float(clim_std[m]):5.2f}")
print(f"Seasonal amplitude (hottest month - coldest month): {amplitude:.2f} deg C")
print(f"Std of the anomalies (year-to-year variability):    {float(anom.std()):.2f} deg C")

# linear trend of the anomalies (careful: only ~11.75 years of data)
x_years = np.arange(len(anom)) / 12.0
coef, cov = np.polyfit(x_years, anom.values, 1, cov=True)
slope, slope_se = coef[0], float(np.sqrt(cov[0, 0]))
print(f"Linear trend of the anomalies: {slope * 10:.2f} deg C per decade "
      f"(+/- {2 * slope_se * 10:.2f}, about 95% interval, short period!)")

fig2, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5))
axA.errorbar(range(1, 13), clim.values, yerr=clim_std.values, marker="o", capsize=3)
axA.set_xticks(range(1, 13))
axA.set_xticklabels(MONTHS)
axA.set_title("National t2m: climatology (mean +/- std across years)")
axA.set_ylabel("deg C")
axA.grid(alpha=0.3)

axB.plot(anom["valid_time"].values, anom.values, lw=1, label="anomaly")
axB.plot(anom["valid_time"].values, coef[1] + coef[0] * x_years, "r--", label="linear trend")
axB.axhline(0, color="k", lw=0.5)
axB.set_title("National t2m: anomalies (seasonal cycle removed)")
axB.set_ylabel("deg C")
axB.legend()
axB.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("step4_t2m_climatology_anomalies.png", dpi=150)

# ---------------------------------------------------------------------------
# 4. FOUR CONTRASTING PLACES (nearest VALID cell to each approximate position)
# ---------------------------------------------------------------------------
SITES = {
    "Algiers (coast)": (36.75, 3.06),
    "Batna (highlands)": (35.56, 6.17),
    "Adrar (Sahara)": (27.87, -0.29),
    "Tamanrasset (Hoggar)": (22.79, 5.52),
}
valid = ds["t2m"].isel(valid_time=0, drop=True).notnull()   # drop=True: no leftover time coordinate


def nearest_valid(lat, lon):
    d = (ds["latitude"] - lat) ** 2 + (ds["longitude"] - lon) ** 2   # 2D: (latitude, longitude)
    d = d.where(valid)                                                # ignore sea cells
    idx = d.argmin(dim=("latitude", "longitude"))
    return ds.isel(latitude=int(idx["latitude"]), longitude=int(idx["longitude"]))


fig3, (axC, axD) = plt.subplots(1, 2, figsize=(16, 5))
print("\n=== Contrasting cells ===")
for label, (lat, lon) in SITES.items():
    cell = nearest_valid(lat, lon)
    s = cell["t2m"] - 273.15
    c = s.groupby("valid_time.month").mean()
    print(f"{label:22s} cell lat={float(cell['latitude']):.1f} lon={float(cell['longitude']):.1f} | "
          f"mean sp={float(cell['sp'].mean()) / 100:6.1f} hPa | "
          f"t2m: Jan={float(c[0]):5.1f}  Jul={float(c[6]):5.1f}  amplitude={float(c.max() - c.min()):5.1f}")
    axC.plot(range(1, 13), c.values, marker="o", label=label)
    axD.plot(s["valid_time"].values, s.values, lw=0.8, label=label)
axC.set_xticks(range(1, 13))
axC.set_xticklabels(MONTHS)
axC.set_title("t2m climatology of four contrasting cells")
axC.set_ylabel("deg C")
axC.legend()
axC.grid(alpha=0.3)
axD.set_title("t2m month by month, same four cells")
axD.set_ylabel("deg C")
axD.grid(alpha=0.3)
fig3.tight_layout()
fig3.savefig("step4_contrasting_cells.png", dpi=150)

# ---------------------------------------------------------------------------
# 5. DO THE BUMPS OF THE t2m HISTOGRAM COME FROM THE SEASONS?
# ---------------------------------------------------------------------------
month = ds["valid_time.month"]
tc = ds["t2m"] - 273.15
winter = tc.where(month.isin([12, 1, 2])).values.ravel()
summer = tc.where(month.isin([6, 7, 8])).values.ravel()
winter = winter[~np.isnan(winter)]
summer = summer[~np.isnan(summer)]
print(f"\nt2m winter (Dec-Jan-Feb): mean={winter.mean():.1f}  min={winter.min():.1f}  max={winter.max():.1f}")
print(f"t2m summer (Jun-Jul-Aug): mean={summer.mean():.1f}  min={summer.min():.1f}  max={summer.max():.1f}")

fig4 = plt.figure(figsize=(9, 5))
plt.hist(winter, bins=80, alpha=0.6, label="Dec-Jan-Feb")
plt.hist(summer, bins=80, alpha=0.6, label="Jun-Jul-Aug")
plt.title("t2m (deg C): winter vs summer, all valid cells")
plt.xlabel("deg C")
plt.ylabel("count")
plt.legend()
fig4.savefig("step4_t2m_winter_vs_summer.png", dpi=150)

plt.show()