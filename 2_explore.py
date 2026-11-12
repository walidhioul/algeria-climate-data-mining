import xarray as xr

ds = xr.open_dataset("era5land_algeria.nc")
print(ds)

# ---------------------------------------------------------------------------
# CHECK 1: units and descriptions of every variable
# (exploration only: nothing is modified, the dataset stays as downloaded)
# ---------------------------------------------------------------------------
print("\n=== CHECK 1: units ===")
for name in ds.data_vars:
    attrs = ds[name].attrs
    print(f"{name:5s} units={attrs.get('units')!s:10s} long_name={attrs.get('long_name')}")

# ---------------------------------------------------------------------------
# CHECK 2: data version per month (expver)
# ---------------------------------------------------------------------------
print("\n=== CHECK 2: expver ===")
expver = ds["expver"].values
times = ds["valid_time"].values
for v in sorted(set(expver)):
    months = times[expver == v]
    print(f"expver={v}: {len(months)} months, from {str(months.min())[:7]} to {str(months.max())[:7]}")

# ---------------------------------------------------------------------------
# CHECK 3: missing values (NaN) per variable
# ---------------------------------------------------------------------------
print("\n=== CHECK 3: NaN count per variable ===")
for name in ds.data_vars:
    n_nan = int(ds[name].isnull().sum().item())
    total = ds[name].size
    print(f"{name:5s} NaN={n_nan:>10d} / {total}  ({100 * n_nan / total:.2f}%)")

# Is the NaN pattern the same in every month? (cells with NaN in the first month)
first_month_nan = int(ds["t2m"].isel(valid_time=0).isnull().sum().item())
n_cells = ds["t2m"].isel(valid_time=0).size
print(f"\nt2m: NaN cells in the first month = {first_month_nan} / {n_cells}")
cells_nan_all_months = int(ds["t2m"].isnull().all("valid_time").sum().item())
print(f"t2m: cells that are NaN in ALL months = {cells_nan_all_months}")

# ---------------------------------------------------------------------------
# CHECK 4: map of t2m for one month (July 2025)
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

print("\n=== CHECK 4: map of t2m, July 2025 ===")
t2m_month = ds["t2m"].sel(valid_time="2025-07-01")
print("t2m units in the file:", ds["t2m"].attrs.get("units"))

# The plot uses a Celsius copy only if the file is in Kelvin (the dataset itself is not changed)
if ds["t2m"].attrs.get("units") == "K":
    t2m_month = t2m_month - 273.15
    label = "t2m (deg C), July 2025 - converted for plotting only"
else:
    label = "t2m, July 2025"

plt.figure(figsize=(9, 7))
t2m_month.plot(cmap="inferno", cbar_kwargs={"label": label})
plt.title("2m temperature, July 2025 (raw grid 0.1 deg)")
plt.savefig("check4_t2m_map_2025_07.png", dpi=150, bbox_inches="tight")
plt.show()