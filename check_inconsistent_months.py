"""
Diagnostic: which months look inconsistent with the rest of the record? (exploration only, nothing is modified)

For each variable:
  1. national (box) mean of every month,
  2. compare each month with the SAME calendar month in the other years, using a robust z-score
         z = (value - median) / (1.4826 * MAD)      (median and MAD are not distorted by a few bad years)
  3. flag the months with |z| > THRESHOLD and list them as runs of consecutive months.
Also prints, for the flagged variables, how much of the map is affected in the worst month.
"""
import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")
FILE = "era5land_algeria.nc"
THRESHOLD = 5
VARS = ["t2m", "d2m", "ssr", "str", "sshf", "slhf", "e", "tp", "ro", "u10", "v10", "sp"]

ds = xr.open_dataset(FILE).load()
nat = ds[VARS].mean(("latitude", "longitude")).to_dataframe()[VARS]       # (141 months x 12 variables)

z = pd.DataFrame(index=nat.index, columns=VARS, dtype=float)
for v in VARS:
    for m in range(1, 13):
        idx = nat.index.month == m
        x = nat.loc[idx, v]
        med = x.median()
        mad = 1.4826 * (x - med).abs().median()
        z.loc[idx, v] = (x - med) / max(mad, 1e-12)

flag = z.abs() > THRESHOLD
print(f"=== Months with |robust z| > {THRESHOLD} (national mean vs same calendar month in other years) ===")
print(z[flag.any(axis=1)].round(1).where(flag[flag.any(axis=1)]).dropna(how="all").fillna("").to_string())

print("\n=== Summary per variable ===")
print("(isolated flagged months in tp, ro, slhf, e are usually real extreme events;")
print(" a LONG RUN of consecutive flagged months in a smooth variable is the sign of an inconsistent period)")
runs = {}
for v in VARS:
    f = flag[v].values
    best, cur, best_end = 0, 0, None
    for i, val in enumerate(f):
        cur = cur + 1 if val else 0
        if cur > best:
            best, best_end = cur, i
    runs[v] = best
    if f.any():
        months = flag[v][flag[v]].index
        extra = ""
        if best >= 3:
            extra = f"  <-- run of {best} consecutive months ending {flag.index[best_end]:%Y-%m}"
        print(f"{v:5s}: {f.sum():3d} flagged months ({months.min():%Y-%m} ... {months.max():%Y-%m}){extra}")
    else:
        print(f"{v:5s}: nothing flagged")

# spatial extent in the most deviating month of each flagged variable
print("\n=== Variables with a long run: how much of the map is affected in the worst month ===")
print("(share of cells outside the 1%-99% range of the same calendar month in the other years)")
t = pd.DatetimeIndex(ds["valid_time"].values)
for v in [c for c in VARS if runs[c] >= 3]:
    worst = z[v].abs().idxmax()
    same_month_others = np.where((t.month == worst.month) & (t != worst))[0]
    ref = ds[v].isel(valid_time=same_month_others)
    lo, hi = ref.quantile(0.01, "valid_time"), ref.quantile(0.99, "valid_time")
    cur = ds[v].isel(valid_time=int(np.where(t == worst)[0][0]))
    valid = cur.notnull()
    outside = ((cur < lo) | (cur > hi)) & valid
    print(f"{v:5s}: worst month {worst:%Y-%m} -> {100 * float(outside.sum() / valid.sum()):.0f}% of the cells are outside the usual range")