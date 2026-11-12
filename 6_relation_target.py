"""
Step 6-7 - EXPLORATION of relationships between variables and of the target (exploration only).

Questions answered here:
  1. Which variables are correlated (raw data)?  Which stay correlated once the seasonal cycle is removed?
  2. What do the main relationships look like (scatter / density plots)?
  3. Does the surface energy budget close?  (ssr + str + sshf + slhf should be close to 0)
  4. How would the 3 equal-width temperature classes be distributed? (overall, by month, by year)
  5. How well do the trivial baselines work?
       - "predict the same class as this month"
       - "predict the most common class for this calendar month"

For the final models, the class edges are recomputed on the TRAINING period only (see 8_preprocessing.py).
Here the edges use the whole range, only to look at the data.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


FILE = "era5land_algeria.nc"
ORDER = ["t2m", "d2m", "ssr", "str", "sshf", "slhf", "e", "tp", "ro", "u10", "v10", "sp"]
rng = np.random.default_rng(0)

ds = xr.open_dataset(FILE).load()
valid = ds["t2m"].isel(valid_time=0, drop=True).notnull().values          # (lat, lon) land mask
print(f"{valid.sum()} valid cells x {ds.sizes['valid_time']} months")

# ---------------------------------------------------------------------------
# 1. CORRELATIONS: raw values, then values with the seasonal cycle removed
# ---------------------------------------------------------------------------
raw = pd.DataFrame({v: ds[v].values[:, valid].ravel() for v in ORDER})
anom = {}
for v in ORDER:
    g = ds[v].groupby("valid_time.month")
    a = g - g.mean("valid_time")                                          # minus each cell's own calendar-month mean
    anom[v] = a.values[:, valid].ravel()
anom = pd.DataFrame(anom)
assert len(anom) == len(raw), "anomaly table must have the same number of rows as the raw table"

corr_raw = raw.corr()                                                      # Pearson on all values
sample_idx = rng.choice(len(raw), 300_000, replace=False)
corr_spear = raw.iloc[sample_idx].corr(method="spearman")                  # Spearman on a sample (robust to skew)
corr_anom = anom.corr()

pd.set_option("display.width", 220)
print("\n=== Pearson correlation, raw values ===")
print(corr_raw.round(2).to_string())
pairs = [(a, b, corr_raw.loc[a, b]) for i, a in enumerate(ORDER) for b in ORDER[i + 1:] if abs(corr_raw.loc[a, b]) >= 0.9]
print("\nPairs with |r| >= 0.9 (raw):", [(a, b, round(float(r), 2)) for a, b, r in pairs])
print("\n=== Correlation of t2m with each variable ===")
cmp = pd.DataFrame({"pearson_raw": corr_raw["t2m"], "spearman_raw": corr_spear["t2m"], "pearson_anomalies": corr_anom["t2m"]})
print(cmp.drop("t2m").round(2).to_string())

KEY_PAIRS = [("t2m", "d2m"), ("ssr", "sshf"), ("ssr", "t2m"), ("sshf", "t2m"), ("slhf", "e"), ("tp", "e"),
             ("tp", "ro"), ("ssr", "str"), ("str", "d2m"), ("sp", "t2m")]
rows = []
for a, b in KEY_PAIRS:
    r_raw, r_anom = corr_raw.loc[a, b], corr_anom.loc[a, b]
    if abs(r_raw) < 0.5:
        verdict = "weak"
    elif abs(r_anom) >= 0.5:
        verdict = "strong, survives without the seasons -> real physical link"
    else:
        verdict = "strong only through the seasons (shared cycle)"
    rows.append({"pair": f"{a} - {b}", "r_raw": r_raw, "r_spearman": corr_spear.loc[a, b], "r_anomalies": r_anom, "reading": verdict})
print("\n=== Key pairs: raw correlation vs correlation once the seasonal cycle is removed ===")
print(pd.DataFrame(rows).round(2).to_string(index=False))

fig1, axes = plt.subplots(1, 2, figsize=(17, 7))
for ax, c, title in zip(axes, [corr_raw, corr_anom], ["Pearson, raw values", "Pearson, seasonal cycle removed (anomalies)"]):
    im = ax.imshow(c.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, rotation=45)
    ax.set_yticks(range(len(ORDER))); ax.set_yticklabels(ORDER)
    for i in range(len(ORDER)):
        for j in range(len(ORDER)):
            ax.text(j, i, f"{c.values[i, j]:.1f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
fig1.colorbar(im, ax=axes, shrink=0.7)
fig1.savefig("step6_correlations.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------------------
# 2. KEY RELATIONSHIPS (density plots on a sample)
# ---------------------------------------------------------------------------
s = raw.iloc[sample_idx].copy()
s["t2m"] -= 273.15
s["d2m"] -= 273.15
pairs_plot = [("d2m", "t2m"), ("ssr", "sshf"), ("e", "slhf"), ("tp", "ro"), ("sp", "t2m"),
              ("ssr", "t2m"), ("tp", "e"), ("str", "d2m"), ("ssr", "str")]
fig2, axes = plt.subplots(3, 3, figsize=(17, 13))
for ax, (a, b) in zip(axes.ravel(), pairs_plot):
    ax.hexbin(s[a], s[b], gridsize=60, bins="log", cmap="viridis", mincnt=1)
    ax.set_xlabel(a); ax.set_ylabel(b); ax.set_title(f"{b} vs {a}  (r = {corr_raw.loc[a, b]:.2f})")
fig2.tight_layout()
fig2.savefig("step6_scatter.png", dpi=150)

# ---------------------------------------------------------------------------
# 2b. SURFACE ENERGY BUDGET (assuming the flux values are daily totals in J/m2: divide by 86400 -> W/m2)
#     ERA5 sign convention: downward flux is positive, so a surface that heats the air has sshf < 0.
#     Energy gained  : net radiation  = ssr + str
#     Energy leaving : turbulent flux = sshf + slhf   (negative numbers)
#     Budget          : ssr + str + sshf + slhf = what goes into the ground (about 0 on average)
# ---------------------------------------------------------------------------
flux = {v: ds[v].values[:, valid] / 86400.0 for v in ("ssr", "str", "sshf", "slhf")}
net_rad = flux["ssr"] + flux["str"]
turb = flux["sshf"] + flux["slhf"]
resid = net_rad + turb
months_all = np.repeat(pd.DatetimeIndex(ds["valid_time"].values).month.values[:, None], resid.shape[1], axis=1)
print("\n=== Energy budget (W/m2, mean over all valid cells and months) ===")
print(f"ssr {flux['ssr'].mean():8.1f} | str {flux['str'].mean():8.1f} | net radiation (ssr+str) {net_rad.mean():8.1f}")
print(f"sshf {flux['sshf'].mean():7.1f} | slhf {flux['slhf'].mean():7.1f} | turbulent (sshf+slhf) {turb.mean():8.1f}")
print(f"Residual (ssr+str+sshf+slhf) = {resid.mean():.1f} W/m2  ({100 * resid.mean() / flux['ssr'].mean():.1f}% of ssr), "
      f"std {resid.std():.1f}")
print(f"Share of the heat given back to the atmosphere as sensible heat: "
      f"{100 * flux['sshf'].mean() / turb.mean():.0f}%  (rest = evaporation)")
print(f"Correlation net radiation vs -(turbulent flux): {np.corrcoef(net_rad.ravel()[::7], -turb.ravel()[::7])[0, 1]:.2f}")
resid_month = pd.Series(resid.ravel()).groupby(months_all.ravel()).mean()
print("Mean residual by calendar month (W/m2; positive = ground storing heat):\n", resid_month.round(1).to_string())

fig4, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5))
sub_i = rng.choice(net_rad.size, min(300_000, net_rad.size), replace=False)
axA.hexbin(net_rad.ravel()[sub_i], -turb.ravel()[sub_i], gridsize=60, bins="log", cmap="viridis", mincnt=1)
lim = [min(net_rad.min(), (-turb).min()), max(net_rad.max(), (-turb).max())]
axA.plot(lim, lim, "r--", label="1:1 line (budget closed)")
axA.set_xlabel("net radiation ssr + str (W/m2)"); axA.set_ylabel("- (sshf + slhf) (W/m2)")
axA.set_title("Energy gained vs energy given back"); axA.legend()
resid_month.plot.bar(ax=axB)
axB.set_title("Residual of the budget by calendar month (W/m2)"); axB.set_xlabel("month")
fig4.tight_layout()
fig4.savefig("step6_energy_budget.png", dpi=150)

# ---------------------------------------------------------------------------
# 3. TARGET: next-month t2m, 3 equal-width classes (exploration edges)
# ---------------------------------------------------------------------------
t2m = ds["t2m"].values - 273.15                                            # (time, lat, lon)
y_now = t2m[:-1][:, valid]                                                 # month t
y_next = t2m[1:][:, valid]                                                 # month t+1 = the target
lo, hi = np.nanmin(y_next), np.nanmax(y_next)
edges = np.linspace(lo, hi, 4)
cls_next = np.digitize(y_next, edges[1:3])
cls_now = np.digitize(y_now, edges[1:3])
names = ["cold", "medium", "hot"]
print(f"\nExploration edges (deg C): {np.round(edges, 2).tolist()}")
print("Class shares overall:", dict(zip(names, np.round(np.bincount(cls_next.ravel(), minlength=3) / cls_next.size, 3).tolist())))

times = pd.DatetimeIndex(ds["valid_time"].values[1:])                       # target months
n_cells = cls_next.shape[1]
tab = pd.DataFrame({"month": np.repeat(times.month, n_cells), "year": np.repeat(times.year, n_cells), "cls": cls_next.ravel()})
by_month = pd.crosstab(tab["month"], tab["cls"], normalize="index").rename(columns=dict(enumerate(names)))
by_year = pd.crosstab(tab["year"], tab["cls"], normalize="index").rename(columns=dict(enumerate(names)))
print("\nClass shares by TARGET calendar month:\n", by_month.round(2).to_string())
print("\nClass shares by TARGET year:\n", by_year.round(2).to_string())

# ---------------------------------------------------------------------------
# 4. TRIVIAL BASELINE: persistence (next class = this month's class)
# ---------------------------------------------------------------------------
same = (cls_now == cls_next)
acc_month = pd.Series(same.mean(axis=1), index=times.month).groupby(level=0).mean()
print(f"\nPersistence accuracy overall: {same.mean():.3f}")
print("Persistence accuracy by target month:\n", acc_month.round(2).to_string())

# Baseline 2: "most common class for this calendar month"
# (no train/test cut exists yet at the exploration stage, so this is measured on all the data:
#  it is slightly optimistic and will be redone properly with a chronological split later)
major = tab.groupby("month")["cls"].agg(lambda x: x.mode()[0])
pred_month = tab["month"].map(major).values
true_all = tab["cls"].values
print("\nMost common class for each target calendar month:", dict(zip(major.index, [names[c] for c in major.values])))
print(f"Calendar-month baseline accuracy overall: {(pred_month == true_all).mean():.3f}")
print("Calendar-month baseline accuracy by target month:\n",
      pd.Series(pred_month == true_all).groupby(tab["month"].values).mean().round(2).to_string())

# Summary of the three trivial rules (accuracy, balanced accuracy, macro-F1)
from sklearn.metrics import balanced_accuracy_score, f1_score
pred_pers = cls_now.ravel()
pred_major = np.full_like(true_all, pd.Series(true_all).mode()[0])
summary = pd.DataFrame({
    "accuracy": [(p == true_all).mean() for p in (pred_major, pred_pers, pred_month)],
    "balanced_accuracy": [balanced_accuracy_score(true_all, p) for p in (pred_major, pred_pers, pred_month)],
    "macro_F1": [f1_score(true_all, p, average="macro") for p in (pred_major, pred_pers, pred_month)],
}, index=["always the most common class", "persistence (same class as this month)", "most common class of the calendar month"])
print("\n=== Trivial baselines (a model must beat these) ===")
print(summary.round(3).to_string())
imbalance = np.bincount(true_all, minlength=3)
print(f"\nClass counts {dict(zip(names, imbalance.tolist()))}; largest / smallest = {imbalance.max() / imbalance.min():.2f}")

fig3, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5))
by_month.plot.bar(stacked=True, ax=axA, color=["tab:blue", "tab:orange", "tab:red"])
axA.set_title("Class shares by target calendar month")
by_year.plot.bar(stacked=True, ax=axB, color=["tab:blue", "tab:orange", "tab:red"])
axB.set_title("Class shares by target year")
fig3.tight_layout()
fig3.savefig("step7_target_classes.png", dpi=150)
plt.show()