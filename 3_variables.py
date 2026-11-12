
"""
Step 3 - Univariate analysis of the 12 variables.

Exploration only:
- No preprocessing.
- No missing-value removal.
- No unit conversion.
- No logarithmic transformation.
- Original values are preserved.

For each variable:
1. Extract its original values.
2. Calculate descriptive statistics.
3. Draw a histogram.
4. Export the statistical summary to CSV.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# 1. Open the original dataset
ds = xr.open_dataset("era5land_algeria.nc")

# 2. Variables to analyze
ORDER = [
    "t2m", "d2m", "ssr", "str",
    "sshf", "slhf", "e", "tp",
    "ro", "u10", "v10", "sp"
]

rows = []

# 3. Analyze each variable
for name in ORDER:

    # Extract original values without modification
    raw = ds[name].values.ravel()

    # Calculate descriptive statistics
    rows.append({
        "variable": name,
        "n_values": raw.size,
        "missing_values": np.isnan(raw).sum(),
        "min": np.nanmin(raw),
        "max": np.nanmax(raw),
        "mean": np.nanmean(raw),
        "median": np.nanmedian(raw),
        "std": np.nanstd(raw),
        "skewness": pd.Series(raw).skew(),
        "zero_share_%": 100 * np.mean(raw == 0)
    })

# 4. Create statistical summary
summary = pd.DataFrame(rows).set_index("variable")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(summary)

# 5. Export summary to CSV
summary.to_csv("step3_summary.csv")

# 6. Create histograms using original values
fig, axes = plt.subplots(3, 4, figsize=(18, 11))

for ax, name in zip(axes.ravel(), ORDER):

    raw = ds[name].values.ravel()

    ax.hist(raw, bins=100, color="steelblue")

    ax.set_title(name)
    ax.set_xlabel("Original value")
    ax.set_ylabel("Frequency")

plt.tight_layout()

# 7. Save and display histograms
plt.savefig("step3_histograms.png", dpi=150)
plt.show()

# 8. Close dataset
ds.close()
