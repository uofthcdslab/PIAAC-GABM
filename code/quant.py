import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "parsed_responses.xlsx"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "rating_histograms.png"

sheets = pd.read_excel(DEFAULT_INPUT, sheet_name=None, usecols=["Rating"])
frequencies = {
    policy: data["Rating"].value_counts().reindex(range(1, 6), fill_value=0)
    for policy, data in sheets.items()
}

column_count = 4
row_count = math.ceil(len(frequencies) / column_count)
figure, axes = plt.subplots(
    row_count,
    column_count,
    figsize=(16, 3.2 * row_count),
    sharex=True,
    sharey=True,
)

for axis, (policy, counts) in zip(axes.flat, frequencies.items()):
    axis.bar(counts.index, counts.values)
    axis.set_title(policy, fontsize=9)
    axis.set_xticks(range(1, 6))
    axis.set_xlabel("Rating")
    axis.set_ylabel("Frequency")

for axis in axes.flat[len(frequencies):]:
    axis.set_visible(False)

figure.tight_layout()
figure.savefig(DEFAULT_OUTPUT, dpi=300, bbox_inches="tight")
plt.close(figure)
print(f"Saved histogram grid to {DEFAULT_OUTPUT}")
