from __future__ import annotations

import base64
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4
import numpy as np


OLD = Path("bnfldquantsM1.c1-old")
NEW = Path("bnfldquantsM1.c1")
DATES = [
    "20250617", "20251119", "20260103", "20260124", "20260215",
    "20260311", "20260405", "20260429", "20260725", "20261010",
]
OUTPUT = Path("rain-rate-difference-time-series.html")


def files_by_date(root: Path) -> dict[str, Path]:
    return {
        re.search(r"(\d{8})", path.name).group(1): path
        for path in root.glob("*.nc")
        if re.search(r"(\d{8})", path.name)
    }


def values(path: Path) -> np.ma.MaskedArray:
    with netCDF4.Dataset(path) as dataset:
        return np.ma.asarray(dataset.variables["rain_rate"][:])


def make_figure(old_files: dict[str, Path], new_files: dict[str, Path]) -> str:
    fig, axes = plt.subplots(5, 2, figsize=(13, 17), sharex=False)
    axes = axes.ravel()
    for index, day in enumerate(DATES):
        old = values(old_files[day])
        new = values(new_files[day])
        ax = axes[index]
        old_x = np.arange(old.size) / 60.0
        new_x = np.arange(new.size) / 60.0
        ax.plot(old_x, old, color="#2563a6", linewidth=1.1, label="Old")
        ax.plot(new_x, new, color="#c4513b", linewidth=1.1, label="New")
        ax.set_title(f"{day[:4]}-{day[4:6]}-{day[6:]}", loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Rain rate (mm/hour)")
        ax.grid(True, color="#dfe5ea", linewidth=0.6)
        ax.set_xlim(0, 24)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Rain-rate time series on days with >1% DSD differences", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.02, "Old and new ldquants values are overlaid; masked values are left blank.", ha="center", fontsize=10, color="#596773")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    stream = io.StringIO()
    fig.savefig(stream, format="svg", bbox_inches="tight")
    plt.close(fig)
    return stream.getvalue()


def main() -> None:
    old_files = files_by_date(OLD)
    new_files = files_by_date(NEW)
    missing = [day for day in DATES if day not in old_files or day not in new_files]
    if missing:
        raise SystemExit(f"Missing requested dates: {', '.join(missing)}")
    svg = make_figure(old_files, new_files)
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    OUTPUT.write_text(
        '<div id="rain-rate-difference-time-series">\n'
        '<h2>Rain-rate differences by day</h2>\n'
        '<p>Days with more than 1% differences in at least one DSD parameter.</p>\n'
        f'<img src="data:image/svg+xml;base64,{encoded}" alt="Overlaid old and new rain-rate time series for ten difference days">\n'
        '</div>\n',
        encoding="utf-8",
    )
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
