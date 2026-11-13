from pathlib import Path
import html
import re
import base64
import io
import netCDF4
import numpy as np
import matplotlib.pyplot as plt


DAYS = ["20250617", "20251119", "20260103", "20260124", "20260215", "20260311", "20260429", "20260725"]
OUTPUT = Path("bnfld_rain_rate_vs_ldquants_report.html")


def read(path, variable):
    with netCDF4.Dataset(path) as ds:
        return np.ma.asarray(ds.variables[variable][:])


def metrics(reference, comparison):
    if reference.shape != comparison.shape:
        return {"n": 0, "rmse": None, "bias": None, "correlation": None, "r2": None, "shape": f"{reference.shape} vs {comparison.shape}"}
    valid = ~(np.ma.getmaskarray(reference) | np.ma.getmaskarray(comparison))
    x = np.asarray(reference)[valid]; y = np.asarray(comparison)[valid]
    finite = np.isfinite(x) & np.isfinite(y); x = x[finite].astype(float); y = y[finite].astype(float)
    if not len(x): return {"n": 0, "rmse": None, "bias": None, "correlation": None, "r2": None, "shape": ""}
    correlation = float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 else None
    return {"n": len(x), "rmse": float(np.sqrt(np.mean((y - x) ** 2))), "bias": float(np.mean(y - x)), "correlation": correlation, "r2": correlation ** 2 if correlation is not None else None, "shape": ""}


def fmt(value):
    return "—" if value is None else f"{value:.6g}"


def plot_day(day, b1, old, new):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    hours = np.arange(b1.size) / 60.0
    axes[0].plot(hours, b1, color="#19212a", linewidth=1.0, label="bnfld precip_rate")
    axes[0].plot(np.arange(old.size) / 60.0, old, color="#2563a6", linewidth=1.0, label="Old ldquants")
    axes[0].plot(np.arange(new.size) / 60.0, new, color="#c4513b", linewidth=1.0, label="New ldquants")
    axes[0].set_title(f"{day[:4]}-{day[4:6]}-{day[6:]} time series")
    axes[0].set_xlabel("Hour of day"); axes[0].set_ylabel("Rain rate (mm/hour)"); axes[0].set_xlim(0, 24)
    axes[0].grid(True, color="#dfe5ea", linewidth=.6); axes[0].legend(frameon=False, fontsize=8)
    for values, label, color in [(old, "Old ldquants", "#2563a6"), (new, "New ldquants", "#c4513b")]:
        valid = ~(np.ma.getmaskarray(b1) | np.ma.getmaskarray(values))
        x = np.asarray(b1)[valid]; y = np.asarray(values)[valid]
        finite = np.isfinite(x) & np.isfinite(y); x = x[finite]; y = y[finite]
        axes[1].scatter(x, y, s=10, alpha=.5, label=label, color=color)
    finite_all = np.concatenate([np.asarray(b1)[~np.ma.getmaskarray(b1)], np.asarray(old)[~np.ma.getmaskarray(old)], np.asarray(new)[~np.ma.getmaskarray(new)]])
    finite_all = finite_all[np.isfinite(finite_all)]
    if finite_all.size:
        axes[1].plot([finite_all.min(), finite_all.max()], [finite_all.min(), finite_all.max()], "--", color="#596773", linewidth=.9)
    axes[1].set_title("Scatter against bnfld precip_rate"); axes[1].set_xlabel("bnfld (mm/hour)"); axes[1].set_ylabel("ldquants (mm/hour)")
    axes[1].grid(True, color="#dfe5ea", linewidth=.6); axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    stream = io.StringIO(); figure.savefig(stream, format="svg", bbox_inches="tight"); plt.close(figure)
    return base64.b64encode(stream.getvalue().encode()).decode()


def main():
    rows = []
    for day in DAYS:
        b1 = read(Path(f"bnfldM1.b1/bnfldM1.b1.{day}.000000.cdf"), "precip_rate")
        old = read(Path(f"bnfldquantsM1.c1-old/bnfldquantsM1.c1.{day}.000000.nc"), "rain_rate")
        new = read(Path(f"bnfldquantsM1.c1/bnfldquantsM1.c1.{day}.000000.nc"), "rain_rate")
        rows.append((day, metrics(b1, old), metrics(b1, new), plot_day(day, b1, old, new)))
    body = []
    plots = []
    for day, old, new, image in rows:
        date = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        body.append(
            "<tr>" + "".join([
                f"<td>{date}</td>",
                f"<td>{old['n']:,}</td><td>{fmt(old['rmse'])}</td><td>{fmt(old['bias'])}</td><td>{fmt(old['correlation'])}</td><td>{fmt(old['r2'])}</td>",
                f"<td>{new['n']:,}</td><td>{fmt(new['rmse'])}</td><td>{fmt(new['bias'])}</td><td>{fmt(new['correlation'])}</td><td>{fmt(new['r2'])}</td>",
            ]) + "</tr>"
        )
        plots.append(f'<figure><img src="data:image/svg+xml;base64,{image}" alt="Rain-rate time series and scatter plots for {day}"><figcaption>{day[:4]}-{day[4:6]}-{day[6:]}</figcaption></figure>')
    OUTPUT.write_text(f'''<section class="report">
<h1>bnfld rain rate versus ldquants</h1>
<p>Rainfall-rate comparison for the eight dates where ldquants rain rates differed by more than 1%. The bnfld input variable is <code>precip_rate</code>; both ldquants products use <code>rain_rate</code>.</p>
<table><thead><tr><th rowspan="2">Date</th><th colspan="5">Old ldquants</th><th colspan="5">New ldquants</th></tr><tr><th>Paired values</th><th>RMSE</th><th>Bias</th><th>Correlation</th><th>R²</th><th>Paired values</th><th>RMSE</th><th>Bias</th><th>Correlation</th><th>R²</th></tr></thead><tbody>{''.join(body)}</tbody></table>
<p class="note">Bias is ldquants minus bnfld. R² is correlation squared; both use aligned finite values only. Missing values are excluded separately for each comparison.</p>
<h2>Plots by day</h2>
{''.join(plots)}
<style>
.report {{ font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:#19212a; max-width:1400px; }}
h1 {{ margin-bottom:8px; }} p {{ color:#596773; }} table {{ border-collapse:collapse; width:100%; }} th,td {{ border-bottom:1px solid #d9e0e7; padding:8px 10px; text-align:right; }} th {{ background:#f1f5f8; }} th:first-child,td:first-child {{ text-align:left; }} code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }} .note {{ font-size:.9rem; }}
figure {{ margin:24px 0 34px; }} figure img {{ width:100%; max-width:1300px; height:auto; }} figcaption {{ color:#596773; font-weight:600; }}
</style></section>''', encoding="utf-8")
    print(OUTPUT.resolve())


if __name__ == "__main__": main()
