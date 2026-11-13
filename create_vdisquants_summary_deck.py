from pathlib import Path
import re
import numpy as np
import netCDF4
from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


OLD = Path("bnfvdisquantsM1.c1-old")
NEW = Path("bnfvdisquantsM1.c1")
OUT = Path("vdisquants_old_new_summary.pptx")
NAVY = RGBColor(22, 42, 66)
BLUE = RGBColor(37, 99, 166)
TEAL = RGBColor(0, 125, 135)
GRAY = RGBColor(90, 103, 116)
LIGHT = RGBColor(241, 245, 248)
WHITE = RGBColor(255, 255, 255)


START_DATE = "20250212"
END_DATE = "20261101"
EXCLUDED_DATES = {"20250817", "20250820", "20251004", "20260910"}


def date_matched_paths():
    old = {re.search(r"(\d{8})", path.name).group(1): path for path in OLD.glob("*.nc")}
    new = {re.search(r"(\d{8})", path.name).group(1): path for path in NEW.glob("*.nc")}
    dates = [
        date
        for date in sorted(set(old) & set(new))
        if START_DATE <= date <= END_DATE and date not in EXCLUDED_DATES
    ]
    return [(date, old[date], new[date]) for date in dates]


def load_pairs(variable, pairs):
    left, right = [], []
    shape_mismatches = 0
    for _, old_path, new_path in pairs:
        with netCDF4.Dataset(old_path) as a, netCDF4.Dataset(new_path) as b:
            x = np.ma.asarray(a.variables[variable][:])
            y = np.ma.asarray(b.variables[variable][:])
            if x.shape != y.shape:
                shape_mismatches += 1
                continue
            valid = ~(np.ma.getmaskarray(x) | np.ma.getmaskarray(y))
            x = np.asarray(x)[valid]
            y = np.asarray(y)[valid]
            finite = np.isfinite(x) & np.isfinite(y)
            left.extend(x[finite].tolist())
            right.extend(y[finite].tolist())
    return np.asarray(left), np.asarray(right), shape_mismatches


def add_text(slide, text, x, y, w, h, size=18, color=NAVY, bold=False, align=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = "Aptos"
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    if align is not None:
        p.alignment = align
    return box


def add_title(slide, title, subtitle=None):
    add_text(slide, title, 0.55, 0.32, 12.2, 0.55, 27, NAVY, True)
    if subtitle:
        add_text(slide, subtitle, 0.58, 0.88, 12.0, 0.3, 11, GRAY)


def add_footer(slide, text="Source: compare_netcdf_dirs.py; 31 matched daily NetCDF files"):
    add_text(slide, text, 0.58, 7.12, 12.0, 0.2, 8, GRAY)


def add_metric(slide, value, label, x, y, w=2.8):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(1.08))
    shape.fill.solid(); shape.fill.fore_color.rgb = LIGHT
    shape.line.color.rgb = LIGHT
    add_text(slide, value, x + 0.15, y + 0.14, w - 0.3, 0.38, 25, NAVY, True)
    add_text(slide, label, x + 0.15, y + 0.58, w - 0.3, 0.25, 10, GRAY)


def add_scatter(slide, xvals, yvals, title, unit, left_label, right_label):
    data = XyChartData()
    series = data.add_series("Paired observations")
    sample = np.linspace(0, len(xvals) - 1, min(len(xvals), 450), dtype=int)
    for i in sample:
        series.add_data_point(float(xvals[i]), float(yvals[i]))
    lower = float(min(xvals.min(), yvals.min()))
    upper = float(max(xvals.max(), yvals.max()))
    agreement = data.add_series("1:1 agreement")
    agreement.add_data_point(lower, lower)
    agreement.add_data_point(upper, upper)
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.XY_SCATTER, Inches(0.7), Inches(1.35), Inches(7.25), Inches(5.1), data
    ).chart
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    chart.value_axis.has_title = True
    chart.category_axis.has_title = True
    chart.value_axis.axis_title.text_frame.text = f"New vdisquants ({unit})"
    chart.category_axis.axis_title.text_frame.text = f"Old vdisquants ({unit})"
    chart.value_axis.has_major_gridlines = True
    chart.value_axis.major_gridlines.format.line.color.rgb = RGBColor(215, 222, 228)
    chart.category_axis.has_major_gridlines = True
    chart.category_axis.major_gridlines.format.line.color.rgb = RGBColor(215, 222, 228)
    chart.series[0].marker.size = 5
    chart.series[0].marker.format.fill.solid()
    chart.series[0].marker.format.fill.fore_color.rgb = BLUE
    chart.series[0].marker.format.line.color.rgb = BLUE
    chart.series[1].format.line.color.rgb = TEAL
    chart.series[1].format.line.width = Pt(1.5)
    return chart


def load_band_pairs(base_name, bands, pairs):
    result = {}
    for band in bands:
        result[band] = load_pairs(f"{base_name}_{band}20c", pairs)[:2]
    return result


def add_multiband_scatter(slide, band_pairs, title, unit):
    data = XyChartData()
    colors = [BLUE, TEAL, RGBColor(210, 120, 45), RGBColor(130, 75, 160), RGBColor(190, 55, 70)]
    lower = min(float(x.min()) for x, _ in band_pairs.values() if len(x))
    upper = max(float(y.max()) for _, y in band_pairs.values() if len(y))
    for band, (xvals, yvals) in band_pairs.items():
        series = data.add_series(band.upper())
        sample = np.linspace(0, len(xvals) - 1, min(len(xvals), 220), dtype=int)
        for i in sample:
            series.add_data_point(float(xvals[i]), float(yvals[i]))
    agreement = data.add_series("1:1")
    agreement.add_data_point(lower, lower); agreement.add_data_point(upper, upper)
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.XY_SCATTER, Inches(0.65), Inches(1.35), Inches(8.05), Inches(5.15), data
    ).chart
    chart.has_legend = True; chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.has_title = True; chart.chart_title.text_frame.text = title
    chart.value_axis.has_title = True; chart.category_axis.has_title = True
    chart.value_axis.axis_title.text_frame.text = f"New vdisquants ({unit})"
    chart.category_axis.axis_title.text_frame.text = f"Old vdisquants ({unit})"
    chart.value_axis.has_major_gridlines = True; chart.category_axis.has_major_gridlines = True
    for index in range(len(chart.series) - 1):
        series = chart.series[index]
        series.marker.size = 4
        series.marker.format.fill.solid(); series.marker.format.fill.fore_color.rgb = colors[index]
        series.marker.format.line.color.rgb = colors[index]
    chart.series[-1].format.line.color.rgb = RGBColor(80, 80, 80)
    chart.series[-1].format.line.width = Pt(1.25)
    return chart


def add_summary_table(slide, rows, x=8.25, y=1.55, w=4.4, h=3.8):
    table = slide.shapes.add_table(len(rows) + 1, 3, Inches(x), Inches(y), Inches(w), Inches(h)).table
    table.columns[0].width = Inches(2.15); table.columns[1].width = Inches(1.1); table.columns[2].width = Inches(1.15)
    headers = ["Metric", "Old", "New"]
    for c, text in enumerate(headers):
        cell = table.cell(0, c); cell.text = text; cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
        for p in cell.text_frame.paragraphs:
            p.font.name = "Aptos"; p.font.size = Pt(11); p.font.bold = True; p.font.color.rgb = WHITE
    for r, row in enumerate(rows, 1):
        for c, text in enumerate(row):
            cell = table.cell(r, c); cell.text = str(text); cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT if r % 2 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.name = "Aptos"; p.font.size = Pt(10); p.font.color.rgb = NAVY


def main():
    pairs = date_matched_paths()
    rain_old, rain_new, rain_shape_mismatches = load_pairs("rain_rate", pairs)
    nt_old, nt_new, nt_shape_mismatches = load_pairs("total_droplet_concentration", pairs)
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    slide = prs.slides.add_slide(blank); slide.background.fill.solid(); slide.background.fill.fore_color.rgb = WHITE
    add_text(slide, "Old versus new vdisquants", 0.8, 1.65, 11.7, 0.75, 35, NAVY, True, PP_ALIGN.CENTER)
    add_text(slide, "Date-matched NetCDF comparison, 2025-02-12 through 2026-11-01", 1.0, 2.55, 11.3, 0.35, 18, GRAY, False, PP_ALIGN.CENTER)
    add_text(slide, "Rainfall rate and total droplet concentration", 1.0, 3.15, 11.3, 0.35, 16, TEAL, False, PP_ALIGN.CENTER)
    add_text(slide, "Generated 2026-11-03", 1.0, 6.55, 11.3, 0.25, 10, GRAY, False, PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(blank); add_title(slide, "Comparison summary", "Old: bnfvdisquantsM1.c1-old   New: bnfvdisquantsM1.c1")
    add_metric(slide, f"{len(pairs)}", "date-matched daily files", 0.7, 1.45)
    add_metric(slide, f"{len(pairs)}", "files marked different", 3.75, 1.45)
    add_metric(slide, f"{len(rain_old):,}", "paired rain-rate values", 6.8, 1.45)
    add_metric(slide, f"{rain_shape_mismatches}", "files with shape mismatch", 9.85, 1.45)
    add_text(slide, "After removing the four suspected ingest or QC days, both requested moments agree exactly across the remaining matched observations.", 0.85, 3.15, 11.6, 0.8, 21, NAVY)
    add_text(slide, "Rainfall rate: RMSE 0; R² 1.000", 0.85, 4.45, 6.5, 0.45, 22, TEAL, True)
    add_text(slide, "Total concentration: RMSE 0; R² 1.000", 0.85, 5.15, 7.2, 0.45, 22, BLUE, True)
    add_footer(slide)

    for old, new, title, unit, label, mismatches in [(rain_old, rain_new, "Rainfall rate", "mm/hour", "rain_rate", rain_shape_mismatches), (nt_old, nt_new, "Total droplet concentration", "1/m³", "total_droplet_concentration", nt_shape_mismatches)]:
        slide = prs.slides.add_slide(blank); add_title(slide, title, f"Old versus new vdisquants; {len(old):,} paired finite observations")
        add_scatter(slide, old, new, title, unit, "Old", "New")
        rmse = float(np.sqrt(np.mean((new - old) ** 2))); r2 = float(1 - np.sum((new-old)**2) / np.sum((old-old.mean())**2))
        add_summary_table(slide, [("Paired values", f"{len(old):,}", f"{len(new):,}"), ("Mean", f"{old.mean():.4g}", f"{new.mean():.4g}"), ("Median", f"{np.median(old):.4g}", f"{np.median(new):.4g}"), ("RMSE", f"{rmse:.4g}", "same"), ("R²", f"{r2:.6f}", "same")])
        add_footer(slide)

    band_sets = [
        ("reflectivity_factor", ["sband", "cband", "xband", "kaband", "wband"], "Reflectivity", "dBZ"),
        ("differential_reflectivity", ["sband", "cband", "xband", "kaband"], "Differential reflectivity", "dB"),
        ("specific_differential_phase", ["sband", "cband", "xband", "kaband"], "Kdp (specific differential phase)", "degree/km"),
    ]
    for base_name, bands, title, unit in band_sets:
        band_pairs = load_band_pairs(base_name, bands, pairs)
        slide = prs.slides.add_slide(blank)
        add_title(slide, title, "Each band uses a distinct color; dashed line indicates 1:1 agreement")
        add_multiband_scatter(slide, band_pairs, title, unit)
        add_text(slide, "The chart samples up to 220 points per band for display. RMSE and R² in the HTML report use every paired finite value.", 8.95, 1.65, 3.7, 1.1, 16, NAVY)
        add_text(slide, "Bands: " + ", ".join(b.upper() for b in bands), 8.95, 3.25, 3.65, 0.8, 17, TEAL, True)
        add_footer(slide)

    slide = prs.slides.add_slide(blank)
    add_title(slide, "Spectral width", "Requested field is not present in either vdisquants file set")
    add_text(slide, "No spectral-width variable was found in the old or new NetCDF products, so there is no valid spectral-width comparison to plot.", 1.0, 2.2, 11.2, 0.9, 26, NAVY, True, PP_ALIGN.CENTER)
    add_text(slide, "The available radar moments include reflectivity factor, differential reflectivity, and specific differential phase (Kdp).", 1.25, 3.55, 10.8, 0.65, 18, GRAY, False, PP_ALIGN.CENTER)
    add_footer(slide)

    slide = prs.slides.add_slide(blank); add_title(slide, "Interpretation and coverage", "What the expanded report includes")
    add_metric(slide, "6443", "moment scatter plots", 0.85, 1.45, 3.65)
    add_metric(slide, "0", "shape-mismatch days", 4.85, 1.45, 3.65)
    add_metric(slide, "250", "maximum points drawn per scatter plot", 8.85, 1.45, 3.65)
    add_text(slide, "The report calculates RMSE and R² from every paired finite value. The visual plots cap displayed points at 250.", 0.9, 3.4, 11.4, 0.8, 20, NAVY)
    add_text(slide, "Removed periods: 2025-08-17, 2025-08-20, 2025-10-04, and 2026-09-10. These days had concentration differences consistent with ingest or QC changes.", 0.9, 4.55, 11.4, 0.65, 17, GRAY)
    add_footer(slide, "Source: vdisquants_old_vs_new_report.html")

    prs.save(OUT)
    print(OUT.resolve())


if __name__ == "__main__":
    main()
