from pathlib import Path
import re
import netCDF4
import numpy as np
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

OLD = Path("bnfldquantsM1.c1-old")
NEW = Path("bnfldquantsM1.c1")
OUT = Path("ldquants_gt1pct_difference_report.pptx")
NAVY = RGBColor(22, 42, 66)
BLUE = RGBColor(37, 99, 166)
TEAL = RGBColor(0, 125, 135)
GRAY = RGBColor(90, 103, 116)
LIGHT = RGBColor(241, 245, 248)
WHITE = RGBColor(255, 255, 255)

DSD = [
    "rain_rate", "gammapsd_slope", "gammapsd_shape", "num_concen", "norm_num_concen",
    "med_diameter", "mass_weighted_mean_diameter", "lwc", "total_droplet_concentration", "exppsd_slope",
]
RADAR = [
    "reflectivity_factor_sband20c", "reflectivity_factor_cband20c", "reflectivity_factor_xband20c", "reflectivity_factor_kaband20c", "reflectivity_factor_wband20c",
    "differential_reflectivity_sband20c", "differential_reflectivity_cband20c", "differential_reflectivity_xband20c", "differential_reflectivity_kaband20c",
    "specific_differential_phase_sband20c", "specific_differential_phase_cband20c", "specific_differential_phase_xband20c", "specific_differential_phase_kaband20c",
    "specific_attenuation_sband20c", "specific_attenuation_cband20c", "specific_attenuation_xband20c", "specific_attenuation_kaband20c",
    "specific_differential_attenuation_sband20c", "specific_differential_attenuation_cband20c", "specific_differential_attenuation_kaband20c", "specific_differential_attenuation_xband20c",
]


def paired_paths():
    old = {re.search(r"(\d{8})", p.name).group(1): p for p in OLD.glob("*.nc")}
    new = {re.search(r"(\d{8})", p.name).group(1): p for p in NEW.glob("*.nc")}
    return [(d, old[d], new[d]) for d in sorted(set(old) & set(new)) if "20250201" <= d <= "20261101"]


def difference_stats(paths, variables):
    stats = {v: {"points": 0, "dates": []} for v in variables}
    for day, old_path, new_path in paths:
        with netCDF4.Dataset(old_path) as old, netCDF4.Dataset(new_path) as new:
            for variable in variables:
                left = np.ma.asarray(old.variables[variable][:]); right = np.ma.asarray(new.variables[variable][:])
                if left.shape != right.shape:
                    continue
                valid = ~(np.ma.getmaskarray(left) | np.ma.getmaskarray(right))
                x = np.asarray(left)[valid]; y = np.asarray(right)[valid]
                finite = np.isfinite(x) & np.isfinite(y); x = x[finite].astype(float); y = y[finite].astype(float)
                diff = np.abs(y - x); denominator = np.abs(x)
                exceeds = diff > 0.01 * denominator
                exceeds[denominator == 0] = diff[denominator == 0] > 0
                count = int(exceeds.sum()); stats[variable]["points"] += count
                if count:
                    stats[variable]["dates"].append(day)
    return stats


def add_text(slide, text, x, y, w, h, size=18, color=NAVY, bold=False, align=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    p = box.text_frame.paragraphs[0]; p.text = text; p.font.name = "Aptos"; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
    if align is not None: p.alignment = align
    return box


def title(slide, text, subtitle=None):
    add_text(slide, text, .55, .3, 12.2, .5, 27, NAVY, True)
    if subtitle: add_text(slide, subtitle, .58, .84, 12.1, .28, 11, GRAY)


def footer(slide):
    add_text(slide, "Relative difference: |new − old| / |old| > 1%; values with old = 0 count when new is nonzero", .58, 7.12, 12.0, .2, 8, GRAY)


def table_slide(prs, heading, subtitle, rows, color):
    slide = prs.slides.add_slide(prs.slide_layouts[6]); title(slide, heading, subtitle)
    x, y, w, h = .65, 1.3, 12.0, 5.55
    table = slide.shapes.add_table(len(rows) + 1, 3, Inches(x), Inches(y), Inches(w), Inches(h)).table
    table.columns[0].width = Inches(3.4); table.columns[1].width = Inches(1.3); table.columns[2].width = Inches(7.3)
    for c, text in enumerate(["Variable", ">1% values", "Affected dates"]):
        cell = table.cell(0, c); cell.text = text; cell.fill.solid(); cell.fill.fore_color.rgb = color
        for p in cell.text_frame.paragraphs: p.font.name = "Aptos"; p.font.size = Pt(12); p.font.bold = True; p.font.color.rgb = WHITE
    for r, row in enumerate(rows, 1):
        for c, text in enumerate(row):
            cell = table.cell(r, c); cell.text = str(text); cell.fill.solid(); cell.fill.fore_color.rgb = LIGHT if r % 2 else WHITE
            for p in cell.text_frame.paragraphs: p.font.name = "Aptos"; p.font.size = Pt(9.5); p.font.color.rgb = NAVY
    footer(slide)
    return slide


def main():
    paths = paired_paths(); dsd_stats = difference_stats(paths, DSD); radar_stats = difference_stats(paths, RADAR)
    dsd_dates = sorted({d for v in DSD for d in dsd_stats[v]["dates"]}); radar_dates = sorted({d for v in RADAR for d in radar_stats[v]["dates"]})
    prs = Presentation(); prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5); blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank); slide.background.fill.solid(); slide.background.fill.fore_color.rgb = WHITE
    add_text(slide, ">1% differences in ldquants", .8, 1.65, 11.7, .75, 35, NAVY, True, PP_ALIGN.CENTER)
    add_text(slide, "bnfldquantsM1.c1-old versus bnfldquantsM1.c1", 1, 2.55, 11.3, .35, 18, GRAY, False, PP_ALIGN.CENTER)
    add_text(slide, "Date range: 2025-02-01 through 2026-11-01", 1, 3.12, 11.3, .35, 16, TEAL, False, PP_ALIGN.CENTER)
    add_text(slide, "619 date-matched files scanned", 1, 6.55, 11.3, .25, 10, GRAY, False, PP_ALIGN.CENTER)

    slide = prs.slides.add_slide(blank); title(slide, "Where differences occur", "Days with at least one finite value more than 1% different")
    add_text(slide, "DSD parameters", .9, 1.55, 5.2, .4, 24, TEAL, True)
    add_text(slide, "10 days", .9, 2.05, 3.0, .45, 28, NAVY, True)
    add_text(slide, ", ".join(dsd_dates), .9, 2.75, 5.4, 1.0, 18, NAVY)
    add_text(slide, "Radar moments", 7.0, 1.55, 5.2, .4, 24, BLUE, True)
    add_text(slide, "8 days", 7.0, 2.05, 3.0, .45, 28, NAVY, True)
    add_text(slide, ", ".join(radar_dates), 7.0, 2.75, 5.4, 1.0, 18, NAVY)
    add_text(slide, "Eight days affect both groups. DSD-only days are 2026-04-05 and 2026-10-10.", .9, 4.7, 11.2, .5, 19, GRAY)
    footer(slide)

    dsd_rows = [(v, f"{dsd_stats[v]['points']:,}", ", ".join(dsd_stats[v]["dates"])) for v in DSD if dsd_stats[v]["dates"]]
    table_slide(prs, "DSD parameters with >1% differences", "Counts aggregate finite time points across the affected days", dsd_rows, TEAL)
    radar_rows_1 = [(v, f"{radar_stats[v]['points']:,}", ", ".join(radar_stats[v]["dates"])) for v in RADAR[:9] if radar_stats[v]["dates"]]
    table_slide(prs, "Radar moments: reflectivity and Zdr", "All affected radar bands share the same eight dates", radar_rows_1, BLUE)
    radar_rows_2 = [(v, f"{radar_stats[v]['points']:,}", ", ".join(radar_stats[v]["dates"])) for v in RADAR[9:] if radar_stats[v]["dates"]]
    table_slide(prs, "Radar moments: Kdp and attenuation", "Specific differential phase and attenuation bands", radar_rows_2, BLUE)

    slide = prs.slides.add_slide(blank); title(slide, "Interpretation", "How to read the mined differences")
    add_text(slide, "The differences cluster on eight dates across both DSD and radar fields. Two additional dates affect DSD parameters only.", .9, 1.5, 11.5, .7, 24, NAVY, True)
    add_text(slide, "The scan uses a relative threshold of 1% against the old value. Near-zero old values can produce large percentages, so inspect the raw values and QC flags for those points.", .9, 2.75, 11.5, .9, 20, GRAY)
    add_text(slide, "The HTML comparison report contains the day-by-day values, missingness, and variable attributes.", .9, 4.25, 11.5, .5, 18, TEAL, True)
    footer(slide)
    prs.save(OUT); print(OUT.resolve())


if __name__ == "__main__": main()
