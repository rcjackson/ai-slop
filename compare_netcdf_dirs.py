#!/usr/bin/env python3
"""Compare NetCDF files in two directory trees and write an HTML report."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    import netCDF4
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
    netCDF4 = None
    np = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


NETCDF_SUFFIXES = {".nc", ".nc4", ".cdf"}


def builtin(value: Any) -> Any:
    """Convert NetCDF/NumPy metadata values into ordinary Python values."""
    if np is not None:
        if isinstance(value, np.ma.MaskedArray):
            return builtin(value.filled(None).tolist())
        if isinstance(value, np.ndarray):
            return [builtin(item) for item in value.tolist()]
        if isinstance(value, np.generic):
            return builtin(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [builtin(item) for item in value]
    if isinstance(value, dict):
        return {str(key): builtin(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def metadata(dataset: Any) -> dict[str, Any]:
    return {
        "dimensions": {
            name: {"size": len(dimension), "unlimited": dimension.isunlimited()}
            for name, dimension in dataset.dimensions.items()
        },
        "global_attributes": {
            name: builtin(dataset.getncattr(name)) for name in dataset.ncattrs()
        },
    }


def attributes(variable: Any) -> dict[str, Any]:
    return {name: builtin(variable.getncattr(name)) for name in variable.ncattrs()}


def values_equal(left: Any, right: Any) -> bool:
    try:
        a = np.asarray(left)
        b = np.asarray(right)
        if a.shape != b.shape:
            return False
        try:
            return bool(np.array_equal(a, b, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(a, b))
    except Exception:
        return left == right


def compare_attributes(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    changes = []
    for name in sorted(set(left) | set(right)):
        if name not in left:
            changes.append({"name": name, "status": "only right", "left": None, "right": right[name]})
        elif name not in right:
            changes.append({"name": name, "status": "only left", "left": left[name], "right": None})
        elif not values_equal(left[name], right[name]):
            changes.append({"name": name, "status": "changed", "left": left[name], "right": right[name]})
    return changes


def is_vdisquants_pair(
    relative_path: str, left_metadata: dict[str, Any], right_metadata: dict[str, Any]
) -> bool:
    """Recognize vdisquants files by their path or product metadata."""
    identifying_text = [relative_path]
    for dataset_metadata in (left_metadata, right_metadata):
        identifying_text.extend(
            f"{name} {value}"
            for name, value in dataset_metadata.get("global_attributes", {}).items()
        )
    return "vdisquants" in " ".join(identifying_text).casefold()


def is_vdisquants_moment(name: str, left_variable: Any, right_variable: Any) -> bool:
    """Return whether a variable is a floating-point vdisquants measurement."""
    coordinate_names = {
        "alt", "altitude", "base_time", "height", "lat", "latitude", "lon",
        "longitude", "time", "time_offset",
    }
    if name.casefold() in coordinate_names:
        return False
    for variable in (left_variable, right_variable):
        if not np.issubdtype(variable.dtype, np.floating) or variable.ndim == 0:
            return False
        standard_name = str(getattr(variable, "standard_name", "")).casefold()
        if standard_name in {"time", "latitude", "longitude", "altitude", "height"}:
            return False
        if str(getattr(variable, "axis", "")).casefold() in {"t", "x", "y", "z"}:
            return False
    return True


def scatter_statistics(
    left_flat: Any,
    right_flat: Any,
    valid: Any,
    max_points: int,
) -> dict[str, Any]:
    """Summarize finite paired values and retain a bounded set for plotting."""
    left_values = left_flat[valid].astype(np.float64, copy=False)
    right_values = right_flat[valid].astype(np.float64, copy=False)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    left_values = left_values[finite]
    right_values = right_values[finite]
    count = int(left_values.size)
    result: dict[str, Any] = {
        "count": count,
        "plotted": min(count, max_points),
        "rmse": None,
        "r_squared": None,
        "left_points": [],
        "right_points": [],
        "plot_min": None,
        "plot_max": None,
    }
    if not count:
        return result

    left_extended = left_values.astype(np.longdouble)
    right_extended = right_values.astype(np.longdouble)
    residual = right_extended - left_extended
    residual_sum_squares = np.sum(residual * residual, dtype=np.longdouble)
    rmse = np.sqrt(residual_sum_squares / count)
    if np.isfinite(rmse):
        result["rmse"] = float(rmse)

    if count > 1:
        left_centered = left_extended - np.mean(left_extended, dtype=np.longdouble)
        total_sum_squares = np.sum(left_centered * left_centered, dtype=np.longdouble)
        if total_sum_squares > 0:
            r_squared = 1 - residual_sum_squares / total_sum_squares
            if np.isfinite(r_squared):
                result["r_squared"] = float(r_squared)

    result["plot_min"] = float(min(np.min(left_values), np.min(right_values)))
    result["plot_max"] = float(max(np.max(left_values), np.max(right_values)))

    if count > max_points:
        indexes = np.linspace(0, count - 1, max_points, dtype=np.int64)
        left_values = left_values[indexes]
        right_values = right_values[indexes]
    result["left_points"] = left_values.tolist()
    result["right_points"] = right_values.tolist()
    return result


def compare_dimensions(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    changes = []
    for name in sorted(set(left) | set(right)):
        if name not in left:
            changes.append({"name": name, "status": "only right", "left": None, "right": right[name]})
        elif name not in right:
            changes.append({"name": name, "status": "only left", "left": left[name], "right": None})
        elif left[name] != right[name]:
            changes.append({"name": name, "status": "changed", "left": left[name], "right": right[name]})
    return changes


def value_for_display(value: Any) -> str:
    if value is None:
        return "<missing>"
    if np is not None and isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return repr(value.decode("utf-8", errors="replace"))
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return repr(value)
    return repr(value)


def scalar_at(array: Any, flat_index: int) -> Any:
    value = array.reshape(-1)[flat_index]
    if np.ma.is_masked(value):
        return None
    return value


def compare_values(
    left_variable: Any,
    right_variable: Any,
    atol: float,
    rtol: float,
    max_differences: int,
    max_scatter_points: int | None = None,
) -> dict[str, Any]:
    left = np.ma.asarray(left_variable[...])
    right = np.ma.asarray(right_variable[...])
    left_data = np.asarray(np.ma.getdata(left))
    right_data = np.asarray(np.ma.getdata(right))
    left_mask = np.ma.getmaskarray(left)
    right_mask = np.ma.getmaskarray(right)

    result: dict[str, Any] = {
        "left_shape": list(left_data.shape),
        "right_shape": list(right_data.shape),
        "total": int(left_data.size),
        "equal": 0,
        "changed": 0,
        "one_missing": 0,
        "both_missing": 0,
        "samples": [],
        "samples_truncated": False,
        "max_abs_difference": None,
        "mean_abs_difference": None,
        "max_relative_difference": None,
    }
    if left_data.shape != right_data.shape:
        result["shape_mismatch"] = True
        return result

    left_mask = np.broadcast_to(left_mask, left_data.shape).reshape(-1)
    right_mask = np.broadcast_to(right_mask, right_data.shape).reshape(-1)
    left_flat = left_data.reshape(-1)
    right_flat = right_data.reshape(-1)
    both_missing = left_mask & right_mask
    one_missing = left_mask ^ right_mask
    valid = ~(left_mask | right_mask)
    result["both_missing"] = int(np.count_nonzero(both_missing))
    result["one_missing"] = int(np.count_nonzero(one_missing))

    numeric = (
        np.issubdtype(left_data.dtype, np.number)
        and np.issubdtype(right_data.dtype, np.number)
    )
    if numeric and max_scatter_points is not None:
        result["scatter_comparison"] = scatter_statistics(
            left_flat, right_flat, valid, max_scatter_points
        )
    if numeric:
        try:
            equal = np.isclose(left_flat, right_flat, atol=atol, rtol=rtol, equal_nan=True)
        except (TypeError, ValueError):
            equal = left_flat == right_flat
    else:
        try:
            equal = np.equal(left_flat, right_flat)
        except (TypeError, ValueError):
            equal = np.fromiter(
                (a == b for a, b in zip(left_flat, right_flat)), dtype=bool, count=left_flat.size
            )
    equal = np.asarray(equal, dtype=bool)
    changed_mask = valid & ~equal
    result["equal"] = int(np.count_nonzero(valid & equal))
    result["changed"] = int(np.count_nonzero(changed_mask))
    differing = changed_mask | one_missing
    differing_indexes = np.flatnonzero(differing)

    if numeric and np.any(changed_mask):
        indexes = np.flatnonzero(changed_mask)
        try:
            a = left_flat[indexes].astype(np.longdouble)
            b = right_flat[indexes].astype(np.longdouble)
            delta = np.abs(b - a)
            finite = np.isfinite(delta)
            finite_delta = delta[finite]
            if finite_delta.size:
                result["max_abs_difference"] = float(np.max(finite_delta))
                result["mean_abs_difference"] = float(np.mean(finite_delta))
            nonzero_scale = np.maximum(np.abs(a), np.abs(b))
            ratios = np.divide(
                delta,
                nonzero_scale,
                out=np.zeros_like(delta),
                where=(nonzero_scale != 0) & np.isfinite(delta),
            )
            finite_ratios = ratios[np.isfinite(ratios)]
            if finite_ratios.size:
                result["max_relative_difference"] = float(np.max(finite_ratios))
        except (TypeError, ValueError, OverflowError):
            pass

    limit = differing_indexes.size if max_differences == 0 else min(differing_indexes.size, max_differences)
    for flat_index in differing_indexes[:limit]:
        flat_index = int(flat_index)
        if left_data.shape:
            index = tuple(int(i) for i in np.unravel_index(flat_index, left_data.shape))
            index_text = "[" + ", ".join(str(i) for i in index) + "]"
        else:
            index_text = "scalar"
        left_value = None if left_mask[flat_index] else left_flat[flat_index]
        right_value = None if right_mask[flat_index] else right_flat[flat_index]
        sample: dict[str, Any] = {
            "index": index_text,
            "left": value_for_display(left_value),
            "right": value_for_display(right_value),
        }
        if numeric and left_value is not None and right_value is not None:
            try:
                sample["absolute_difference"] = value_for_display(
                    abs(np.longdouble(right_value) - np.longdouble(left_value))
                )
            except (TypeError, ValueError, OverflowError):
                pass
        result["samples"].append(sample)
    result["samples_truncated"] = differing_indexes.size > limit
    return result


def variable_info(variable: Any) -> dict[str, Any]:
    return {
        "dtype": str(variable.dtype),
        "dimensions": list(variable.dimensions),
        "shape": list(variable.shape),
        "attributes": attributes(variable),
    }


def compare_variable(
    name: str,
    left_dataset: Any,
    right_dataset: Any,
    atol: float,
    rtol: float,
    max_differences: int,
    max_scatter_points: int | None = None,
) -> dict[str, Any]:
    left_exists = name in left_dataset.variables
    right_exists = name in right_dataset.variables
    result: dict[str, Any] = {"name": name}
    if left_exists:
        result["left"] = variable_info(left_dataset.variables[name])
    if right_exists:
        result["right"] = variable_info(right_dataset.variables[name])
    if not left_exists:
        result.update(status="only right", value_comparison=None, attribute_changes=[])
    elif not right_exists:
        result.update(status="only left", value_comparison=None, attribute_changes=[])
    else:
        left_variable = left_dataset.variables[name]
        right_variable = right_dataset.variables[name]
        value_comparison = compare_values(
            left_variable,
            right_variable,
            atol,
            rtol,
            max_differences,
            max_scatter_points=max_scatter_points,
        )
        attribute_changes = compare_attributes(
            result["left"]["attributes"], result["right"]["attributes"]
        )
        structure_changed = any(
            result["left"][key] != result["right"][key]
            for key in ("dtype", "dimensions", "shape")
        )
        if value_comparison.get("shape_mismatch"):
            status = "shape mismatch"
        elif value_comparison["changed"] or value_comparison["one_missing"] or structure_changed or attribute_changes:
            status = "different"
        else:
            status = "identical"
        result.update(
            status=status,
            value_comparison=value_comparison,
            attribute_changes=attribute_changes,
        )
    return result


def compare_pair(
    relative_path: str,
    left_path: Path,
    right_path: Path,
    atol: float,
    rtol: float,
    max_differences: int,
    max_scatter_points: int = 2000,
    common_variables_only: bool = False,
) -> dict[str, Any]:
    pair: dict[str, Any] = {
        "relative_path": relative_path,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "error": None,
    }
    try:
        with netCDF4.Dataset(left_path, "r") as left_dataset, netCDF4.Dataset(
            right_path, "r"
        ) as right_dataset:
            left_meta = metadata(left_dataset)
            right_meta = metadata(right_dataset)
            vdisquants = common_variables_only or is_vdisquants_pair(
                relative_path, left_meta, right_meta
            )
            variable_names = (
                set(left_dataset.variables) & set(right_dataset.variables)
                if common_variables_only
                else set(left_dataset.variables) | set(right_dataset.variables)
            )
            variables = [
                compare_variable(
                    name,
                    left_dataset,
                    right_dataset,
                    atol,
                    rtol,
                    max_differences,
                    max_scatter_points=(
                        max_scatter_points
                        if vdisquants
                        and name in left_dataset.variables
                        and name in right_dataset.variables
                        and is_vdisquants_moment(
                            name,
                            left_dataset.variables[name],
                            right_dataset.variables[name],
                        )
                        else None
                    ),
                )
                for name in sorted(variable_names)
            ]
            pair.update(
                left_format=left_dataset.data_model,
                right_format=right_dataset.data_model,
                vdisquants=vdisquants,
                common_variables_only=common_variables_only,
                left_metadata=left_meta,
                right_metadata=right_meta,
                dimension_changes=(
                    []
                    if common_variables_only
                    else compare_dimensions(left_meta["dimensions"], right_meta["dimensions"])
                ),
                global_attribute_changes=(
                    []
                    if common_variables_only
                    else compare_attributes(
                        left_meta["global_attributes"], right_meta["global_attributes"]
                    )
                ),
                variables=variables,
            )
            pair["status"] = (
                "different"
                if pair["dimension_changes"]
                or pair["global_attribute_changes"]
                or any(variable["status"] != "identical" for variable in variables)
                else "identical"
            )
    except Exception as exc:
        pair["status"] = "error"
        pair["error"] = f"{type(exc).__name__}: {exc}"
    return pair


def find_netcdf_files(directory: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in NETCDF_SUFFIXES:
            found[path.relative_to(directory).as_posix()] = path
    return found


def find_netcdf_files_by_date(directory: Path) -> dict[str, Path]:
    """Index files by their first YYYYMMDD token, rejecting ambiguous days."""
    found: dict[str, Path] = {}
    for relative_path, path in find_netcdf_files(directory).items():
        match = re.search(r"(?<!\d)(\d{8})(?!\d)", relative_path)
        if not match:
            continue
        date = match.group(1)
        if date in found:
            raise ValueError(
                f"multiple NetCDF files in {directory} have date {date}: "
                f"{found[date].name} and {path.name}"
            )
        found[date] = path
    return found


def badge(status: str) -> str:
    css_class = status.replace(" ", "-")
    return f'<span class="badge {html.escape(css_class)}">{html.escape(status)}</span>'


def code(value: Any) -> str:
    return f"<code>{html.escape(value_for_display(value))}</code>"


def change_table(changes: list[dict[str, Any]], left_label: str, right_label: str) -> str:
    if not changes:
        return '<p class="quiet">No changes.</p>'
    rows = []
    for change in changes:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(change['name']))}</td>"
            f"<td>{badge(change['status'])}</td>"
            f"<td>{code(change.get('left'))}</td>"
            f"<td>{code(change.get('right'))}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table><thead><tr><th>Name</th><th>Status</th>'
        f"<th>{html.escape(left_label)}</th><th>{html.escape(right_label)}</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>"
    )


def render_scatter_plot(
    comparison: dict[str, Any], variable: dict[str, Any], left_label: str, right_label: str
) -> str:
    left_values = comparison["left_points"]
    right_values = comparison["right_points"]
    if not comparison["count"]:
        return ""

    width, height = 620, 440
    plot_left, plot_top = 82, 20
    plot_width, plot_height = 510, 340
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height
    minimum = comparison["plot_min"]
    maximum = comparison["plot_max"]
    span = maximum - minimum
    padding = span * 0.05 if math.isfinite(span) and span > 0 else max(abs(minimum) * 0.05, 0.5)
    axis_min = minimum - padding
    axis_max = maximum + padding
    if not math.isfinite(axis_min):
        axis_min = minimum
    if not math.isfinite(axis_max):
        axis_max = maximum
    if axis_min == axis_max:
        axis_min, axis_max = minimum - 0.5, maximum + 0.5
    axis_span = axis_max - axis_min

    grid = []
    ticks = []
    for index in range(5):
        fraction = index / 4
        x = plot_left + fraction * plot_width
        y = plot_bottom - fraction * plot_height
        value = axis_min + fraction * axis_span
        label = html.escape(f"{value:.4g}")
        grid.append(
            f'<line x1="{x:.2f}" y1="{plot_top}" x2="{x:.2f}" y2="{plot_bottom}" />'
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" />'
        )
        ticks.append(
            f'<text x="{x:.2f}" y="{plot_bottom + 22}" text-anchor="middle">{label}</text>'
            f'<text x="{plot_left - 10}" y="{y + 4:.2f}" text-anchor="end">{label}</text>'
        )

    circles = []
    for left_value, right_value in zip(left_values, right_values):
        x = plot_left + (left_value - axis_min) / axis_span * plot_width
        y = plot_bottom - (right_value - axis_min) / axis_span * plot_height
        circles.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" '
            'fill="#2563a6" fill-opacity=".48" />'
        )

    left_name = Path(left_label).name or left_label
    right_name = Path(right_label).name or right_label
    variable_name = variable["name"]
    units = variable.get("left", {}).get("attributes", {}).get("units")
    if isinstance(units, bytes):
        units = units.decode("utf-8", errors="replace")
    units_text = f" ({units})" if units is not None else ""
    rmse = comparison["rmse"]
    r_squared = comparison["r_squared"]
    rmse_text = "undefined" if rmse is None else f"{rmse:.6g}{units_text}"
    r_squared_text = "undefined" if r_squared is None else f"{r_squared:.6g}"
    plotted_note = (
        f"The plot shows {comparison['plotted']:,} of {comparison['count']:,} paired observations."
        if comparison["plotted"] < comparison["count"]
        else f"All {comparison['count']:,} paired observations are plotted."
    )
    description = (
        f"{variable_name} scatter plot: {left_name} values on the x axis and "
        f"{right_name} values on the y axis."
    )

    metrics = (
        ("Paired finite values", f"{comparison['count']:,}"),
        ("RMSE", rmse_text),
        ("R² (right vs left)", r_squared_text),
    )
    metric_html = "".join(
        f'<div class="metric"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong></div>'
        for label, value in metrics
    )
    svg = (
        f'<svg class="scatter-svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(description)}">'
        f'<title>{html.escape(description)}</title>'
        f'<g class="scatter-grid">{"".join(grid)}</g>'
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_top}" '
        'class="one-to-one" />'
        f'<g>{"".join(circles)}</g>'
        f'<g class="scatter-ticks">{"".join(ticks)}</g>'
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="scatter-axis" />'
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" class="scatter-axis" />'
        f'<text x="{plot_left + plot_width / 2:.1f}" y="{height - 12}" text-anchor="middle" '
        f'class="scatter-label">{html.escape(left_name)} value{html.escape(units_text)}</text>'
        f'<text x="18" y="{plot_top + plot_height / 2:.1f}" text-anchor="middle" '
        f'class="scatter-label" transform="rotate(-90 18 {plot_top + plot_height / 2:.1f})">'
        f'{html.escape(right_name)} value{html.escape(units_text)}</text>'
        f'<line x1="{plot_right - 128}" y1="{plot_top + 12}" '
        f'x2="{plot_right - 100}" y2="{plot_top + 12}" class="one-to-one" />'
        f'<text x="{plot_right - 94}" y="{plot_top + 16}" class="scatter-legend">1:1 agreement</text>'
        "</svg>"
    )
    return (
        f'<div class="scatter-comparison"><h4>Moment scatter plot: '
        f'{html.escape(variable_name)}</h4><div class="metrics">{metric_html}</div>'
        f'<figure class="scatter-figure">{svg}<figcaption>{html.escape(plotted_note)} '
        'R² is the coefficient of determination with the left version as the reference; '
        'the dashed line marks 1:1 agreement.</figcaption></figure></div>'
    )


def render_variable(variable: dict[str, Any], left_label: str, right_label: str) -> str:
    name = html.escape(variable["name"])
    status = variable["status"]
    left = variable.get("left", {})
    right = variable.get("right", {})
    left_summary = (
        f"{html.escape(left.get('dtype', ''))} {html.escape(str(left.get('dimensions', [])))} "
        f"shape={html.escape(str(left.get('shape', [])))}"
        if left
        else "—"
    )
    right_summary = (
        f"{html.escape(right.get('dtype', ''))} {html.escape(str(right.get('dimensions', [])))} "
        f"shape={html.escape(str(right.get('shape', [])))}"
        if right
        else "—"
    )
    summary = (
        f"<div class=\"var-heading\"><span class=\"var-name\">{name}</span>{badge(status)}"
        f"<span class=\"var-side\">{html.escape(left_label)}: {left_summary}</span>"
        f"<span class=\"var-side\">{html.escape(right_label)}: {right_summary}</span></div>"
    )
    content = []
    value_comparison = variable.get("value_comparison")
    if value_comparison is None:
        content.append('<p>This variable exists in only one file.</p>')
    elif value_comparison.get("shape_mismatch"):
        content.append(
            "<p>Array values could not be compared element by element because their shapes differ: "
            f"{html.escape(str(value_comparison['left_shape']))} versus "
            f"{html.escape(str(value_comparison['right_shape']))}.</p>"
        )
    else:
        metrics = [
            ("Elements", value_comparison["total"]),
            ("Equal values", value_comparison["equal"]),
            ("Changed values", value_comparison["changed"]),
            ("Missing on one side", value_comparison["one_missing"]),
            ("Missing on both sides", value_comparison["both_missing"]),
        ]
        if value_comparison["max_abs_difference"] is not None:
            metrics.extend(
                [
                    ("Max absolute difference", value_comparison["max_abs_difference"]),
                    ("Mean absolute difference", value_comparison["mean_abs_difference"]),
                    ("Max relative difference", value_comparison["max_relative_difference"]),
                ]
            )
        metric_html = "".join(
            f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'
            for label, value in metrics
        )
        content.append(f'<div class="metrics">{metric_html}</div>')
        scatter_comparison = value_comparison.get("scatter_comparison")
        if scatter_comparison and scatter_comparison["count"]:
            content.append(
                render_scatter_plot(scatter_comparison, variable, left_label, right_label)
            )
        if value_comparison["samples"]:
            rows = []
            for sample in value_comparison["samples"]:
                delta = sample.get("absolute_difference", "—")
                rows.append(
                    "<tr>"
                    f"<td>{html.escape(sample['index'])}</td>"
                    f"<td>{code(sample['left'])}</td>"
                    f"<td>{code(sample['right'])}</td>"
                    f"<td>{code(delta)}</td>"
                    "</tr>"
                )
            sample_note = ""
            if value_comparison["samples_truncated"]:
                sample_note = " (additional changed positions are omitted; increase --max-differences or use 0 for all)"
            content.append(
                '<h4>Changed positions' + html.escape(sample_note) + "</h4>"
                '<div class="table-wrap"><table><thead><tr><th>Index</th>'
                f"<th>{html.escape(left_label)} value</th><th>{html.escape(right_label)} value</th>"
                "<th>Absolute difference</th></tr></thead><tbody>"
                + "".join(rows)
                + "</tbody></table></div>"
            )
        else:
            content.append('<p class="quiet">No differing values.</p>')
    if variable.get("attribute_changes"):
        content.append("<h4>Variable attribute changes</h4>")
        content.append(change_table(variable["attribute_changes"], left_label, right_label))
    return f"<details class=\"variable\" data-status=\"{html.escape(status)}\"><summary>{summary}</summary><div class=\"details-body\">{''.join(content)}</div></details>"


def render_pair(pair: dict[str, Any], left_label: str, right_label: str, index: int) -> str:
    title = html.escape(pair["relative_path"])
    summary = f"{badge(pair['status'])} {title}"
    if pair.get("error"):
        body = f'<p class="error">Could not compare this pair: {html.escape(pair["error"])}</p>'
    else:
        counts: dict[str, int] = {}
        for variable in pair["variables"]:
            counts[variable["status"]] = counts.get(variable["status"], 0) + 1
        count_text = ", ".join(f"{count} {status}" for status, count in sorted(counts.items())) or "no variables"
        has_scatter_values = any(
            (variable.get("value_comparison") or {})
            .get("scatter_comparison", {})
            .get("count", 0)
            for variable in pair["variables"]
        )
        scatter_note = (
            '<p class="quiet">No scatter plots or RMSE/R² values are available because this file '
            "pair has no finite paired moment values with compatible shapes.</p>"
            if pair.get("vdisquants") and not has_scatter_values
            else ""
        )
        sections = [
            f"<p><strong>Formats:</strong> {html.escape(pair['left_format'])} → {html.escape(pair['right_format'])}</p>",
            f"<p><strong>Variables:</strong> {html.escape(count_text)}</p>",
            "<h3>Dimension changes</h3>",
            change_table(pair["dimension_changes"], left_label, right_label),
            "<h3>Global attribute changes</h3>",
            change_table(pair["global_attribute_changes"], left_label, right_label),
            '<h3 class="variable-list-title">Variable comparisons</h3>',
            scatter_note,
            "".join(render_variable(variable, left_label, right_label) for variable in pair["variables"]),
        ]
        body = "".join(sections)
    return (
        f'<details class="file-pair" open data-status="{html.escape(pair["status"])}" id="file-{index}">'
        f"<summary>{summary}</summary><div class=\"details-body\">"
        f"<p class=\"paths\"><strong>{html.escape(left_label)}:</strong> {html.escape(pair['left_path'])}<br>"
        f"<strong>{html.escape(right_label)}:</strong> {html.escape(pair['right_path'])}</p>"
        f"{body}</div></details>"
    )


def render_unmatched(title: str, paths: list[str]) -> str:
    if not paths:
        return ""
    items = "".join(f"<li><code>{html.escape(path)}</code></li>" for path in paths)
    return f"<section><h2>{html.escape(title)} ({len(paths)})</h2><ul>{items}</ul></section>"


def render_report(
    left_root: Path,
    right_root: Path,
    left_files: dict[str, Path],
    right_files: dict[str, Path],
    pairs: list[dict[str, Any]],
    atol: float,
    rtol: float,
    match_by_date: bool = False,
    common_variables_only: bool = False,
) -> str:
    matched = len(pairs)
    different = sum(pair["status"] == "different" for pair in pairs)
    errors = sum(pair["status"] == "error" for pair in pairs)
    identical = sum(pair["status"] == "identical" for pair in pairs)
    left_only = sorted(set(left_files) - set(right_files))
    right_only = sorted(set(right_files) - set(left_files))
    left_label, right_label = str(left_root), str(right_root)
    pair_html = "".join(
        render_pair(pair, left_label, right_label, index) for index, pair in enumerate(pairs)
    )
    generated = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    pairing_note = (
        "Files are paired by the YYYYMMDD date in their filenames. "
        if match_by_date
        else "Files are paired by relative path. "
    )
    variable_note = (
        "Only variables present in both files are included. "
        if common_variables_only
        else ""
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NetCDF directory comparison</title>
<style>
:root {{ color-scheme: light; --ink:#19212a; --muted:#647180; --line:#d9e0e7; --panel:#fff; --bg:#f3f6f9; --green:#197447; --red:#ad2d29; --amber:#8c5b00; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1280px; margin:0 auto; padding:32px 22px 64px; }} h1 {{ margin:0 0 8px; font-size:2rem; }} h2 {{ margin:30px 0 12px; }} h3 {{ margin:24px 0 8px; font-size:1.1rem; }} h4 {{ margin:18px 0 8px; }} .lede,.quiet,.paths {{ color:var(--muted); }} .roots {{ overflow-wrap:anywhere; margin:18px 0; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:12px; margin:24px 0; }} .card,.file-pair,section {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; }} .card {{ padding:15px; }} .card span {{ display:block; color:var(--muted); font-size:.88rem; }} .card strong {{ font-size:1.55rem; }}
.toolbar {{ display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin:24px 0 14px; }} input[type=search] {{ min-width:240px; flex:1; border:1px solid var(--line); border-radius:7px; padding:10px 12px; font:inherit; }} button {{ border:1px solid var(--line); border-radius:7px; background:#fff; padding:9px 12px; cursor:pointer; font:inherit; }} .file-pair {{ margin:12px 0; overflow:hidden; }} summary {{ cursor:pointer; padding:15px 18px; font-weight:650; }} .file-pair > summary {{ background:#fff; }} .details-body {{ padding:0 18px 18px; }} .paths {{ overflow-wrap:anywhere; }}
.badge {{ display:inline-block; margin-right:8px; border-radius:999px; padding:2px 9px; background:#edf0f3; color:#39434d; font-size:.78rem; font-weight:700; text-transform:capitalize; vertical-align:middle; }} .badge.identical {{ background:#e4f4eb; color:var(--green); }} .badge.different,.badge.error,.badge.shape-mismatch {{ background:#ffebe8; color:var(--red); }} .badge.only-left,.badge.only-right {{ background:#fff2d7; color:var(--amber); }}
.variable {{ border-top:1px solid var(--line); }} .variable > summary {{ padding:12px 8px; }} .var-heading {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; }} .var-name {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.96rem; }} .var-side {{ color:var(--muted); font-size:.8rem; font-weight:400; overflow-wrap:anywhere; }} .details-body .variable {{ margin-left:8px; }} .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(135px,1fr)); gap:8px; margin:14px 0; }} .metric {{ background:#f5f7f9; border-radius:7px; padding:9px 11px; overflow-wrap:anywhere; }} .metric span {{ display:block; color:var(--muted); font-size:.78rem; }} .metric strong {{ font-size:.98rem; }}
.scatter-comparison {{ margin:20px 0 24px; }} .scatter-comparison h4 {{ margin-bottom:4px; }} .scatter-figure {{ margin:0; padding:10px 12px 8px; background:#fbfcfd; border:1px solid var(--line); border-radius:8px; overflow:auto; }} .scatter-svg {{ display:block; width:min(100%,620px); min-width:450px; height:auto; margin:0 auto; font:11px system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }} .scatter-grid line {{ stroke:#e4e9ef; stroke-width:1; }} .scatter-ticks {{ fill:var(--muted); }} .scatter-axis {{ stroke:#7b8794; stroke-width:1.2; }} .one-to-one {{ stroke:#687584; stroke-width:1.5; stroke-dasharray:6 5; }} .scatter-label {{ fill:#39434d; font-size:12px; }} .scatter-legend {{ fill:var(--muted); font-size:11px; }} .scatter-figure figcaption {{ color:var(--muted); font-size:.82rem; margin:2px 8px 4px; }}
.table-wrap {{ overflow:auto; margin:8px 0 14px; }} table {{ border-collapse:collapse; width:100%; min-width:560px; }} th,td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }} th {{ background:#f5f7f9; font-size:.83rem; }} code {{ white-space:pre-wrap; overflow-wrap:anywhere; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88em; }} section {{ padding:5px 18px 14px; margin:16px 0; }} .error {{ color:var(--red); }} .foot {{ margin-top:30px; color:var(--muted); font-size:.85rem; }}
@media (max-width:700px) {{ main {{ padding:20px 12px 40px; }} .var-side {{ width:100%; }} }}
</style>
</head>
<body><main>
<h1>NetCDF directory comparison</h1>
<p class="lede">{pairing_note}{variable_note}Numeric values use atol={atol:g} and rtol={rtol:g}; NaN values in the same positions are treated as equal. Vdisquants moment fields include scatter plots and RMSE/R² from paired finite values.</p>
<p class="roots"><strong>Left:</strong> {html.escape(left_label)}<br><strong>Right:</strong> {html.escape(right_label)}<br><strong>Generated:</strong> {html.escape(generated)}</p>
<div class="cards">
<div class="card"><span>Matched files</span><strong>{matched}</strong></div><div class="card"><span>Identical files</span><strong>{identical}</strong></div><div class="card"><span>Different files</span><strong>{different}</strong></div><div class="card"><span>Comparison errors</span><strong>{errors}</strong></div><div class="card"><span>Left-only files</span><strong>{len(left_only)}</strong></div><div class="card"><span>Right-only files</span><strong>{len(right_only)}</strong></div>
</div>
<div class="toolbar"><input id="search" type="search" placeholder="Filter files and variables…" aria-label="Filter files and variables"><button id="expand">Expand all</button><button id="collapse">Collapse all</button><button id="differences">Show differences only</button></div>
{render_unmatched(f"Only in {left_label}", left_only)}
{render_unmatched(f"Only in {right_label}", right_only)}
<h2>Matched files</h2>
{pair_html or '<p class="quiet">No matching NetCDF files were found.</p>'}
<p class="foot">For large arrays, each variable shows complete comparison counts and up to the configured number of changed positions. Run with --max-differences 0 to include every changed position. Scatter plot metrics use all paired finite values; the plot displays up to --max-scatter-points observations.</p>
</main>
<script>
const search=document.getElementById('search');
search.addEventListener('input',()=>{{const q=search.value.toLowerCase();document.querySelectorAll('.file-pair').forEach(file=>{{const text=file.textContent.toLowerCase();file.hidden=!!q&&!text.includes(q);}});}});
document.getElementById('expand').addEventListener('click',()=>document.querySelectorAll('details').forEach(x=>x.open=true));
document.getElementById('collapse').addEventListener('click',()=>document.querySelectorAll('details').forEach(x=>x.open=false));
let diffOnly=false;document.getElementById('differences').addEventListener('click',event=>{{diffOnly=!diffOnly;event.currentTarget.textContent=diffOnly?'Show all':'Show differences only';document.querySelectorAll('.variable').forEach(x=>x.hidden=diffOnly&&x.dataset.status==='identical');document.querySelectorAll('.file-pair').forEach(x=>x.hidden=diffOnly&&x.dataset.status==='identical');}});
</script></body></html>"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare NetCDF variables and metadata in two directory trees; write an HTML report."
    )
    parser.add_argument("left_dir", type=Path, help="first/reference directory")
    parser.add_argument("right_dir", type=Path, help="second/comparison directory")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("netcdf_comparison_report.html"),
        help="HTML report path (default: ./netcdf_comparison_report.html)",
    )
    parser.add_argument("--atol", type=float, default=0.0, help="absolute tolerance for numeric comparisons (default: 0)")
    parser.add_argument("--rtol", type=float, default=0.0, help="relative tolerance for numeric comparisons (default: 0)")
    parser.add_argument(
        "--max-differences", type=int, default=1000,
        help="maximum changed positions shown per variable; use 0 to show all (default: 1000)",
    )
    parser.add_argument(
        "--max-scatter-points", type=int, default=2000,
        help="maximum paired values drawn per scatter plot (default: 2000)",
    )
    parser.add_argument(
        "--common-variables", action="store_true",
        help="compare only variables present in both files; omit file-wide metadata changes",
    )
    parser.add_argument(
        "--match-by-date", action="store_true",
        help="pair files by their YYYYMMDD filename token instead of relative path",
    )
    args = parser.parse_args(argv)
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    if args.max_differences < 0:
        parser.error("--max-differences must be non-negative")
    if args.max_scatter_points < 1:
        parser.error("--max-scatter-points must be at least 1")
    for root in (args.left_dir, args.right_dir):
        if not root.is_dir():
            parser.error(f"directory does not exist: {root}")
    return args


def main(argv: list[str] | None = None) -> int:
    if _IMPORT_ERROR is not None:
        print(
            "Missing dependency: install netCDF4 and numpy to compare NetCDF files "
            f"(import error: {_IMPORT_ERROR}).",
            file=sys.stderr,
        )
        return 2
    args = parse_args(argv)
    left_root, right_root = args.left_dir.resolve(), args.right_dir.resolve()
    if args.match_by_date:
        left_files = find_netcdf_files_by_date(left_root)
        right_files = find_netcdf_files_by_date(right_root)
    else:
        left_files = find_netcdf_files(left_root)
        right_files = find_netcdf_files(right_root)
    shared = sorted(set(left_files) & set(right_files))
    pairs = [
        compare_pair(
            relpath,
            left_files[relpath],
            right_files[relpath],
            args.atol,
            args.rtol,
            args.max_differences,
            args.max_scatter_points,
            args.common_variables,
        )
        for relpath in shared
    ]
    report = render_report(
        left_root,
        right_root,
        left_files,
        right_files,
        pairs,
        args.atol,
        args.rtol,
        args.match_by_date,
        args.common_variables,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    different = sum(pair["status"] == "different" for pair in pairs)
    errors = sum(pair["status"] == "error" for pair in pairs)
    print(f"Compared {len(shared)} matched NetCDF file(s): {different} different, {errors} error(s).")
    print(f"Report written to {args.output.resolve()}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
