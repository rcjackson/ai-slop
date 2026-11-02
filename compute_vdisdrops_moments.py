#!/usr/bin/env python3
"""Compute vdisquants rainfall rate and total concentration from vdisdrops."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import netCDF4
import numpy as np


REPO_ROOT = Path.home() / "vdisquants"
sys.path.insert(0, str(REPO_ROOT))
from vdisquants.disdrodb_compat import (  # noqa: E402
    DIAMETER_BIN_CENTER_MM,
    DIAMETER_BIN_WIDTH_MM,
    VELOCITY_BIN_CENTER_MS,
    bin_vdisdrops,
    get_legacy_2dvd_effective_sampling_area,
)


def finite_data(dataset: netCDF4.Dataset, name: str) -> np.ndarray:
    values = np.ma.asarray(dataset.variables[name][:])
    return np.asarray(np.ma.filled(values, np.nan), dtype=float)


def compute_moments(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    with netCDF4.Dataset(path) as dataset:
        time = finite_data(dataset, "time")
        diameter = finite_data(dataset, "equivolumetric_sphere_diameter")
        velocity = finite_data(dataset, "fall_speed")
        qc_diameter = finite_data(dataset, "qc_equivolumetric_sphere_diameter")
        qc_velocity = finite_data(dataset, "qc_fall_speed")
        units = str(getattr(dataset.variables["time"], "units", "seconds since midnight"))

    valid = (
        np.isfinite(time)
        & np.isfinite(diameter)
        & np.isfinite(velocity)
        & np.isfinite(qc_diameter)
        & np.isfinite(qc_velocity)
    )
    time_valid = time[valid]
    diameter_valid = diameter[valid]
    velocity_valid = velocity[valid]
    qc_diameter_valid = qc_diameter[valid]
    qc_velocity_valid = qc_velocity[valid]
    if time_valid[-1] - time_valid[0] < 60.0:
        # The repository reader expects a positive span. Add an invalid
        # sentinel one minute later so the real observation remains in its
        # original minute without changing the spectrum.
        time_valid = np.append(time_valid, time_valid[-1] + 60.0)
        diameter_valid = np.append(diameter_valid, 1.0)
        velocity_valid = np.append(velocity_valid, 1.0)
        qc_diameter_valid = np.append(qc_diameter_valid, 1.0)
        qc_velocity_valid = np.append(qc_velocity_valid, 1.0)
    integration_time, spectrum = bin_vdisdrops(
        time_valid,
        diameter_valid,
        velocity_valid,
        qc_diameter_valid,
        qc_velocity_valid,
        sampling_interval=60.0,
    )
    # The repository's ARM reader computes Nd in m^-3 mm^-1 from counts,
    # measured fall speed, the 2DVD effective area, and the bin widths.
    area_mm2 = (100.0 - 0.5 * DIAMETER_BIN_CENTER_MM) ** 2
    diameter_width = DIAMETER_BIN_WIDTH_MM
    nd = 1e6 * np.sum(
        spectrum / VELOCITY_BIN_CENTER_MS[None, :, None], axis=1
    ) / (area_mm2[None, :] * diameter_width[None, :] * 60.0)
    # Match the moment fields consumed from PyDSD in vdisquants/__main__.py.
    total_concentration = np.sum(nd * diameter_width[None, :], axis=1)
    # Use the velocity-weighted drop flux directly. Substituting the ARM
    # reader's Nd expression leaves a count-based form of PyDSD calculate_RR.
    rain_rate = (
        np.pi
        / 6.0
        * 3.6e-3
        * np.sum(
            np.sum(spectrum, axis=1)
            * (DIAMETER_BIN_CENTER_MM[None, :] ** 3)
            * 1e6
            / (area_mm2[None, :] * 60.0),
            axis=1,
        )
    )
    total_drops = np.sum(spectrum, axis=(1, 2))
    invalid = total_drops <= 20
    rain_rate[invalid] = np.nan
    total_concentration[invalid] = np.nan
    return integration_time, rain_rate, total_concentration, units


def write_output(input_path: Path, output_path: Path) -> None:
    integration_time, rain_rate, total_concentration, time_units = compute_moments(input_path)
    date_token = next(token for token in input_path.name.split(".") if len(token) == 8)
    base_time = int(
        (dt.datetime.strptime(date_token, "%Y%m%d") - dt.datetime(1970, 1, 1)).total_seconds()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(output_path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("time", 1440)
        base = dataset.createVariable("base_time", "i4")
        base[:] = base_time
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = f"seconds since {date_token[:4]}-{date_token[4:6]}-{date_token[6:]} 00:00:00"
        time[:] = np.arange(1440, dtype=float) * 60.0
        rain = dataset.createVariable("rain_rate", "f4", ("time",), fill_value=-9999.0)
        rain.units = "mm/hour"
        rain.long_name = "Rainfall rate computed from vdisdrops"
        concentration = dataset.createVariable(
            "total_droplet_concentration", "f4", ("time",), fill_value=-9999.0
        )
        concentration.units = "1/m^3"
        concentration.long_name = "Total droplet concentration computed from vdisdrops"
        rain[:] = -9999.0
        concentration[:] = -9999.0
        indexes = np.rint(integration_time / 60.0).astype(int)
        valid = (
            (indexes >= 0)
            & (indexes < 1440)
            & np.isfinite(rain_rate)
            & np.isfinite(total_concentration)
        )
        rain[indexes[valid]] = rain_rate[valid].astype(np.float32)
        concentration[indexes[valid]] = total_concentration[valid].astype(np.float32)
        dataset.setncattr("source_file", str(input_path))
        dataset.setncattr("processing", "vdisquants disdrodb_compat binning and moment integrals")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    inputs = sorted(args.input_dir.glob("*.cdf")) + sorted(args.input_dir.glob("*.nc"))
    for input_path in inputs:
        date_token = next(token for token in input_path.name.split(".") if len(token) == 8)
        write_output(input_path, args.output_dir / f"bnfvdisquantsM1.c1.{date_token}.nc")
    print(f"Wrote {len(inputs)} daily moment files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
