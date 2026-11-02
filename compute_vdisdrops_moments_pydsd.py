#!/usr/bin/env python3
"""Run the PyDSD rustmatrix-branch processing path on ARM vdisdrops files."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import netCDF4
import numpy as np
from pydsd.io.ARM_vdisdrops_reader import read_arm_vdisdrops_netcdf
from pydsd.utility import filter


def process_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dsd = read_arm_vdisdrops_netcdf(path, sampling_interval=60.0)
    filter.filter_spectrum_with_parsivel_matrix(dsd, 0.5, 0.5, replace=True)
    dsd.calculate_dsd_from_spectrum()
    dsd.set_canting_angle(7)
    dsd.calculate_dsd_parameterization()
    dsd.calculate_RR()
    rain_rate = np.asarray(dsd.fields["rain_rate"]["data"].data, dtype=float)
    total_concentration = np.asarray(dsd.fields["Nt"]["data"].data, dtype=float)
    drop_spectrum = np.asarray(dsd.fields["drop_spectrum"]["data"], dtype=float)
    filter.filter_nd_on_dropsize(dsd, 0.3, 5, replace=True)
    total_drops = np.sum(drop_spectrum, axis=tuple(range(1, drop_spectrum.ndim)))
    invalid = total_drops <= 20
    rain_rate[invalid] = np.nan
    total_concentration[invalid] = np.nan
    integration_time = np.asarray(dsd.time["data"], dtype=float)
    return integration_time, rain_rate, total_concentration


def write_output(input_path: Path, output_path: Path) -> None:
    date_token = next(token for token in input_path.name.split(".") if len(token) == 8)
    rain = np.full(1440, -9999.0, dtype=np.float32)
    concentration = np.full(1440, -9999.0, dtype=np.float32)
    try:
        integration_time, rain_rate, total_concentration = process_file(input_path)
        indexes = np.rint(integration_time / 60.0).astype(int)
        valid = (
            (indexes >= 0)
            & (indexes < 1440)
            & np.isfinite(rain_rate)
            & np.isfinite(total_concentration)
        )
        rain[indexes[valid]] = rain_rate[valid].astype(np.float32)
        concentration[indexes[valid]] = total_concentration[valid].astype(np.float32)
    except (IndexError, ValueError, RuntimeError) as exc:
        print(f"warning: {input_path.name}: {type(exc).__name__}: {exc}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(output_path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.createDimension("time", 1440)
        base = dataset.createVariable("base_time", "i4")
        base[:] = int(
            (dt.datetime.strptime(date_token, "%Y%m%d") - dt.datetime(1970, 1, 1)).total_seconds()
        )
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = f"seconds since {date_token[:4]}-{date_token[4:6]}-{date_token[6:]} 00:00:00"
        time[:] = np.arange(1440, dtype=float) * 60.0
        rain_var = dataset.createVariable("rain_rate", "f4", ("time",), fill_value=-9999.0)
        rain_var.units = "mm/hour"
        rain_var.long_name = "Rainfall rate computed by PyDSD"
        concentration_var = dataset.createVariable(
            "total_droplet_concentration", "f4", ("time",), fill_value=-9999.0
        )
        concentration_var.units = "1/m^3"
        concentration_var.long_name = "Total droplet concentration computed by PyDSD"
        rain_var[:] = rain
        concentration_var[:] = concentration
        dataset.setncattr("source_file", str(input_path))
        dataset.setncattr("processing", "PyDSD rustmatrix branch reader and moment methods")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    inputs = sorted(args.input_dir.glob("*.cdf")) + sorted(args.input_dir.glob("*.nc"))
    for input_path in inputs:
        date_token = next(token for token in input_path.name.split(".") if len(token) == 8)
        write_output(input_path, args.output_dir / f"bnfvdisquantsM1.c1.{date_token}.nc")
    print(f"Wrote {len(inputs)} daily PyDSD moment files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
