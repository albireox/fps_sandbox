#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-12-10
# @Filename: boss_focus_temperature.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import polars
from astropy.time import Time
from rheader import read_header
from rich.progress import track


def collect_header_data(root: pathlib.Path | str):
    """Collects header data from all BOSS FITS files.

    Parameters
    ----------
    root
        The root directory containing BOSS FITS files. It expects subdirectories
        for each MJD and files in the format `sdR-r*-*.fit*`. Only the red camera
        files are parsed since the information we are interested in is the same
        for all cameras.

    """

    root_path = pathlib.Path(root)
    fits_files = list(root_path.glob("**/sdR-r*-*.fit*"))

    header_data = []
    for fits_file in track(fits_files, description="Collecting header data ..."):
        header = read_header(str(fits_file))

        try:
            data = (
                header["EXPOSURE"][0],
                header["FILENAME"][0],
                header["DATE-OBS"][0],
                header["MJD"][0],
                header["FLAVOR"][0],
                header["EXPTIME"][0],
                header["T_OUT"][0],
                header["T_IN"][0],
                header["T_PRIM"][0],
                header["T_CELL"][0],
                header["T_FLOOR"][0],
                header["T_TRUSS"][0],
                header["HARTMANN"][0],
                header["COLLA"][0],
                header["COLLB"][0],
                header["COLLC"][0],
                header["B2CAMT"][0],
                header["B2CAMH"][0],
                header["R2CAMT"][0],
                header["R2CAMH"][0],
                header["COLLT"][0],
                header["COLLH"][0],
            )
        except KeyError:
            continue

        header_data.append(data)

    df = polars.DataFrame(
        header_data,
        orient="row",
        schema={
            "exposure": polars.String,
            "filename": polars.String,
            "date-obs": polars.String,
            "mjd": polars.UInt32,
            "flavor": polars.String,
            "exptime": polars.Float32,
            "t_out": polars.Float32,
            "t_in": polars.Float32,
            "t_prim": polars.Float32,
            "t_cell": polars.Float32,
            "t_floor": polars.Float32,
            "t_truss": polars.Float32,
            "hartmann": polars.String,
            "colla": polars.Float32,
            "collb": polars.Float32,
            "collc": polars.Float32,
            "b2camt": polars.Float32,
            "b2camh": polars.Float32,
            "r2camt": polars.Float32,
            "r2camh": polars.Float32,
            "collt": polars.Float32,
            "collh": polars.Float32,
        },
    )

    return df


def add_hartmann_data(df: polars.DataFrame) -> polars.DataFrame:
    """Adds Hartmann data to the DataFrame.

    Parameters
    ----------
    df
        The input DataFrame.

    Returns
    -------
    polars.DataFrame
        The DataFrame with Hartmann data added.

    """

    df = df.sort(["mjd", "exposure"])

    is_hartmann: list[bool] = []
    first_after_hartmann: list[bool] = []
    time_after_hartmann: list[float] = []

    _first_after_hartmann = False
    _hartmann_time = 0.0
    _current_mjd = -1
    for row in df.iter_rows(named=True):
        _is_hartmann = row["hartmann"].strip().lower() != "out"
        is_hartmann.append(_is_hartmann)

        mjd = row["mjd"]
        if mjd != _current_mjd:
            _first_after_hartmann = False
            _hartmann_time = 0.0
            _current_mjd = mjd

        time = Time(row["date-obs"]).unix

        if _is_hartmann:
            _hartmann_time = time
            _first_after_hartmann = True

        time_after_hartmann.append(
            time - _hartmann_time if _hartmann_time > 0 else -1.0
        )

        if not _is_hartmann and _first_after_hartmann:
            first_after_hartmann.append(True)
            _first_after_hartmann = False
        else:
            first_after_hartmann.append(False)

    df = df.with_columns(
        polars.Series(is_hartmann).alias("is_hartmann"),
        polars.Series(first_after_hartmann).alias("first_after_hartmann"),
        polars.Series(time_after_hartmann).alias("time_after_hartmann"),
    )

    return df
