#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-02-06
# @Filename: disabled_robots.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Literal, overload

import polars


DATA_PATH = pathlib.Path(__file__).parent / "../data"
CONFSUMMARIES = DATA_PATH / "confsummaries.parquet"


@overload
def disabled_robots(
    data: pathlib.Path | str | polars.DataFrame | None = None,
    return_count: Literal[False] = False,
) -> polars.DataFrame: ...


@overload
def disabled_robots(
    data: pathlib.Path | str | polars.DataFrame | None = None,
    return_count: Literal[True] = True,
) -> tuple[polars.DataFrame, polars.DataFrame]: ...


def disabled_robots(
    data: pathlib.Path | str | polars.DataFrame | None = None,
    return_count: bool = False,
) -> polars.DataFrame | tuple[polars.DataFrame, polars.DataFrame]:
    """Returns a data frame of disabled robots per MJD and observatory.

    Parameters
    ----------
    data
        A concatenated data frame with all the confSummary file information or
        the path to the file.
    return_count
        If :obj:`True`, returns a tuple with the data frame of disabled robots
        per MJD and observatory, and a data frame with the number of MJDs a robot
        was disabled.

    """

    if data is None:
        data = CONFSUMMARIES

    if not isinstance(data, polars.DataFrame):
        data = polars.read_parquet(data)

    # Cast to proper boolean type.
    d1 = data.cast(
        {
            "decollided": polars.Boolean,
            "valid": polars.Boolean,
            "too": polars.Boolean,
            "assigned": polars.Boolean,
            "on_target": polars.Boolean,
        }
    )

    pcol = polars.col
    over_c = ["MJD", "positionerId", "observatory"]
    hypot = (pcol.alpha.std().pow(2) + pcol.beta.std().pow(2)).sqrt()

    # Determine whether a robot was disabled for a given MJD and observatory.
    # We use two metrics: either the robot was never on target for that MJD (even if
    # it had assigned targets) or the standard deviation of the positioner coordinates
    # is below 0.5 for the entire night, which means that it didn't move.
    d2 = (
        d1.select(
            "MJD",
            "positionerId",
            "observatory",
            "valid",
            "on_target",
            "assigned",
            "alpha",
            "beta",
        )
        .filter(pcol.assigned)
        .with_columns(
            none_on_target=(pcol.on_target.not_().all()).over(over_c),
            positioner_std=(hypot).over(over_c),
        )
        .with_columns(disabled=(pcol.none_on_target | (pcol.positioner_std < 0.5)))
    )

    # Create a list of disabled robots per MJD and observatory.
    d3 = d2.filter(pcol.disabled).select("observatory", "MJD", "positionerId").unique()

    d4 = d3.sort("observatory", "MJD", "positionerId")

    if return_count:
        count = d3.group_by("positionerId").count().sort("positionerId")
        return d4, count

    return d4
