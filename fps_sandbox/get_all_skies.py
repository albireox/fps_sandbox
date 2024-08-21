#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-08-12
# @Filename: get_all_skies.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib

from typing import Any

import polars
from rich.progress import Progress

from jaeger.target.tools import read_confSummary


SDSSCORE_DIR = "/home/gallegoj/software/sdsscore"


def _get_data_one(file: pathlib.Path):
    """Processes one confSummaryF file."""

    columns = [
        "catalogid",
        "positionerId",
        "fiberType",
        "assigned",
        "on_target",
        "valid",
        "decollided",
        "racat",
        "deccat",
        "ra",
        "dec",
        "cadence",
        "firstcarton",
        "program",
    ]

    header, fibermap = read_confSummary(file)

    parent_configuration = header.get("parent_configuration", -999)
    dither_radius = header.get("dither_radius", -999.0)
    cloned_from = header.get("cloned_from", -999)

    if parent_configuration != -999 or dither_radius != -999.0 or cloned_from != -999:
        return None

    observatory = header["observatory"]
    design_id = header["design_id"]
    configuration_id = header["configuration_id"]
    epoch = header["epoch"]

    try:
        c_skies = (
            fibermap.filter(
                polars.col.category.str.contains("sky"),
                polars.col.on_target == 1,
            )
            .select(columns)
            .with_columns(
                fiberType=polars.col.fiberType.str.to_lowercase(),
                observatory=polars.lit(observatory, polars.String),
                design_id=design_id,
                configuration_id=configuration_id,
                epoch=epoch,
            )
            .to_dicts()
        )

        c_assigned = (
            fibermap.filter(polars.col.on_target == 1)
            .select(columns)
            .with_columns(
                fiberType=polars.col.fiberType.str.to_lowercase(),
                observatory=polars.lit(observatory, polars.String),
                design_id=design_id,
                configuration_id=configuration_id,
                epoch=epoch,
            )
            .to_dicts()
        )

        c_unassigned = (
            fibermap.filter(polars.col.on_target == 0)
            .select(columns)
            .with_columns(
                fiberType=polars.col.fiberType.str.to_lowercase(),
                observatory=polars.lit(observatory, polars.String),
                design_id=design_id,
                configuration_id=configuration_id,
                epoch=epoch,
            )
            .to_dicts()
        )

    except Exception as err:
        print(f"Error processing {file}: {err}")
        return None

    return c_skies, c_assigned, c_unassigned


def get_all_skies():
    """Compiles a list of sky fibres and non-assigned fibre positions."""

    confSummaryF = sorted(pathlib.Path(SDSSCORE_DIR).glob("**/confSummaryF*.par"))

    schema = [
        ("observatory", polars.String),
        ("design_id", polars.Int32),
        ("epoch", polars.Float32),
        ("configuration_id", polars.Int32),
        ("catalogid", polars.Int64),
        ("positionerId", polars.Int32),
        ("fiberType", polars.String),
        ("assigned", polars.Boolean),
        ("on_target", polars.Boolean),
        ("valid", polars.Boolean),
        ("decollided", polars.Boolean),
        ("racat", polars.Float64),
        ("deccat", polars.Float64),
        ("ra", polars.Float64),
        ("dec", polars.Float64),
        ("cadence", polars.String),
        ("firstcarton", polars.String),
        ("program", polars.String),
    ]

    skies: list[dict[str, Any]] = []
    assigned: list[dict[str, Any]] = []
    unassigned: list[dict[str, Any]] = []

    with Progress(expand=True) as progress:
        task = progress.add_task("[green]Processing...", total=len(confSummaryF))

        with multiprocessing.Pool(8) as pool:
            for data in pool.imap_unordered(_get_data_one, confSummaryF):
                progress.update(task, advance=1)

                if data is None:
                    continue

                c_skies, c_assigned, c_unassigned = data
                skies.extend(c_skies)
                assigned.extend(c_assigned)
                unassigned.extend(c_unassigned)

    skies_df = polars.DataFrame(skies, schema=schema)
    assigned_df = polars.DataFrame(assigned, schema=schema)
    unassigned_df = polars.DataFrame(unassigned, schema=schema)

    return skies_df, assigned_df, unassigned_df
