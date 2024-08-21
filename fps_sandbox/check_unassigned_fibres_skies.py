#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-08-12
# @Filename: check_unassigned_fibres_skies.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import numpy
import polars
from rich.progress import track
from target_selection.skies import is_valid_sky

from sdssdb.peewee.sdss5db import database
from sdsstools.utils import Timer


CATALOGUES = ["gaia_dr3_source", "twomass_psc", "tycho2"]
PWD = pathlib.Path(__file__)


def check_sky_fibres(
    skies: str | pathlib.Path | polars.DataFrame,
    observatory: str = "APO",
    n_sample: int = 100,
    design_mode_label: str = ".*",
):
    """Checks the known skies against ``is_valid_sky``."""

    if isinstance(skies, str) or isinstance(skies, pathlib.Path):
        skies = polars.read_parquet(skies)

    skies = skies.filter(polars.col.observatory == observatory, polars.col.valid)

    # Add design mode label information.
    design = polars.read_parquet(PWD.parents[1] / "data/design.parquet")
    design = design.cast({"design_id": polars.Int32})

    skies = skies.join(design, on="design_id")

    # Subselect on design mode label.
    skies = skies.filter(polars.col.design_mode_label.str.contains(design_mode_label))

    assert database.set_profile("tunnel_operations")

    configuration_ids = skies["configuration_id"].unique().sample(n_sample, seed=42)
    skies = skies.filter(polars.col.configuration_id.is_in(configuration_ids))

    epoch: float = skies[0, "epoch"]

    runtime: list[float] = []
    new_data: list[polars.DataFrame] = []

    gby = skies.group_by(polars.col.configuration_id)
    for _, data in track(gby, total=n_sample):
        with Timer() as timer:
            mask = is_valid_sky(
                data.select(polars.col("ra", "dec")).to_numpy(),
                database,
                catalogues=CATALOGUES,
                epoch=epoch,
            )

        runtime.append(timer.elapsed)

        data_masked = data.with_columns(
            is_valid_sky=polars.Series(mask, dtype=polars.Boolean)
        )
        new_data.append(data_masked)

    print(f"Average runtime: {numpy.mean(runtime):.2f}+/-{numpy.std(runtime):.2f} s")

    skies_mask = polars.concat(new_data)
    valid = skies_mask["is_valid_sky"].sum()
    print(f"Valid skies: {valid / len(skies_mask) * 100:.2f}%")

    return skies_mask
