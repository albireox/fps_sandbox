#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-02-08
# @Filename: unused_stats.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import polars
import seaborn
from polars import col as pcol


DATA_PATH = pathlib.Path(__file__).parent / "../data"
RESULTS_PATH = pathlib.Path(__file__).parent / "../results"

CONFSUMMARIES = DATA_PATH / "confsummaries.parquet"
DESIGN_MODES = DATA_PATH / "design_modes.parquet"


def unused_stats(
    data: pathlib.Path | str | polars.DataFrame | None = None,
    observatory: str = "APO",
    design_mode: str | None = None,
    plot: bool = True,
    outpath: pathlib.Path | str | None = None,
) -> polars.DataFrame:
    """Plots the number of unused robots per MJD and observatory.

    Parameters
    ----------
    data
        A concatenated data frame with all the confSummary file information or
        the path to the file.
    observatory
        The observatory for which to plot the data.
    design_mode
        The design mode to use. If not provided, all design modes are used. If a
        design mode does not have data, no plots are generated.
    plot
        Whether to plot the results.
    outpath
        The path where to save the plots.

    """

    if data is None:
        data = CONFSUMMARIES

    if outpath is None:
        outpath = RESULTS_PATH / "unused_stats"
    outpath = pathlib.Path(outpath)
    outpath.mkdir(exist_ok=True, parents=True)

    observatory = observatory.upper()

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
            "is_dithered": polars.Boolean,
        }
    ).with_columns(
        pcol(
            "decollided",
            "valid",
            "too",
            "assigned",
            "on_target",
            "is_dithered",
        ).fill_null(False)
    )

    # Reject configurations without entries or with more entries than the expected
    # 1500 rows (not sure what causes this).
    d1 = d1.filter((pcol.positionerId.len() == 1500).over("configuration_id"))

    # Join with design modes
    design_modes = polars.read_parquet(DESIGN_MODES)
    d2 = d1.join(design_modes, on="design_id", how="left")

    if design_mode:
        d2 = d2.filter(pcol.design_mode == design_mode)

    # Reject designs and RS runs that are commissioning or not normal science.
    d3 = d2.filter(
        pcol.observatory == observatory,
        pcol.program != "commissioning",
        pcol.program != "",
        pcol.rs_version != "manual",
        pcol.design_mode.str.ends_with("eng").not_(),
        pcol.is_dithered.not_(),
    )

    # List of MJDs when new RS versions were deployed.
    rs_version = d3.group_by("rs_version").agg(pcol.MJD.min().alias("MJD"))

    # For each MJD and observatory, determine the percentage of robots that were
    # not on target, and the percentage that were disabled.
    n_confs = pcol.configuration_id.n_unique()
    n_robots = n_confs * 500

    # Fraction of robots that were assigned by robostrategy.
    assigned_frac = pcol.assigned.sum() / n_robots

    # Fraction of robots that were disabled.
    disabled_frac = pcol.disabled.sum() / 3 / n_robots

    # Fraction of robots that were assigned and on target.
    on_target_frac = (pcol.assigned & pcol.on_target).sum() / n_robots

    # Fraction of robots that were assigned but not on target or disabled.
    failed_frac = assigned_frac - (on_target_frac + disabled_frac)

    d4 = (
        d3.select(
            "MJD",
            "observatory",
            "on_target",
            "assigned",
            "disabled",
            "configuration_id",
            "rs_version",
        )
        .group_by("MJD")
        .agg(
            n_confs.alias("n_configurations"),
            (assigned_frac * 100).alias("assigned_perc"),
            (on_target_frac * 100).alias("on_target_perc"),
            (disabled_frac * 100).alias("disabled_perc"),
            (failed_frac * 100).alias("failed_perc"),
            pcol.rs_version.first().alias("rs_version"),
        )
        .sort("MJD")
    )

    # Filter out some MJDs with weird data or in which the percentage of failed
    # robots is negative.
    d5 = d4.filter(pcol.disabled_perc < 90, pcol.failed_perc >= 0)

    if d5.height == 0:
        return d5

    if not plot:
        return d5

    # Plot results.
    seaborn.set_theme()
    plt.ioff()

    fig, ax = plt.subplots(figsize=(30, 12))

    ax.fill_between(
        d5["MJD"],
        0,
        d5["disabled_perc"],
        color="r",
        edgecolor="w",
        lw=1,
        zorder=100,
        label="Disabled",
    )

    ax.fill_between(
        d5["MJD"],
        0,
        d5["disabled_perc"] + d5["failed_perc"],
        color="k",
        edgecolor="w",
        lw=1,
        zorder=90,
        label="Disabled + on_target=0",
    )

    ax.fill_between(
        d5["MJD"],
        0,
        d5["on_target_perc"],
        color="g",
        edgecolor="w",
        lw=1,
        zorder=80,
        label="On target",
    )

    ax.fill_between(
        d5["MJD"],
        0,
        d5["assigned_perc"],
        color="b",
        edgecolor="w",
        lw=1,
        zorder=70,
        label="Assigned",
    )

    # Add dates when new RS versions were deployed.
    for row in rs_version.iter_rows(named=True):
        vpos = 104
        if observatory == "LCO":
            if row["rs_version"] in ["zeta-3", "theta-3", "eta-5"]:
                vpos = 102

        ax.plot(
            [row["MJD"], row["MJD"]],
            [-2, 106],
            color="r",
            ls="--",
            lw=1.5,
            zorder=200,
        )
        ax.text(
            row["MJD"] + 3,
            vpos,
            row["rs_version"],
            ha="left",
            va="center",
            fontsize=10,
        )

    ax.grid(False)

    ax.set_ylim(-5, 110)

    ax.set_xlabel("MJD")
    ax.set_ylabel("Percentage")

    if design_mode is None:
        if observatory == "APO":
            ax.legend(bbox_to_anchor=(0.235, 0.5)).set_zorder(300)
        else:
            ax.legend(bbox_to_anchor=(0.25, 0.5)).set_zorder(300)

    ax.set_title(f"{observatory} — {design_mode or 'All design modes'}")

    fig.tight_layout()

    filename = (
        f"unused_stats_{observatory}"
        if design_mode is None
        else f"unused_stats_{observatory}_{design_mode}"
    )

    fig.savefig(outpath / f"{filename}.pdf")
    fig.savefig(outpath / f"{filename}.png", dpi=300)

    plt.close(fig)

    return d5


def plot_all_design_modes(
    data: pathlib.Path | str | polars.DataFrame | None = None,
    observatory: str = "APO",
    outpath: pathlib.Path | str | None = None,
) -> polars.DataFrame:
    """Plots the number of unused robots per MJD and observatory for all design modes.

    Parameters
    ----------
    data
        A concatenated data frame with all the confSummary file information or
        the path to the file.
    observatory
        The observatory for which to plot the data.
    outpath
        The path where to save the plots.

    """

    design_modes = polars.read_parquet(DESIGN_MODES)["design_mode"].unique().to_list()
    dm_data: list[polars.DataFrame] = []

    for design_mode in design_modes:
        if design_mode.endswith("_eng"):
            continue
        elif design_mode.endswith("_no_apogee_skies"):
            # Eventually we'll want to include these, but for now we'll skip them.
            continue

        dm = plot_unused_stats(
            data=data,
            observatory=observatory,
            design_mode=design_mode,
            outpath=outpath,
        ).with_columns(design_mode=polars.lit(design_mode))
        dm_data.append(dm)

    return polars.concat(dm_data).sort("design_mode", "MJD")
