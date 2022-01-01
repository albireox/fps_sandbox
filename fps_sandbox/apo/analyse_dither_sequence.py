#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-26
# @Filename: analyse_dither_sequence.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os

import numpy
import pandas
import seaborn
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt


# Sequence 1 59571
# SEQID = 1
# MJD = 59571
# PARENT_CONFIGURATION_ID = 582

# PARENT_FVC = [18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42]
# DITHER_FVC = [19, 21]

# Sequence 1 59575
SEQID = 1
MJD = 59575
PARENT_CONFIGURATION_ID = 736

PARENT_FVC = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
DITHER_FVC = [4, 6, 8, 10, 12, 14, 15, 18, 20]

DIRNAME = os.path.join(os.path.dirname(__file__), "../results")


def plot_parent_wok():
    """Plot of distances of successive visits to the parent configuration."""

    seaborn.set_theme("notebook", "white")
    fig, ax = plt.subplots()

    base_df = None
    diff_data = []
    radial_data = []

    for ii, fimg_no in enumerate(PARENT_FVC):
        df = pandas.DataFrame(
            fits.getdata(
                f"/data/fcam/{MJD}/proc-fimg-fvc1n-{fimg_no:04}.fits",
                "FIBERDATA",
            )
        )
        df.set_index(["positioner_id", "fibre_type"], inplace=True)
        df = df.loc[pandas.IndexSlice[:, "Metrology"], :]

        if ii == 0:
            base_df = df
        else:
            cols = ["xwok_measured", "ywok_measured"]
            diff = base_df.loc[:, cols] - df.loc[:, cols]

            for nn in range(len(diff)):
                row = diff.iloc[nn]
                diff_data.append((ii, "xwok", row.xwok_measured))
                diff_data.append((ii, "ywok", row.ywok_measured))

                brow = base_df.iloc[nn]
                r = numpy.sqrt(brow.xwok_measured ** 2 + brow.ywok_measured ** 2)
                dist = numpy.sqrt(row.xwok_measured ** 2 + row.ywok_measured ** 2)
                radial_data.append((ii, r, dist))

            ax.scatter(
                df.xwok_measured - base_df.xwok_measured,
                df.ywok_measured - base_df.ywok_measured,
                marker=".",
                s=5,
                edgecolor="None",
                c=numpy.sqrt(df.xwok_measured ** 2 + df.ywok_measured ** 2),
            )

    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel(r"$x_{\rm wok}$")
    ax.set_ylabel(r"$y_{\rm wok}$")
    ax.set_title(f"Sequence {SEQID} ({MJD})")
    fig.savefig(os.path.join(DIRNAME, f"parent_{MJD}_{SEQID}.pdf"))
    plt.close(fig)

    seaborn.set_theme("notebook", "whitegrid")

    diff_df = pandas.DataFrame(diff_data, columns=["iter", "Measurement", "diff"])
    ax = seaborn.boxplot(
        x="iter",
        y="diff",
        hue="Measurement",
        palette=["m", "g"],
        data=diff_df,
    )

    ax2 = ax.secondary_yaxis(
        "right",
        functions=(lambda x: x / 217.7358 * 3600.0, lambda x: x * 217.7358 / 3600.0),
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Difference [mm]")
    ax2.set_ylabel("Difference [arcsec]")
    ax.set_title(f"Sequence {SEQID} ({MJD})")
    ax.figure.savefig(os.path.join(DIRNAME, f"parent_box_{MJD}_{SEQID}.pdf"))
    plt.close(ax.figure)

    radial_df = pandas.DataFrame(radial_data, columns=["iter", "r", "dist"])
    radial_df.sort_values(["iter", "r"], inplace=True)

    ax = seaborn.lineplot(x="r", y="dist", hue="iter", data=radial_df)
    ax.figure.savefig(os.path.join(DIRNAME, f"parent_radial_{MJD}_{SEQID}.pdf"))
    ax.set_xlabel("Radial distance [mm]")
    ax.set_ylabel("Measurement distance [mm]")
    ax.set_title(f"Sequence {SEQID} ({MJD})")
    plt.close(ax.figure)


def plot_parent_positioner():
    """Plot of distances of successive visits to the parent configuration."""

    base_df = None
    diff_data = []

    for ii, fimg_no in enumerate(PARENT_FVC):
        data = Table(
            fits.getdata(
                f"/data/fcam/{MJD}/proc-fimg-fvc1n-{fimg_no:04}.fits",
                "POSANGLES",
            )
        )
        df = data.to_pandas()
        df.set_index("positionerID", inplace=True)

        if ii == 0:
            base_df = df
        else:
            cols = ["alphaReport", "betaReport"]
            diff = base_df.loc[:, cols] - df.loc[:, cols]

            for nn in range(len(diff)):
                row = diff.iloc[nn]
                diff_data.append((ii, "alpha", row.alphaReport))
                diff_data.append((ii, "beta", row.betaReport))

    seaborn.set_theme("notebook", "whitegrid")

    diff_df = pandas.DataFrame(diff_data, columns=["iter", "Measurement", "diff"])
    ax = seaborn.boxplot(
        x="iter",
        y="diff",
        hue="Measurement",
        palette=["m", "g"],
        data=diff_df,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Difference [deg]")
    ax.set_title(f"Sequence {SEQID} ({MJD})")
    ax.figure.savefig(os.path.join(DIRNAME, f"parent_box_positioner_{MJD}_{SEQID}.pdf"))
    plt.close(ax.figure)


if __name__ == "__main__":
    plot_parent_wok()
    plot_parent_positioner()
