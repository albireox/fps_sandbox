#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-26
# @Filename: analyse_dither_sequence.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

import numpy
import pandas
import seaborn
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
from tqdm import tqdm

from sdsstools import yanny


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
                r = numpy.sqrt(brow.xwok_measured**2 + brow.ywok_measured**2)
                dist = numpy.sqrt(row.xwok_measured**2 + row.ywok_measured**2)
                radial_data.append((ii, r, dist))

            ax.scatter(
                df.xwok_measured - base_df.xwok_measured,
                df.ywok_measured - base_df.ywok_measured,
                marker=".",
                s=5,
                edgecolor="None",
                c=numpy.sqrt(df.xwok_measured**2 + df.ywok_measured**2),
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


def create_dataframe(start_mjd: int = 59564):
    """Creates a dataframe with all dither sequences."""

    sdsscore_dir = pathlib.Path(os.environ["SDSSCORE_DIR"])
    summaryF_files = sorted(sdsscore_dir.glob("apo/**/confSummaryF*"))
    summary_files = sorted(sdsscore_dir.glob("apo/**/confSummary-*"))

    parents = {}
    dithers = []

    for summary_file in tqdm(summaryF_files + summary_files):
        summary = yanny(str(summary_file))

        if "MJD" in summary and float(summary["MJD"]) < start_mjd:
            continue

        if "is_dithered" not in summary:
            continue

        fibermap = summary["FIBERMAP"]
        names = list(fibermap.dtype.names)
        names.remove("mag")
        fibermap = fibermap[names]

        df = pandas.DataFrame(fibermap)

        conf_id = int(summary["configuration_id"])

        if int(summary["is_dithered"]) == 0:
            isF = "confSummaryF" in str(summary_file)
            if (isF is True) or (isF is False and conf_id not in parents):
                parents[conf_id] = df
            continue

        if int(summary["parent_configuration"]) not in parents:
            continue

        df.loc[:, "configuration_id"] = conf_id
        df.loc[:, "parent_configuration"] = int(summary["parent_configuration"])
        df.loc[:, "dither_radius"] = float(summary["dither_radius"])
        df.loc[:, "mjd"] = int(summary["MJD"])
        df.loc[:, "racen"] = float(summary["raCen"])
        df.loc[:, "deccen"] = float(summary["decCen"])

        df = pandas.merge(
            df,
            parents[int(summary["parent_configuration"])],
            how="left",
            left_on=["positionerId", "fiberType"],
            right_on=["positionerId", "fiberType"],
            suffixes=[None, "_parent"],
        )

        dithers.append(df)

    dithers_df: pandas.DataFrame = pandas.concat(dithers)

    for col, dtype in dithers_df.dtypes.items():
        if dtype == object:
            dithers_df.loc[:, col] = dithers_df[col].str.decode("utf-8")

    dithers_df = dithers_df.set_index(["configuration_id", "positionerId", "fiberType"])

    dithers_df.to_hdf(os.path.join(DIRNAME, "../results/dither_data.h5"), "data")


def plot_wok_radec(
    xy_range=None,
    mjd_range=None,
    filename=None,
    s=5,
    use_cat=False,
):
    """Plots dither wok spread vs RA/Dec."""

    data = pandas.read_hdf(os.path.join(DIRNAME, "../results/dither_data.h5"))

    if filename is not None:
        outpath = pathlib.Path(DIRNAME) / "../results" / filename
        new_path = outpath.parent / outpath.stem
        ext = outpath.suffix
        if not new_path.parent.exists():
            new_path.parent.mkdir()
    else:
        new_path = ext = None

    data = data.loc[
        (data.dither_radius == 0.1)
        & (data.assigned == 1)
        & (data.on_target_parent == 1)
    ]
    data = data.loc[
        (data.parent_configuration <= 713) | (data.parent_configuration >= 736)
    ]

    if mjd_range is not None:
        if isinstance(mjd_range, (list, tuple)):
            data = data.loc[(data.mjd >= mjd_range[0]) & (data.mjd <= mjd_range[1])]
        else:
            raise ValueError("Invalid MJD range.")

    for fibre_type in ["APOGEE", "BOSS"]:
        if new_path is not None:
            new_path_ftype = str(new_path) + f"_{fibre_type}"

        dftype = data.loc[pandas.IndexSlice[:, :, fibre_type.upper()], :].copy()
        dftype.reset_index(inplace=True)

        dftype.loc[:, "xwok_diff"] = dftype.xFocal - dftype.xFocal_parent
        dftype.loc[:, "ywok_diff"] = dftype.yFocal - dftype.yFocal_parent

        if xy_range is not None:
            dftype = dftype.loc[
                (dftype.xwok_diff >= xy_range[0])
                & (dftype.xwok_diff <= xy_range[1])
                & (dftype.ywok_diff >= xy_range[0])
                & (dftype.ywok_diff <= xy_range[1])
            ]

        cos = numpy.cos(numpy.deg2rad(dftype.deccen))
        if use_cat:
            dftype = dftype.loc[(dftype.racat > 0) & (dftype.ra > 0)]
            ra_diff = (dftype.ra - dftype.racat) * cos
            dec_diff = dftype.dec - dftype.deccat
        else:
            dftype = dftype.loc[(dftype.ra_parent > 0) & (dftype.ra > 0)]
            ra_diff = (dftype.ra - dftype.ra_parent) * cos
            dec_diff = dftype.dec - dftype.dec_parent

        dftype.loc[:, "ra_diff"] = ra_diff * 3600.0
        dftype.loc[:, "dec_diff"] = dec_diff * 3600.0

        jg = seaborn.jointplot(
            x="xwok_diff",
            y="ywok_diff",
            hue="mjd",
            data=dftype,
            s=s,
            legend="full",
        )

        jg.set_axis_labels(r"$\Delta x_{\rm wok}$", r"$\Delta y_{\rm wok}$")

        legend = jg.figure.axes[0].get_legend()
        legend.set_title("MJD")

        jg.figure.suptitle(f"Dither scatter: {fibre_type}")

        if filename is not None:
            jg.figure.savefig(f"{str(new_path_ftype)}-wok{ext}", dpi=300)
        else:
            plt.show()

        plt.close("all")

        jg = seaborn.jointplot(
            x="ra_diff",
            y="dec_diff",
            hue="mjd",
            data=dftype,
            s=s,
            legend="full",
        )

        jg.set_axis_labels(r"$\Delta\alpha$", r"$\Delta\delta$")

        legend = jg.figure.axes[0].get_legend()
        legend.set_title("MJD")

        jg.figure.suptitle(f"Dither scatter: {fibre_type}")

        if filename is not None:
            jg.figure.savefig(f"{str(new_path_ftype)}-radec{ext}", dpi=300)
        else:
            plt.show()

        plt.close("all")

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        seaborn.scatterplot(
            x="xwok_diff",
            y="ywok_diff",
            hue="mjd",
            data=dftype,
            s=s,
            legend="full",
            ax=axes[0],
        )

        axes[0].set_xlabel(r"$\Delta x_{\rm wok}$")
        axes[0].set_ylabel(r"$\Delta y_{\rm wok}$")

        legend = axes[0].get_legend()
        legend.set_title("MJD")

        seaborn.scatterplot(
            x="ra_diff",
            y="dec_diff",
            hue="mjd",
            data=dftype,
            s=s,
            legend=False,
            ax=axes[1],
        )

        axes[1].set_xlabel(r"$\Delta\alpha$")
        axes[1].set_ylabel(r"$\Delta\delta$")

        fig.suptitle(f"Wok vs RA/Dec: {fibre_type}")

        if filename is None:
            plt.show()
        else:
            fig.savefig(f"{str(new_path_ftype)}-both{ext}", dpi=300)

        plt.close("all")


def plot_focal_plane(
    filename,
    xy_range=None,
    mjd_range=None,
    gridsize=25,
    use_cat=False,
):
    """Plots dither wok spread vs RA/Dec."""

    data = pandas.read_hdf(os.path.join(DIRNAME, "../results/dither_data.h5"))

    outpath = pathlib.Path(DIRNAME) / "../results" / filename
    if not outpath.parent.exists():
        outpath.parent.mkdir()

    data = data.loc[
        (data.dither_radius == 0.1)
        & (data.assigned == 1)
        & (data.on_target_parent == 1)
    ]
    data = data.loc[
        (data.parent_configuration <= 713) | (data.parent_configuration >= 736)
    ]

    if mjd_range is not None:
        if isinstance(mjd_range, (list, tuple)):
            data = data.loc[(data.mjd >= mjd_range[0]) & (data.mjd <= mjd_range[1])]
        else:
            raise ValueError("Invalid MJD range.")

    for fibre_type in ["APOGEE", "BOSS"]:
        ftype_outpath = outpath.with_stem(outpath.stem + f"_{fibre_type}")

        dftype = data.loc[pandas.IndexSlice[:, :, fibre_type.upper()], :].copy()
        dftype.reset_index(inplace=True)

        dftype.loc[:, "delta_xwok"] = dftype.xFocal - dftype.xFocal_parent
        dftype.loc[:, "delta_ywok"] = dftype.yFocal - dftype.yFocal_parent

        if xy_range is not None:
            dftype = dftype.loc[
                (dftype.xwok_diff >= xy_range[0])
                & (dftype.xwok_diff <= xy_range[1])
                & (dftype.ywok_diff >= xy_range[0])
                & (dftype.ywok_diff <= xy_range[1])
            ]

        dftype = dftype.loc[(dftype.racat > 0) & (dftype.ra > 0)]

        cos = numpy.cos(numpy.deg2rad(dftype.deccen))
        if use_cat is False:
            ra_diff = (dftype.ra - dftype.ra_parent) * cos
            dec_diff = dftype.dec - dftype.dec_parent
        else:
            ra_diff = (dftype.ra - dftype.racat) * cos
            dec_diff = dftype.dec - dftype.deccat
        dftype.loc[:, "delta_racat"] = ra_diff * 3600.0
        dftype.loc[:, "delta_deccat"] = dec_diff * 3600.0

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        cmap = seaborn.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

        hex_ra = axes[0].hexbin(
            x=dftype.xFocal,
            y=dftype.yFocal,
            C=dftype.delta_racat,
            gridsize=gridsize,
            cmap=cmap,
            reduce_C_function=numpy.median,
        )
        cmap_ra = plt.colorbar(hex_ra, ax=axes[0])
        cmap_ra.set_label(r"$\Delta\alpha\ {\rm [arcsec]}$")
        axes[0].set_xlabel(r"$x_{\rm focal}$")
        axes[0].set_ylabel(r"$y_{\rm focal}$")

        hex_dec = axes[1].hexbin(
            x=dftype.xFocal,
            y=dftype.yFocal,
            C=dftype.delta_deccat,
            gridsize=gridsize,
            cmap=cmap,
            reduce_C_function=numpy.median,
        )
        cmap_dec = plt.colorbar(hex_dec, ax=axes[1])
        cmap_dec.set_label(r"$\Delta\delta\ {\rm [arcsec]}$")
        axes[1].set_xlabel(r"$x_{\rm focal}$")
        axes[1].set_ylabel(r"$y_{\rm focal}$")

        fig.suptitle(fibre_type)

        seaborn.despine()
        plt.tight_layout()

        fig.savefig(ftype_outpath, dpi=300)

        plt.close("all")


if __name__ == "__main__":
    # plot_parent_wok()
    # plot_parent_positioner()

    # create_dataframe()

    # plot_wok_radec(
    #     mjd_range=[59575, 59579],
    #     filename="dithers/dithers_59575-59579.png",
    # )
    # plot_wok_radec(
    #     mjd_range=[59575, 59579],
    #     use_cat=True,
    #     filename="dithers/dithers_59575-59579_cat.png",
    # )

    # plot_wok_radec(
    #     mjd_range=[59575, 59579],
    #     xy_range=[-2, 2],
    #     filename="dithers/dithers_59575-59579_zoom2.png",
    # )
    # plot_wok_radec(
    #     mjd_range=[59575, 59579],
    #     use_cat=True,
    #     xy_range=[-2, 2],
    #     filename="dithers/dithers_59575-59579_zoom2_cat.png",
    # )

    # plot_wok_radec(
    #     mjd_range=[59575, 59579],
    #     xy_range=[-0.5, 0.5],
    #     filename="dithers/dithers_59575-59579_zoom05.png",
    # )
    # plot_wok_radec(
    #     mjd_range=[59575, 59579],
    #     use_cat=True,
    #     xy_range=[-0.5, 0.5],
    #     filename="dithers/dithers_59575-59579_zoom05_cat.png",
    # )

    # plot_wok_radec(
    #     mjd_range=[59575, 59579],
    #     xy_range=[-0.25, 0.25],
    #     filename="dithers/dithers_59575-59579_zoom025.png",
    # )
    # plot_wok_radec(
    #     mjd_range=[59575, 59579],
    #     use_cat=True,
    #     xy_range=[-0.25, 0.25],
    #     filename="dithers/dithers_59575-59579_zoom025_cat.png",
    # )

    plot_focal_plane(
        "dithers/dithers_59575-59579_focal.pdf",
        mjd_range=[59575, 59579],
    )
    plot_focal_plane(
        "dithers/dithers_59575-59579_focal_cat.pdf",
        mjd_range=[59575, 59579],
        use_cat=True,
    )
