#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-02-17
# @Filename: fvc_analysis.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages
from rich.progress import track


RESULTS = pathlib.Path(".").parent / "../results/"
MJD = 59626


def collect_fvc_data(mjd: int, overwrite: bool = False):
    """Collect FVC fibre data."""

    proc_fimg = pathlib.Path(f"/data/fcam/{mjd}").glob("proc-*.fits")

    outfile = RESULTS / f"fvc_data_{mjd}.hdf"
    if outfile.exists() and not overwrite:
        return

    dfs = []
    for fimg in proc_fimg:
        hdus = fits.open(fimg)

        match = re.match(r".+-([0-9]+)+\.fits", str(fimg))
        if not match:
            continue
        exp_no = int(match.group(1))

        fiber_data = pandas.DataFrame(hdus["FIBERDATA"].data)

        fiber_data["configuration_id"] = hdus[1].header["CONFIGID"]
        fiber_data["exposure_no"] = exp_no
        fiber_data["RMS"] = hdus[1].header["FITRMS"]

        if "dubious" not in fiber_data:
            fiber_data["dubious"] = fiber_data["mismatched"]

        dfs.append(fiber_data)

    if len(dfs) == 0:
        return

    fiber_data = pandas.concat(dfs)
    fiber_data.set_index(["configuration_id", "exposure_no"], inplace=True)

    outfile.unlink(True)
    fiber_data.to_hdf(outfile, "data")


def plot_histograms(
    mjd: int,
    overwrite: bool = False,
    chunks: None | list[tuple] = None,
):
    """Plot distance from measured to expected as a histogram."""

    mjd_output_file = RESULTS / f"fvc_histogram_{mjd}.pdf"
    if mjd_output_file.exists() and not overwrite:
        return

    data_file = RESULTS / f"fvc_data_{mjd}.hdf"
    if not data_file.exists():
        return

    fiber_data: pandas.DataFrame = pandas.read_hdf(data_file)
    fiber_data.sort_index(inplace=True)

    assigned = fiber_data.groupby(
        ["configuration_id", "exposure_no", "positioner_id"]
    ).filter(
        lambda g: g.assigned.any()
        & (g.offline == 0).all()
        & (g.dubious == 0).all()
        & (g.on_target == 1).any()
    )

    assigned = assigned.loc[assigned.fibre_type == "Metrology"]

    assigned["distance"] = (
        numpy.hypot(
            assigned.xwok - assigned.xwok_measured,
            assigned.ywok - assigned.ywok_measured,
        )
        * 1000.0
    )

    assigned = assigned.reset_index()
    assigned["exposure_idx"] = assigned.groupby(["configuration_id"])[
        "exposure_no"
    ].transform(lambda x: x.values - x.values[0] + 1)

    if chunks is None:
        cids = assigned.groupby("configuration_id").first().index

        mjd_output_file.unlink(True)

        with PdfPages(mjd_output_file) as pdf:
            for cid in cids:
                conf_data = assigned.loc[assigned.configuration_id == cid]

                fig, ax = plt.subplots()
                seaborn.histplot(
                    conf_data,
                    x="distance",
                    hue="exposure_idx",
                    element="step",
                    palette="deep",
                    multiple="dodge",
                    ax=ax,
                )

                ax.set_xlim(0, 150)

                rms = numpy.round(conf_data.RMS.min(), 2)
                reached = conf_data.groupby("exposure_idx").apply(
                    lambda g: sum(g.distance < 10) / g.distance.size * 100.0
                )
                reached_max = numpy.round(reached.max(), 1)

                ax.set_title(
                    rf"MDJ: {mjd} - Configuration ID: {cid} - RMS: {rms} $\rm\mu m$ - "
                    rf"{reached_max}\%"
                )
                ax.set_xlabel(r"Wok distance [$\rm\mu m$]")

                legend = ax.get_legend()
                legend.set_title(r"Exposure \#")

                pdf.savefig()
                plt.close(fig)

    else:
        with PdfPages(mjd_output_file) as pdf:
            for exp_nos in chunks:
                exp_data = assigned.loc[assigned.exposure_no.isin(exp_nos)]

                cid = exp_data.iloc[0].configuration_id
                fig, ax = plt.subplots()
                seaborn.histplot(
                    exp_data,
                    x="distance",
                    hue="exposure_no",
                    element="step",
                    palette="deep",
                    multiple="dodge",
                    ax=ax,
                )

                ax.set_xlim(0, 150)

                rms = numpy.round(exp_data.RMS.min(), 2)
                reached = exp_data.groupby("exposure_idx").apply(
                    lambda g: sum(g.distance < 10) / g.distance.size * 100.0
                )
                reached_max = numpy.round(reached.max(), 1)

                ax.set_title(
                    rf"MDJ: {mjd} - Configuration ID: {cid} - RMS: {rms} $\rm\mu m$ - "
                    rf"{reached_max}\%"
                )
                ax.set_xlabel(r"Wok distance [$\rm\mu m$]")

                legend = ax.get_legend()
                legend.set_title(r"Exposure \#")

                pdf.savefig()
                plt.close(fig)


def plot_all():
    """Combines all FVC data."""

    files = RESULTS.glob("fvc_data_[0-9]*.hdf")

    dfs: list[pandas.DataFrame] = []
    for file_ in files:
        dfs.append(pandas.read_hdf(file_).reset_index())

    fiber_data = pandas.concat(dfs)

    fiber_data = fiber_data.dropna(
        subset=["positioner_id", "exposure_no", "configuration_id"]
    )
    assigned = fiber_data.groupby("positioner_id").filter(
        lambda g: g.assigned.any() & (g.offline == 0).all() & (g.on_target == 1).any()
    )
    assigned = assigned.loc[assigned.fibre_type == "Metrology"]

    assigned["distance"] = (
        numpy.hypot(
            assigned.xwok - assigned.xwok_measured,
            assigned.ywok - assigned.ywok_measured,
        )
        * 1000.0
    )

    assigned["exposure_idx"] = assigned.groupby("configuration_id")[
        "exposure_no"
    ].transform(lambda x: x.values - x.values[0] + 1)

    assigned = assigned.loc[(assigned.exposure_idx > 0) & (assigned.exposure_idx <= 5)]
    assigned = assigned.groupby("configuration_id").filter(
        lambda g: (g.exposure_idx == 5).any()
    )

    fig, ax = plt.subplots()
    seaborn.histplot(
        data=assigned,
        x="distance",
        hue="exposure_idx",
        element="step",
        palette="deep",
        multiple="dodge",
        binwidth=1,
        stat="density",
        hue_order=[5, 4, 3, 2, 1],
        ax=ax,
    )

    ax.set_xlim(0, 80)
    ax.set_xlabel(r"Wok distance [$\rm\mu m$]")

    legend = ax.get_legend()
    ax.legend(
        legend.legendHandles[::-1],
        ["1", "2", "3", "4", "5"],
        title=r"Exposure \#",
    )

    fig.savefig(RESULTS / "fvc_loop_all.pdf")
    plt.close(fig)

    breakpoint()


def concat_all(filter: bool = True):
    """Concatenates all FVC data."""

    files = RESULTS.glob("fvc_data_[0-9]*.hdf")

    dfs: list[pandas.DataFrame] = []
    for file_ in files:
        dfs.append(pandas.read_hdf(file_).reset_index())

    fiber_data = pandas.concat(dfs)
    fiber_data: pandas.DataFrame = fiber_data.loc[fiber_data.configuration_id.notna()]
    fiber_data.loc[:, "configuration_id"] = pandas.to_numeric(
        fiber_data.configuration_id,
        downcast="integer",
    )
    fiber_data.drop(columns=["mismatched"], inplace=True)

    fiber_data["exposure_idx"] = fiber_data.groupby(
        "configuration_id"
    ).exposure_no.transform(lambda x: x.values - min(x.values) + 1)

    fiber_data["distance"] = (
        numpy.hypot(
            fiber_data.xwok - fiber_data.xwok_measured,
            fiber_data.ywok - fiber_data.ywok_measured,
        )
        * 1000.0
    )

    if filter:
        fiber_data = fiber_data.groupby("positioner_id").filter(
            lambda g: g.assigned.any()
            & (g.offline == 0).all()
            & (g.on_target == 1).any()
        )
        fiber_data = fiber_data.loc[fiber_data.fibre_type == "Metrology"]
        fiber_data.dropna(subset=["distance"], inplace=True)

    fiber_data.set_index(["configuration_id", "exposure_no"], inplace=True)
    fiber_data.sort_index(inplace=True)

    return fiber_data


if __name__ == "__main__":

    # for mjd in track(list(range(59600, 59629)), description="MJD"):
    #     # collect_fvc_data(mjd, overwrite=True)
    #     plot_histograms(mjd, overwrite=True)

    # mjd = 59632
    # collect_fvc_data(mjd)
    # plot_histograms(
    #     mjd,
    #     overwrite=True,
    #     chunks=[
    #         (10, 11, 12),
    #         (13, 14, 15),
    #         (16, 17, 18, 19),
    #         (20, 21),
    #         (22, 23, 24, 25),
    #         (26, 27),
    #         (28, 29, 30, 31),
    #         (32, 33, 34, 35, 36, 37, 38),
    #         (39, 40, 41, 42, 43, 44),
    #         (45, 46, 47, 48, 49),
    #         (50, 51, 52),
    #         (53, 54, 55),
    #         (56, 57, 58),
    #         (59, 60, 61, 62),
    #         (63, 64, 65),
    #         (66, 67, 68, 69),
    #         (70, 71),
    #         (72, 73, 74, 75),
    #         (76, 77, 78, 79),
    #     ],
    # )

    # mjd = 59629
    # collect_fvc_data(mjd)
    # plot_histograms(mjd, overwrite=True)

    plot_all()
