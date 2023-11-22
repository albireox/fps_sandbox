#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-10-18
# @Filename: focus_sweep.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Sequence

import fitsio
import numpy
import pandas
import seaborn
from astropy.visualization import MinMaxInterval, SqrtStretch, imshow_norm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import UnivariateSpline

from coordio.extraction import extract_sources


OBSERVATORY: str = "apo"
MJD: int = 59787

# FRAMES: Sequence[int] = [33, 42]
# FRAMES: Sequence[int] = [96, 114]
# FRAMES: Sequence[int] = [281, 302]
FRAMES: Sequence[int] = [414, 435]

RESULTS = pathlib.Path(__file__).parents[2] / "results"


def get_filepath(camera: int, frameno: int):
    """Returns the path for a frame."""

    camera = int(camera)
    frameno = int(frameno)

    base = pathlib.Path(f"/data/gcam/{OBSERVATORY.lower()}/{MJD}")
    suffix = "n" if OBSERVATORY.lower() == "apo" else "s"

    frame_path = base / f"gimg-gfa{camera}{suffix}-{frameno:04d}.fits"

    if frame_path.exists():
        return frame_path
    return None


def get_data():
    """Retrieves focus sweep data."""

    output = RESULTS / f"focus_sweep_{MJD}_{FRAMES[0]}_{FRAMES[1]}.parquet"
    if output.exists():
        return pandas.read_parquet(output)

    data: list[pandas.DataFrame] = []

    for camera in range(1, 7):
        for frame in range(FRAMES[0], FRAMES[1] + 1):
            path = get_filepath(camera, frame)
            if path is None:
                continue

            sources = extract_sources(path)
            sources["camera"] = camera
            sources["frame"] = frame
            sources["fwhm"] *= 0.216

            header = fitsio.read_header(path, ext=1)
            ipa = header["IPA"]
            m2 = header["M2PISTON"]
            focus = header["FOCUS"]

            sources["ipa"] = ipa
            sources["m2"] = m2
            sources["focus"] = focus

            data.append(sources)

    all_sources = pandas.concat(data)
    all_sources.to_parquet(output)

    return all_sources


def fit_spline(data: pandas.DataFrame):
    """Fits data using a spline."""

    data = data.sort_values("m2")
    spl = UnivariateSpline(data.m2, data.fwhm)

    xfine = numpy.arange(data.m2.min(), data.m2.max(), 0.01)
    yfine = spl(xfine)

    xmin = xfine[numpy.argmin(yfine)]

    return xmin


def fit_parabola(data: pandas.DataFrame):
    """Fits a parabola to the data."""

    data = data.sort_values("m2")

    a, b, c = numpy.polyfit(data.m2, data.fwhm, 2, full=False)

    return -b / 2 / a


def plot_sweep(data: pandas.DataFrame):
    """Calculates the best focus and plots the focus sweep."""

    output = RESULTS / f"focus_sweep_{MJD}_{FRAMES[0]}_{FRAMES[1]}.pdf"

    fwhm = (
        data.groupby(["frame", "camera", "ipa", "m2"])
        .fwhm.apply(lambda g: numpy.percentile(g, 10))
        .reset_index()
    )

    fg = seaborn.lmplot(data=fwhm, x="m2", y="fwhm", hue="camera", order=2)
    fg.savefig(output)

    best = fwhm.groupby(["camera"]).apply(lambda g: fit_parabola(g))

    best = (best - best.loc[4]) * 5.94

    print(f"{MJD} - {FRAMES[0]}-{FRAMES[1]} (IPA={fwhm.iloc[0].ipa:.2f})")
    print(best)


def plot_cutouts(data: pandas.DataFrame):
    """Plots cutouts of the data."""

    data = data.sort_values(["m2", "camera"])

    data = data.groupby(["m2", "camera", "flux"]).first().reset_index()
    data = data.groupby(["m2", "camera"]).filter(lambda g: len(g) > 0)

    ncols = len(data.groupby(["m2"]))

    INCH = 5
    fig = plt.figure(figsize=(ncols * INCH, 6 * INCH), layout="constrained")
    gs = GridSpec(6, ncols, figure=fig)

    icol = 0
    m2_old = data.m2.iloc[0]

    for (m2, camera), g_data in data.groupby(["m2", "camera"]):
        if m2 != m2_old:
            icol += 1
            m2_old = m2

        g_data = g_data.sort_values("flux")
        g_data = g_data.loc[
            (g_data.x > 50) & (g_data.x < 2000) & (g_data.y > 50) & (g_data.y < 2000)
        ]

        source = g_data.iloc[0]

        x = int(source.x)
        y = int(source.y)

        file = get_filepath(camera, source.frame)
        raw_data = fitsio.read(file, "RAW")

        cutout = raw_data[y - 31 : y + 31, x - 31 : x + 31]  # type: ignore

        ax = fig.add_subplot(gs[int(camera) - 1, icol])
        imshow_norm(
            cutout,
            ax,
            origin="lower",
            interval=MinMaxInterval(),
            stretch=SqrtStretch(),
        )

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        if icol == 0:
            ax.set_ylabel(f"GFA{int(camera)}", fontdict={"size": 20})
        if int(camera) == 6:
            ax.set_xlabel(f"{m2:.2f}", fontdict={"size": 20})

    fig.savefig(
        RESULTS / f"focus_sweep_cutouts_{MJD}_{FRAMES[0]}_{FRAMES[1]}.pdf",
        pad_inches=0,
    )


if __name__ == "__main__":
    data = get_data()
    # plot_sweep(data)
    plot_cutouts(data)
