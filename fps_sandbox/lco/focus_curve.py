#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-18
# @Filename: focus_curve.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib
import re

from typing import cast

import numpy
import pandas
import seaborn
from astropy.io import fits
from astropy.stats.sigma_clipping import SigmaClip
from matplotlib import pyplot as plt

from coordio.extraction import extract_marginal


seaborn.set_theme()


MJD = 59959
FRAME0 = 14
FRAME1 = 23


def _extract(file_: str):
    """Extract sources."""

    image = cast(numpy.ndarray, fits.getdata(file_, 1))
    header = fits.getheader(file_, 1)
    extracted = extract_marginal(image, sigma_0=4)

    if len(extracted) > 0:
        extracted["frame"] = get_seqno(file_)
        extracted["m2"] = header.get("FOCUS", numpy.nan)
        extracted["camera"] = header.get("CAMNAME", "NA")
        return extracted

    return None


def get_seqno(path: pathlib.Path):
    """Returns the sequence number."""

    match = re.search(r"gimg-gfa[1-6]s-([0-9]+).fits", str(path))

    if match:
        return int(match.group(1))
    else:
        raise ValueError()


def extract_data():
    """Collects the focus data and produces a data frame."""

    files = pathlib.Path(f"/data/gcam/lco/{MJD}").glob("gimg-gfa[1-6]s*")
    files = [
        str(file_)
        for file_ in files
        if get_seqno(file_) >= FRAME0 and get_seqno(file_) <= FRAME1
    ]

    data_frames = []
    with multiprocessing.Pool(10) as pool:
        for result in pool.imap(_extract, files):
            if result is not None:
                data_frames.append(result)

    sources = pandas.concat(data_frames)
    sources = sources.loc[(sources.xfitvalid == 1) & (sources.yfitvalid == 1)]
    sources["std"] = numpy.average(
        sources.loc[:, ["xstd", "ystd"]],
        weights=1 / sources.loc[:, ["xrms", "yrms"]],
        axis=1,
    )
    sources["rms"] = numpy.average(sources.loc[:, ["xrms", "yrms"]])
    sources["fwhm"] = sources["std"] * 0.146 * 2.355

    return sources


def calculate_median(data: pandas.DataFrame):
    """Sigclips and calculates the median for each M2 position."""

    sigclip = SigmaClip(3)

    fwhm_median = data.groupby("m2").fwhm.apply(
        lambda x: numpy.median(sigclip(x, masked=False))  # type: ignore
    )

    fwhm_median = fwhm_median.to_frame()
    fwhm_median["std"] = data.groupby("m2").fwhm.apply(numpy.std)

    return fwhm_median


def fit_parabola(data: pandas.DataFrame):
    """Fits a parabola to the data."""

    medians = calculate_median(data)

    x = medians.index
    y = medians.fwhm
    w = 1 / medians["std"]
    a, b, c = numpy.polyfit(x, y, 2, w=w, full=False)

    return (a, b, c)


def plot_focus_curve(data: pandas.DataFrame, fit_coeffs: tuple[float, float, float]):
    """Plots the data and parabola."""

    # Plot data.
    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.scatter(data.m2, data.fwhm, s=15, color="m", ec="None", alpha=0.5)

    medians = calculate_median(data)
    ax.errorbar(
        medians.index,
        medians.fwhm,
        yerr=medians["std"],
        fmt="o",
        color="k",
        alpha=0.3,
    )

    # Plot parabola.
    a, b, c = fit_coeffs
    xmin = numpy.min(data.m2)
    xmax = numpy.max(data.m2)

    xx = numpy.arange(xmin - 50, xmax + 50, 1)
    yy = a * xx**2 + b * xx + c
    best_m2 = -b / 2 / a

    ax.plot(xx, yy, "r-")

    ax.set_ylim(0.5, None)
    ax.set_xlabel("M2 position")
    ax.set_xlabel("FWHM [arcsec]")
    ax.set_title(f"M2 position at best focus: {best_m2:.0f} microns")

    return


if __name__ == "__main__":
    extract_data()
