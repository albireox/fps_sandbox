#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-21
# @Filename: gcam_focus.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib
import re

import numpy
import pandas
import seaborn
import sep
from astropy.io import fits
from matplotlib import pyplot as plt


RESULTS = pathlib.Path(__file__).parent / "../results/lco"
PIXEL_SCALE = 0.146

seaborn.set_theme()


def run_sep(path: pathlib.Path, threshold: float = 5, **sep_opts):
    """Substracts background and runs extraction."""

    match = re.search(r"gimg-gfa[1-6]s-([0-9]+).fits", str(path))
    if match:
        seq_no = int(match.group(1))
    else:
        raise ValueError

    data = fits.getdata(str(path)).astype("f8")
    header = fits.getheader(str(path), 1)

    back = sep.Background(data)
    rms = back.rms()

    stars = sep.extract(data - back, threshold, err=rms)

    df = pandas.DataFrame(stars)
    df["camera"] = int(header["CAMNAME"][3])
    df["seq_no"] = seq_no
    df["mjd"] = int(header["SJD"])
    df["m2piston"] = int(header["M2PISTON"])

    return df


def process_mjd(mjd: int):
    """Processes and MJD."""

    gcam = pathlib.Path(f"/data/gcam/{mjd}")
    files = gcam.glob("gimg-gfa[1-6]s-[0-9][0-9][0-9][0-9].fits")

    with multiprocessing.Pool(4) as pool:

        data_list = pool.map(run_sep, files)

    data = pandas.concat(data_list)

    outpath = RESULTS / "gcam_focus" / "sep" / f"sep_t5_default_{mjd}.h5"
    outpath.parent.mkdir(exist_ok=True, parents=True)

    data.to_hdf(str(outpath), "data")

    return data


def filter_data(data: pandas.DataFrame):
    """Excludes bad data."""

    ecc = numpy.sqrt(data.a**2 - data.b**2) / data.a

    # fmt: off
    filter = (
        (data.a * PIXEL_SCALE) < 5 &
        ((data.a * PIXEL_SCALE) > 0.4) &
        (data.cpeak < 60000) &
        (ecc < 0.7)
    )
    # fmt: on

    return data.loc[filter]


def get_fwhm(data: pandas.DataFrame, filter: bool = True):
    """Returns the FWHM in arcsec."""

    def _calc_fwhm(g):
        fwhm_pixel = numpy.mean(2 * numpy.sqrt(numpy.log(2) * (g.a**2 + g.b**2)))
        return PIXEL_SCALE * fwhm_pixel

    data = data.copy()
    data = data.set_index(["seq_no", "camera"])

    if filter:
        data = filter_data(data)

    data["fwhm"] = data.groupby(["seq_no", "camera"]).apply(_calc_fwhm)

    return data.reset_index()


def plot_focus(data: pandas.DataFrame):
    """Filters and plots focus data."""

    plt.ioff()

    mjd = data.mjd.iloc[0]

    OUTPATH = RESULTS / "gcam_focus" / "sep" / str(mjd)
    OUTPATH.mkdir(parents=True, exist_ok=True)

    fwhm = get_fwhm(data.copy())
    fwhm_piston = fwhm.groupby(["camera", "m2piston"]).apply(lambda g: g.fwhm.median())

    violin = seaborn.violinplot(x="camera", y="fwhm", data=fwhm)
    violin.set_title(str(mjd))
    violin_fig = violin.get_figure()
    violin_fig.savefig(OUTPATH / f"violin_{mjd}.pdf")

    for camera in fwhm.camera.unique():

        # FWHM per camera.
        fwhm_camera = fwhm.loc[fwhm.camera == camera]

        fig, ax = plt.subplots()

        seaborn.scatterplot(x="seq_no", y="fwhm", data=fwhm_camera, ax=ax)
        ax.set_title(f"{mjd} - Camera {camera}")
        ax.set_xlabel("Sequence number")
        ax.set_xlabel("FWHM [arcsec]")

        fig.savefig(OUTPATH / f"fwhm_camera_{camera}.pdf")

        plt.close("all")

    # FWHM per camera per piston position.
    fig, ax = plt.subplots()

    seaborn.lineplot(
        x="m2piston",
        y=fwhm_piston,
        data=fwhm_piston,
        hue="camera",
        palette="deep",
        ax=ax,
    )
    ax.set_title(f"{mjd}")
    ax.set_xlabel("M2 Piston [microns]")
    ax.set_ylabel("FWHM [arcsec]")

    fig.savefig(OUTPATH / "fwhm_piston.pdf")

    plt.close("all")

    # Fit parabolas to each camera.

    fig, ax = plt.subplots()

    camera_piston = []

    for camera in sorted(fwhm_piston.index.get_level_values(0).unique()):
        camera_data = fwhm_piston.loc[camera]

        x = camera_data.index
        y = camera_data.values

        a, b, c = numpy.polyfit(x, y, 2, full=False)  # type: ignore

        xrange = numpy.linspace(x.min() - 100, x.max() + 100, 1000)
        fx = numpy.polyval([a, b, c], xrange)

        x_min = -b / 2 / a
        print(f"Camera {camera}: {int(x_min)}")

        camera_piston.append(int(x_min))

        ax.plot(xrange, fx, label=f"Camera {camera}")

    ax.legend()
    ax.set_title(f"{mjd}")
    ax.set_xlabel("M2 Piston [microns]")
    ax.set_ylabel("FWHM [arcsec]")

    fig.savefig(OUTPATH / "fwhm_piston_fit.pdf")

    return numpy.array(camera_piston)
