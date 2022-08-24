#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-18
# @Filename: focus_curve.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
from astropy.io import fits
from astropy.table import Table


matplotlib.use("MacOSX")

MJD = 59810


def collect_data():
    """Collects the focus data and produces a data frame."""

    files = pathlib.Path(f"/data/gcam/{MJD}").glob("proc-gimg-gfa[1-6]s*")

    # max_seq = 0
    # for file in files:
    #     match = re.search(r"proc-gimg-gfa[0-9]s-([0-9]+)\.fits", str(file))
    #     if match:
    #         seq = int(match.group(1))
    #         if seq > max_seq:
    #             max_seq = seq

    data = []

    for file in files:
        table_data = Table.read(str(file), 2).filled().as_array()
        focus = fits.getheader(str(file), 1).get("FOCUS", -999.0)
        df = pandas.DataFrame(table_data)
        df.loc[:, "focus"] = focus
        data.append(df)

    data = pandas.concat(data)

    data.set_index(["mjd", "exposure", "camera"], inplace=True)

    outpath = pathlib.Path(__file__).parents[1] / "data" / "lco"
    outpath.mkdir(exist_ok=True)

    data.to_hdf(str(outpath / f"gcam-{MJD}.h5"), "data")


def analyse_data():

    path = pathlib.Path(__file__).parents[1] / "data" / "lco" / f"gcam-{MJD}.h5"

    data = pandas.read_hdf(str(path))
    data = data.loc[data.fwhm < 5]

    fig, ax = plt.subplots(1)

    best = []
    for camera in [1, 2, 3, 4, 5, 6]:
        camera_data = data.loc[(slice(None), slice(None), camera), :]
        camera_median = camera_data.groupby("focus").min()

        focus = camera_median.index.get_level_values(0)
        fwhm = camera_median.fwhm

        a, b, c = numpy.polyfit(focus, fwhm, 2, full=False)
        x_min = -b / 2 / a

        best.append(x_min)

        print(f"Camera {camera}: {x_min}")

        x0 = numpy.min(focus)
        x1 = numpy.max(focus)
        xs = numpy.linspace(x0 - 0.1 * (x1 - x0), x1 + 0.1 * (x1 - x0))
        ys = a * xs**2 + b * xs + c
        ax.plot(
            xs,
            ys,
            lw=1.0,
            label=f"Camera {camera}",
        )

    best = numpy.array(best)
    best -= best[2]
    best *= 7.25
    print(numpy.round(best, 2))

    ax.legend()

    plt.show()


if __name__ == "__main__":
    # collect_data()
    analyse_data()
