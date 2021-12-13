#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-12
# @Filename: plot_guide_loop.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime
import os

import pandas
import seaborn
from matplotlib import pyplot as plt


seaborn.set_style("darkgrid")
seaborn.set_palette("deep")


FILE = "../data/astrometry_fit_59560.csv"


def plot_guide_loop():

    data = pandas.read_csv(os.path.join(os.path.dirname(__file__), FILE))

    fig, axes = plt.subplots(3, figsize=(15, 8))

    times = [
        datetime.datetime(2021, 12, 12, *list(map(int, t.split(":"))))
        for t in data.time
    ]

    axes[0].plot(times, data.delta_ra, zorder=10)
    axes[1].plot(times, data.delta_dec, zorder=10)
    # axes[2].plot(times, data.delta_rot, zorder=10)

    axes[0].hlines(
        0.0,
        times[0],
        times[-1],
        color="k",
        linestyle="dashed",
        linewidth=0.5,
        zorder=5,
    )

    axes[1].hlines(
        0.0,
        times[0],
        times[-1],
        color="k",
        linestyle="dashed",
        linewidth=0.5,
        zorder=5,
    )

    axes[2].hlines(
        0.0,
        times[0],
        times[-1],
        color="k",
        linestyle="dashed",
        linewidth=0.5,
        zorder=5,
    )

    axes[0].set_ylim(-1, 1)
    axes[1].set_ylim(-1, 1)
    axes[2].set_ylim(-0.1, 0.1)

    axes[2].set_xlabel("Time")

    axes[0].set_ylabel(r"$\Delta\alpha\ {\rm [arcsec]}$")
    axes[1].set_ylabel(r"$\Delta\delta\ {\rm [arcsec]}$")
    axes[2].set_ylabel(r"$\Delta {\rm rot\ [deg]}$")

    # plt.ioff()
    # plt.show()

    fig.savefig("../results/astrometry_fit_59560.pdf")


if __name__ == "__main__":
    plot_guide_loop()
