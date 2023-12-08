#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-04-11
# @Filename: reprocess_gimg.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from astropy.io import fits

from cherno.acquisition import Acquisition
from coordio.defaults import PLATE_SCALE


matplotlib.use("TkAgg")
seaborn.set_theme()

RESULTS = pathlib.Path(os.path.dirname(__file__)) / "../results"
DATA = pathlib.Path("/data/gcam")
MJD = 59560


async def reprocess_gimg():
    mjd_dir = DATA / str(MJD)
    files = list(mjd_dir.glob("gimg-*[0-9].fits"))

    max_no = max([int(str(f)[-9:-5]) for f in files])

    acquisition = Acquisition("APO")

    results = []

    for ii in range(1, max_no):
        images = list(mjd_dir.glob(f"gimg-*-{ii:04d}.fits"))
        proc_images = list(mjd_dir.glob(f"proc-gimg-*-{ii:04d}.fits"))

        if len(proc_images) == 0:
            continue

        proc_header = fits.getheader(proc_images[0], 1)

        if proc_header["SOLVED"] is False:
            continue

        try:
            rms = proc_header["RMS"]
        except KeyError:
            rms = -999.0

        gimg_results = [ii, MJD, rms]
        print(images)
        ast_solution = await acquisition.process(
            None,
            images,
            write_proc=False,
            overwrite=False,
            correct=False,
            full_correction=True,
            scale_rms=True,
        )

        if ast_solution.valid_solution is False:
            continue

        gimg_results.append(ast_solution.rms)

        acq_data = ast_solution.acquisition_data

        for camera in range(1, 7):
            found = False
            for data in acq_data:
                if data.camera == f"gfa{camera}":
                    try:
                        ast_solution_camera = await acquisition.fit([data])
                    except Exception:
                        break
                    gimg_results.append(ast_solution_camera.rms)
                    found = True
                    break

            if not found:
                gimg_results.append(numpy.nan)

        results.append(gimg_results)

    df = pandas.DataFrame.from_records(
        results,
        columns=[
            "image_no",
            "mjd",
            "proc_rms",
            "rms",
            "rms_1",
            "rms_2",
            "rms_3",
            "rms_4",
            "rms_5",
            "rms_6",
        ],
    )

    df.to_hdf(RESULTS / f"gimg_rms_per_camera_{MJD}.h5", "data")


def plot_gimg_rms():
    data = pandas.read_hdf(RESULTS / f"gimg_rms_{MJD}.h5", "data")

    plate_scale = PLATE_SCALE["APO"]

    data.rms = data.rms / plate_scale * 3600.0
    data.scaled_rms = data.scaled_rms / plate_scale * 3600.0
    data.rms_no_3 = data.rms_no_3 / plate_scale * 3600.0
    data.rms_no_6 = data.rms_no_6 / plate_scale * 3600.0

    data = data.loc[data.rms < 1.1]

    fig, ax = plt.subplots()

    ax.fill_between(data.image_no, data.scaled_rms, data.rms, color="y", alpha=0.4)
    ax.plot(data.image_no, data.rms, "r-", label="Current RMS")
    ax.plot(data.image_no, data.scaled_rms, "b-", label="Scaled RMS")

    ax.legend()

    ax.set_title(str(MJD))
    ax.set_xlabel("Image number")
    ax.set_ylabel("RMS [arcsec]")

    fig.savefig(RESULTS / f"rms_vs_scaled_rms_{MJD}.pdf")
    plt.close(fig)

    fig, ax = plt.subplots()

    ax.plot(data.image_no, data.scaled_rms, "b-", label="All cameras")
    ax.plot(data.image_no, data.rms_no_6, "r-", label="Without GFA6")
    ax.plot(data.image_no, data.rms_no_3, "y-", label="Without GFA3")

    ax.legend()

    ax.set_title(str(MJD))
    ax.set_xlabel("Image number")
    ax.set_ylabel("RMS [arcsec]")

    fig.savefig(RESULTS / f"scaled_rms_vs_no_gfa3_gfa6_{MJD}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    import asyncio

    asyncio.run(reprocess_gimg())

    # plot_gimg_rms()
