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
import pandas
import seaborn
from astropy.io import fits
from cherno.acquisition import Acquisition

from coordio.defaults import PLATE_SCALE


matplotlib.use("TkAgg")
seaborn.set_theme()

RESULTS = pathlib.Path(os.path.dirname(__file__)) / "../results"
DATA = pathlib.Path("/data/gcam")
MJD = 59658


async def reprocess_gimg():

    mjd_dir = DATA / str(MJD)
    files = list(mjd_dir.glob("gimg-*[0-9].fits"))

    max_no = max([int(str(f)[-9:-5]) for f in files])

    acquisition = Acquisition("APO")

    results = []

    for ii in range(1, max_no + 1):
        images = list(mjd_dir.glob(f"gimg-*-{ii:04d}.fits"))
        proc_images = list(mjd_dir.glob(f"proc-gimg-*-{ii:04d}.fits"))

        if len(proc_images) == 0:
            continue

        proc_header = fits.getheader(proc_images[0], 1)

        if proc_header["SOLVED"] is False:
            continue

        try:
            deltara = proc_header["DELTARA"]
            deltadec = proc_header["DELTADEC"]
            deltarot = proc_header["DELTAROT"]
            deltascl = proc_header["DELTASCL"]
            rms = proc_header["RMS"]
        except KeyError:
            continue

        ast_solution = await acquisition.process(
            None,
            images,
            write_proc=False,
            overwrite=False,
            correct=False,
            full_correction=True,
        )

        if ast_solution.valid_solution is False:
            continue

        acq_data = ast_solution.acquisition_data

        ast_solution_scale = await acquisition.fit(acq_data, scale_rms=True)

        acq_data_no_3 = [ad for ad in acq_data if int(ad.camera[-1]) != 3]
        ast_solution_no_3 = await acquisition.fit(acq_data_no_3, scale_rms=True)

        acq_data_no_6 = [ad for ad in acq_data if int(ad.camera[-1]) != 6]
        ast_solution_no_6 = await acquisition.fit(acq_data_no_6, scale_rms=True)

        results.append(
            (
                ii,
                MJD,
                deltara,
                deltadec,
                deltarot,
                deltascl,
                rms,
                ast_solution.delta_ra,
                ast_solution.delta_dec,
                ast_solution.delta_rot,
                ast_solution.delta_scale,
                ast_solution.rms,
                ast_solution_scale.rms,
                ast_solution_no_3.rms,
                ast_solution_no_6.rms,
            )
        )

    df = pandas.DataFrame.from_records(
        results,
        columns=[
            "image_no",
            "mjd",
            "proc_deltara",
            "proc_deltadec",
            "proc_deltarot",
            "proc_deltascl",
            "proc_rms",
            "deltara",
            "deltadec",
            "deltarot",
            "deltascl",
            "rms",
            "scaled_rms",
            "rms_no_3",
            "rms_no_6",
        ],
    )

    df.to_hdf(RESULTS / f"gimg_rms_{MJD}.h5", "data")


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

    # import asyncio
    # asyncio.run(reprocess_gimg())

    plot_gimg_rms()
