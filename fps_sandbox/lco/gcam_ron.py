#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-27
# @Filename: gcam_ron.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

# type: ignore

from __future__ import annotations

import pathlib
import re

from typing import Any

import astropy.io.fits as fits
import numpy
import pandas
from astropy import time
from astropy.stats import SigmaClip


AMP_ROWS = [[10, 500], [520, 1010], [1040, 1500], [1600, 2000]]


def gcam_ron_data(path):
    """Analyses the RON properties for the GFAs."""

    path = pathlib.Path(path)
    all_files = path.glob("gimg-*-[0-9][0-9][0-9][0-9].fits")

    raw = []

    s = SigmaClip(1)

    for file_ in list(sorted(all_files))[0::5]:

        data: Any = fits.getdata(str(file_), 1)
        data = data.astype("f8")

        header: Any = fits.getheader(str(file_), 1)

        camera = int(header["CAMNAME"][3])
        seq_no = int(re.search(r"([0-9]{4})\.fits", str(file_)).group(1))

        obstime = time.Time(header["DATE-OBS"], format="iso").jd
        temperature = header["CCDTEMP"]

        for amp in range(4):
            rows = numpy.s_[AMP_ROWS[amp][0] : AMP_ROWS[amp][1]]

            # Global median and RMS
            global_clipped = s(data[rows, :], masked=False)
            if len(global_clipped) == 0:
                global_median = global_std = numpy.nan
            else:
                global_median = numpy.median(global_clipped)
                global_std = numpy.std(global_clipped)

            # "Overscan" median and RMS
            overscan1_clipped = s(data[rows, 0:50], masked=False)
            overscan2_clipped = s(data[rows, 2000:], masked=False)
            overscan_clipped = numpy.hstack((overscan1_clipped, overscan2_clipped))
            if len(overscan_clipped) == 0:
                overscan_median = overscan_std = numpy.nan
            else:
                overscan_median = numpy.median(overscan_clipped)
                overscan_std = numpy.std(overscan_clipped)

            raw.append(
                (
                    camera,
                    seq_no,
                    obstime,
                    amp + 1,
                    temperature,
                    global_median,
                    global_std,
                    overscan_median,
                    overscan_std,
                )
            )

    data = pandas.DataFrame(
        raw,
        columns=[
            "camera",
            "seq_no",
            "obstime",
            "amp",
            "temperature",
            "global_median",
            "global_std",
            "overscan_median",
            "overscan_std",
        ],
    )

    return data


def plot_ron_data(path: str | None = None):
    """Plots the data."""

    if path:
        RESULTS = pathlib.Path(path)
    else:
        RESULTS = pathlib.Path(__file__).parent / "../results/lco/gcam_ron"

    RESULTS.mkdir(exist_ok=True, parents=True)
