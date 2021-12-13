#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-13
# @Filename: plot_quickred.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy
import pandas
from astropy.io import fits


def plot_quickred():

    RANGE = range(39990016, 39990035)

    all_data = []
    for exp_no in RANGE:
        path = f"/data/apogee/quickred/59561/apq-{exp_no}.fits"

        hdus = fits.open(path)
        if hdus[0].header["EXPTYPE"] != "OBJECT":
            continue

        all_data.append(hdus[2].data)

    meas = pandas.DataFrame(numpy.concatenate(all_data))
    obj = meas.loc[meas.objtype == b"OBJECT"]

    detections = obj.loc[(obj.snr > 5)].copy()
    detections.sort_values("snr", inplace=True, ascending=False)
    detections.drop_duplicates("catalogid")

    plt.scatter(detections.ra, detections.dec)
    plt.show()


if __name__ == "__main__":
    plot_quickred()
