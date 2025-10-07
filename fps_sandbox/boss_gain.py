#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-09-16
# @Filename: boss_gain.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import numpy
from astropy.io import fits


REGIONS = [
    (2200, 4000, 200, 2000),
    (2200, 4000, 2400, 4000),
    (300, 2000, 2400, 4000),
    (300, 2000, 200, 2000),
]


def calculate_gain(
    exposure1: str,
    exposure2: str,
    bias: str,
    nbins: int = 20,
) -> list[float]:
    """Calculates the gain between two exposures.

    Parameters
    ----------
    exposure1
        The filename of the first exposure.
    exposure2
        The filename of the second exposure.
    bias
        The filename of the bias exposure.
    nbins
        The number of bins to use in each dimension.

    Returns
    -------
    float
        The calculated gain.

    """

    hdul1 = fits.open(exposure1)
    hdul2 = fits.open(exposure2)

    hdul_bias = fits.open(bias)

    data1 = hdul1[0].data.astype("f4") - hdul_bias[0].data.astype("f4")
    data2 = hdul2[0].data.astype("f4") - hdul_bias[0].data.astype("f4")

    if int(hdul1[0].header["EXPTIME"]) != int(hdul2[0].header["EXPTIME"]):
        raise ValueError("Exposures must have the same exposure time.")

    avg_data = 0.5 * (data1 + data2)
    diff_data = hdul1[0].data.astype("f4") - hdul2[0].data.astype("f4")

    gains = []

    for quad, qs in enumerate(REGIONS):
        data_quad = avg_data[qs[0] : qs[1], qs[2] : qs[3]]
        diff_quad = diff_data[qs[0] : qs[1], qs[2] : qs[3]]

        bin_gain = []

        i_ls = numpy.linspace(0, data_quad.shape[0], nbins + 1, dtype=int)
        j_ls = numpy.linspace(0, data_quad.shape[1], nbins + 1, dtype=int)

        for ii in range(len(i_ls) - 1):
            for jj in range(len(j_ls) - 1):
                bin_data = data_quad[i_ls[ii] : i_ls[ii + 1], j_ls[jj] : j_ls[jj + 1]]
                bin_diff = diff_quad[i_ls[ii] : i_ls[ii + 1], j_ls[jj] : j_ls[jj + 1]]

                mean = numpy.median(bin_data)

                if numpy.std(bin_data) / mean > 0.05:
                    continue

                var = numpy.std(bin_diff) ** 2 / 2
                bin_gain.append(mean / var)

        quad_gain = numpy.array(bin_gain).mean()
        quad_std = numpy.array(bin_gain).std()

        gains.append(quad_gain)

        print(f"Quad {quad}: Gain = {quad_gain:.2f} +/- {quad_std:.2f}")

    return gains
