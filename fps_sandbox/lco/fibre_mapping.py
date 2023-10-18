#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-26
# @Filename: fibre_mapping.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import matplotlib
import numpy
from astropy.io import fits
from matplotlib import pyplot as plt

from coordio import calibration


matplotlib.use("agg")


def plot_data():
    DIR = "/Users/gallegoj/Downloads/59818/"
    MTP_FILES = {
        1: "proc-fimg-fvc1s-0003.fits",
        2: "proc-fimg-fvc1s-0004.fits",
        3: "proc-fimg-fvc1s-0013.fits",
        4: "proc-fimg-fvc1s-0014.fits",
        5: "proc-fimg-fvc1s-0015.fits",
        6: "proc-fimg-fvc1s-0016.fits",
        7: "proc-fimg-fvc1s-0017.fits",
        8: "proc-fimg-fvc1s-0018.fits",
        9: "proc-fimg-fvc1s-0019.fits",
        10: "proc-fimg-fvc1s-0020.fits",
    }

    fa = calibration.fiberAssignments.reset_index().set_index("holeID")
    wok = calibration.wokCoords.reset_index().set_index("holeID")

    fa = fa.join(wok, lsuffix="_1", rsuffix="_2")
    fa = fa.loc[~fa.LongLinkMTP.isna()]

    for mtp, file_ in MTP_FILES.items():
        path = DIR + "/" + file_

        data = fits.getdata(path, 1).astype("f8")

        imbias = numpy.median(data, axis=0)
        # imbias = numpy.outer(numpy.ones(data.shape[0]), imbias)
        im = data - imbias

        im = numpy.rot90(im)

        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(
            im,
            origin="lower",
            vmin=im.mean() - 3 * im.std(),
            vmax=im.mean() + 3 * im.std(),
        )

        axes[0].set_xlabel("xwok")
        axes[0].set_ylabel("xwok")
        axes[0].set_title(f"MTP {mtp}")

        mtp_data = fa.loc[fa.LongLinkMTP == mtp]
        sextants = list(mtp_data.Sextant.unique())

        axes[1].scatter(mtp_data.xWok, mtp_data.yWok, marker=".", s=10, color="b")
        axes[1].set_xlim(fa.xWok.min() - 50, fa.xWok.max() + 50)
        axes[1].set_ylim(fa.yWok.min() - 50, fa.yWok.max() + 50)
        axes[1].set_xlabel("xwok")
        axes[1].set_ylabel("xwok")
        axes[1].set_title(f"Sextants: {sextants}")
        axes[1].set_aspect(1, adjustable="box")

        fig.savefig(f"mtp_{mtp}.png", dpi=150)
        plt.close(fig)
