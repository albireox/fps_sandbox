#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-10-03
# @Filename: fix_boss_headers.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
from glob import glob
from shutil import copy

from astropy.io import fits


def fix_boss_headers():
    MJD = 59855

    path = f"/data/spectro/{MJD}"
    outpath = f"/data/spectro/{MJD}_fixed"

    keywords = [
        "OBJSYS",
        "RA",
        "DEC",
        "RADEG",
        "DECDEG",
        "AZ",
        "ALT",
        "AIRMASS",
        "HA",
        "ROTPOS",
        "IPA",
        "FOCUS",
        "M2PISTON",
        "M2XTILT",
        "M2YTILT",
        "M2XTRAN",
        "M2YTRAN",
        "M2ZROT",
        "T_OUT",
        "T_IN",
        "T_PRIM",
        "T_CELL",
        "T_FLOOR",
        "T_TRUSS",
    ]

    for file in glob(path + "/*.fit.gz"):
        copy_file = outpath + f"/{os.path.basename(file)}"
        copy(file, copy_file)

        print(copy_file)
        hdus = fits.open(str(copy_file), mode="update")
        header = hdus[0].header

        for keyword in keywords:
            header[keyword] = "NaN"

        hdus.close()

    return


if __name__ == "__main__":
    fix_boss_headers()
