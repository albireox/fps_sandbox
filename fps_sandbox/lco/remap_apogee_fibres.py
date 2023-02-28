#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-01-22
# @Filename: remap_apogee_fibres.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

# Fix confSummary files from MJD 59853 on. APOGEE fibres 151-160 and
# 181-190 were swapped sequentially.

from __future__ import annotations

import os
import pathlib

from pydl.pydlutils.sdss import yanny
from rich.progress import track


# These are the correct hole to APOGEE fibre ID.
NEW_FIBERIDS = {
    b"R-9C6": 181,
    b"R-11C11": 182,
    b"R-11C12": 183,
    b"R-11C13": 184,
    b"R-9C13": 185,
    b"R-7C9": 186,
    b"R-9C7": 187,
    b"R-11C5": 188,
    b"R-9C8": 189,
    b"R-7C10": 190,
    b"R-3C11": 151,
    b"R-11C3": 152,
    b"R-12C2": 153,
    b"R-13C1": 154,
    b"R-13C2": 155,
    b"R-13C3": 156,
    b"R-13C9": 157,
    b"R-13C10": 158,
    b"R-13C11": 159,
    b"R-13C12": 160,
}


def remap_apogee_fibres():
    SDSSCORE = os.environ["SDSSCORE_DIR"]

    files = (pathlib.Path(SDSSCORE) / "lco/summary_files").glob("**/*.par")
    for file in track(list(files)):
        yn = yanny(str(file))

        if int(yn["MJD"]) < 59853:
            continue

        for hole, fid in NEW_FIBERIDS.items():
            fmap = yn["FIBERMAP"]
            row = (fmap["holeId"] == hole) & (fmap["fiberType"] == b"APOGEE")
            fmap["fiberId"][row] = fid

        file.unlink()
        yn.write(str(file))


if __name__ == "__main__":
    remap_apogee_fibres()
