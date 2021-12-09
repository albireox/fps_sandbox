#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-08
# @Filename: check_fvc_rotation.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import sys

import numpy
from astropy.io import fits

from coordio import calibration
from jaeger import log, logging
from jaeger.fvc import FVC
from jaeger.target.tools import positioner_to_wok


log.sh.setLevel(logging.INFO)


async def check_fvc_rotation(files: list[str]):
    """Checks that the FVC processing works for images with different rotator angles."""

    fvc = FVC("APO")

    pT = calibration.positionerTable.reset_index()

    pT = pT.loc[:, ["positionerID", "holeID"]]
    pT.rename(
        columns={"positionerID": "positioner_id", "holeID": "hole_id"},
        inplace=True,
    )

    pT.loc[:, "fibre_type"] = "Metrology"
    pT.loc[:, ["alpha", "beta"]] = (10, 170)
    pT.loc[:, ["xwok", "ywok"]] = numpy.nan
    pT.loc[:, ["assigned", "offline"]] = (1, 0)

    for idx, row in pT.iterrows():
        wok, _ = positioner_to_wok(row.hole_id, "APO", "Metrology", 10, 170)
        pT.loc[idx, ["xwok", "ywok"]] = (wok[0], wok[1])

    for file in files:
        header = fits.getheader(file, 1)
        print(f"Processing {file}. Rotator angle {header['ROTPOS']}")
        fvc.process_fvc_image(file, fibre_data=pT.copy(), plot=True)
        print()


if __name__ == "__main__":
    asyncio.run(check_fvc_rotation(sorted(sys.argv[1:])))
