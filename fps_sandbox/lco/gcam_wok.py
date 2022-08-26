#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-25
# @Filename: gcam_wok.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import warnings

from typing import Any

import numpy
import pandas
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

from coordio.guide import radec2wokxy


warnings.filterwarnings("ignore", module="astropy.wcs.wcs")
warnings.filterwarnings("ignore", category=FITSFixedWarning)


MJD = 59814

RESULTS = pathlib.Path(__file__).parent / "../results/lco"


def gcam_wok():
    """Collects the astrometic-based wok positions for each proc- file."""

    files = pathlib.Path(f"/data/lco/gcam/{MJD}").glob("proc-*.fits")

    results = []

    for file in files:
        header = fits.getheader(str(file), 1)
        if header["SOLVED"] is False:
            continue

        field_ra = header["RAFIELD"]
        field_dec = header["DECFIELD"]
        field_pa = header["FIELDPA"]

        if field_ra < 0:
            continue

        offset_ra = header["AOFFRA"] / 3600.0
        offset_dec = header["AOFFDEC"] / 3600.0
        offset_pa = header["AOFFPA"] / 3600.0

        obstime = Time(header["DATE-OBS"]).jd

        wcs = WCS(header)

        coords: Any = wcs.pixel_to_world(1024, 1024)
        ra = coords.ra.value
        dec = coords.dec.value

        xwok_astro, ywok_astro, *_ = radec2wokxy(
            [ra],
            [dec],
            None,
            "GFA",
            field_ra - offset_ra / numpy.cos(numpy.deg2rad(field_dec)),
            field_dec - offset_dec,
            field_pa - offset_pa,
            header["OBSERVAT"],
            obstime,
        )

        results.append(
            (
                int(header["CAMNAME"][3]),
                field_ra,
                field_dec,
                field_pa,
                offset_ra,
                offset_dec,
                offset_pa,
                obstime,
                ra,
                dec,
                xwok_astro[0],
                ywok_astro[0],
            )
        )

    data = pandas.DataFrame(
        results,
        columns=[
            "camera",
            "field_ra",
            "field_dec",
            "field_pa",
            "offset_ra",
            "offset_dec",
            "offset_pa",
            "obstime",
            "ra",
            "dec",
            "xwok_astro",
            "ywok_astro",
        ],
    )

    RESULTS.mkdir(parents=True, exist_ok=True)

    data.to_csv(RESULTS / f"astro_wok_{MJD}.csv", index=False)

    return data


def median_gcam_wok(data: pandas.DataFrame):
    """Returns the sigclipped median xywok position for each camera."""

    sigclip = SigmaClip(3)

    for camera in range(1, 7):
        cam = data.loc[data.camera == camera]
        xw = numpy.median(sigclip(cam.xwok_astro, masked=False))
        yw = numpy.median(sigclip(cam.ywok_astro, masked=False))
        print(camera, xw, yw)
