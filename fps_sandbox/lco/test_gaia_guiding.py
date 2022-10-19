#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-10-03
# @Filename: test_gaia_guiding.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import pathlib
import warnings

import numpy
import pandas
import seaborn
from astropy.io import fits
from astropy.time import Time
from matplotlib import pyplot as plt

from coordio.guide import cross_match, gfa_to_radec, radec_to_gfa
from sdssdb.peewee.sdss5db import database


seaborn.set_theme(style="white", palette="deep")

warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")


def test_gaia_guiding(file: str | pathlib.Path):

    database.connect("sdss5db", user=None, hostname=None)
    assert database.connected, "Database is not connected"

    file = pathlib.Path(file)

    data: numpy.ndarray = fits.getdata(file)  # type:ignore
    regions: numpy.recarray = fits.getdata(file, 2)  # type:ignore
    header = fits.getheader(file, 1)  # type:ignore

    ra = header["RAFIELD"]
    dec = header["DECFIELD"]
    pa = header["FIELDPA"]

    offra = header["AOFFRA"]
    offdec = header["AOFFDEC"]
    offpa = header["AOFFPA"]

    cam_id = int(header["CAMNAME"][-2])
    time = Time(header["DATE-OBS"], format="iso", scale="tai")

    ccd_centre = gfa_to_radec(
        "LCO",
        1024,
        1024,
        cam_id,
        ra,
        dec,
        pa,
        offra,
        offdec,
        offpa,
        time.jd,
        icrs=True,
    )

    gaia_stars = pandas.read_sql(
        "SELECT * FROM catalogdb.gaia_dr2_source_g19 "
        "WHERE q3c_radial_query(ra, dec, "
        f"{ccd_centre[0]}, {ccd_centre[1]}, 0.08333333) AND phot_g_mean_mag < 18",
        database,
    )

    gaia_x, gaia_y = radec_to_gfa(
        "LCO",
        numpy.array(gaia_stars["ra"].values),
        numpy.array(gaia_stars["dec"].values),
        cam_id,
        ra,
        dec,
        pa,
        offra,
        offdec,
        offpa,
        time.jd,
    )

    gaia = numpy.vstack([gaia_x, gaia_y, gaia_stars.ra, gaia_stars.dec]).T
    gaia = gaia[
        (gaia[:, 0] >= 0)
        & (gaia[:, 0] < 2048)
        & (gaia[:, 1] >= 0)
        & (gaia[:, 1] < 2048)
    ]

    detections = numpy.vstack([regions["x"], regions["y"]]).T

    wcs = cross_match(
        detections,
        gaia[:, 0:2],
        gaia[:, 2:],
        2048,
        2048,
        blur=5,
        upsample_factor=100,
    )

    fig, ax = plt.subplots()

    ax.imshow(
        data,
        vmin=data.mean() - data.std(),
        vmax=data.mean() + data.std(),
        origin="lower",
    )

    # ax.scatter(detections[:, 0], detections[:, 1], s=3, c="k")
    # ax.scatter(detections_fit[:, 0], detections_fit[:, 1], s=3, c="m")
    # ax.scatter(gaia[:, 0], gaia[:, 1], s=3, c="b")

    gaia_x_wcs, gaia_y_wcs = wcs.all_world2pix(gaia[:, 2], gaia[:, 3], 0)
    ax.scatter(gaia_x_wcs, gaia_y_wcs, s=3, c="b")
    ax.scatter(gaia[:, 0], gaia[:, 1], s=3, c="k", marker="x")

    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 2048)


if __name__ == "__main__":
    asyncio.run(test_cherno(59860))
