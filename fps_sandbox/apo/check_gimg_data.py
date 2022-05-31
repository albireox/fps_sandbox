#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-05-29
# @Filename: check_gimg_data.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re
import warnings
from multiprocessing import Pool

from typing import Any, cast

import matplotlib
import numpy
import pandas
import seaborn
from astropy.io.fits import getheader
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from rich.progress import Progress

from cherno.utils import gfa_to_wok, umeyama
from coordio import calibration
from coordio.defaults import PLATE_SCALE
from coordio.utils import radec2wokxy


warnings.filterwarnings("ignore", module="astropy.wcs.wcs")
warnings.filterwarnings("ignore", category=FITSFixedWarning)


matplotlib.use("TkAgg")
seaborn.set_theme()

RESULTS = pathlib.Path(__file__).parents[1] / "results"
DATA = pathlib.Path("/data/gcam")
MJD = 59658


def fit_one(
    gimg_path: pathlib.Path,
    grid: tuple[int, int] = (10, 10),
    scale_rms: bool = True,
    internal_fit: bool = True,
) -> tuple | None:
    """Fits one camera with internal rotation."""

    name = gimg_path.name
    parent = gimg_path.parent

    proc_path = parent / ("proc-" + name)
    if not proc_path.exists():
        return None

    solved = parent / "astrometry" / name.replace("fits", "solved")
    if not solved.exists():
        return None

    wcs_path = parent / "astrometry" / name.replace("fits", "wcs")
    if not wcs_path.exists():
        return None

    wcs = WCS(open(wcs_path, "r").read())

    header = getheader(str(gimg_path), 1)
    header_proc = getheader(str(proc_path), 1)

    match = re.match(r"gimg-gfa[1-6][ns]-([0-9]{4})\.fits", name)
    if not match:
        return None

    seq_no = int(match.group(1))

    field_ra = header["RAFIELD"]
    field_dec = header["DECFIELD"]
    field_pa = header["FIELDPA"]

    if "AOFFRA" in header_proc:
        offset_ra = header_proc["AOFFRA"]
        offset_dec = header_proc["AOFFDEC"]
        offset_pa = header_proc["AOFFPA"]
    else:
        offset_ra = header_proc["OFFRA"]
        offset_dec = header_proc["OFFDEC"]
        offset_pa = header_proc["OFFPA"]

    obstime = Time(header["DATE-OBS"]).jd
    camera_id = int(header["CAMNAME"][3:4])

    observatory = header["OBSERVAT"]

    gfa_data = calibration.gfaCoords.loc[(observatory, camera_id), :]

    xwok_gfa: list[float] = []
    ywok_gfa: list[float] = []
    xwok_astro: list[float] = []
    ywok_astro: list[float] = []

    xidx = numpy.arange(2048)[:: 2048 // grid[0]]
    yidx = numpy.arange(2048)[:: 2048 // grid[1]]

    coords: Any = wcs.pixel_to_world(xidx, yidx)
    ra = coords.ra.value
    dec = coords.dec.value

    for x, y in zip(xidx, yidx):
        xw, yw, _ = gfa_to_wok(x, y, camera_id)
        xwok_gfa.append(cast(float, xw))
        ywok_gfa.append(cast(float, yw))

    cos_dec = numpy.cos(numpy.deg2rad(field_dec))
    offset_ra_deg = offset_ra / cos_dec / 3600.0

    _xwok_astro, _ywok_astro, *_ = radec2wokxy(
        ra,
        dec,
        None,
        "GFA",
        field_ra - offset_ra_deg,
        field_dec - offset_dec / 3600.0,
        field_pa - offset_pa / 3600.0,
        observatory,
        obstime,
    )

    xwok_astro += _xwok_astro.tolist()
    ywok_astro += _ywok_astro.tolist()

    X = numpy.array([xwok_gfa, ywok_gfa])
    Y = numpy.array([xwok_astro, ywok_astro])

    if internal_fit:
        X[0, :] -= gfa_data.xWok
        X[1, :] -= gfa_data.yWok

        Y[0, :] -= gfa_data.xWok
        Y[1, :] -= gfa_data.yWok

    try:
        c, R, t = umeyama(X, Y)
    except ValueError:
        return None

    plate_scale = PLATE_SCALE[observatory]

    # delta_x and delta_y only align with RA/Dec if PA=0. Otherwise we need to
    # project using the PA.
    pa_rad = numpy.deg2rad(field_pa)
    delta_ra = t[0] * numpy.cos(pa_rad) + t[1] * numpy.sin(pa_rad)
    delta_dec = -t[0] * numpy.sin(pa_rad) + t[1] * numpy.cos(pa_rad)

    # Convert to arcsec and round up
    delta_ra = numpy.round(delta_ra / plate_scale * 3600.0, 3)
    delta_dec = numpy.round(delta_dec / plate_scale * 3600.0, 3)

    delta_rot = numpy.round(-numpy.rad2deg(numpy.arctan2(R[1, 0], R[0, 0])) * 3600.0, 1)
    delta_scale = numpy.round(c, 6)

    if scale_rms:
        xwok_astro /= delta_scale
        ywok_astro /= delta_scale

    delta_x = (numpy.array(xwok_gfa) - numpy.array(xwok_astro)) ** 2  # type: ignore
    delta_y = (numpy.array(ywok_gfa) - numpy.array(ywok_astro)) ** 2  # type: ignore

    xrms = numpy.sqrt(numpy.sum(delta_x) / len(delta_x))
    yrms = numpy.sqrt(numpy.sum(delta_y) / len(delta_y))
    rms = numpy.sqrt(numpy.sum(delta_x + delta_y) / len(delta_x))

    # Convert to arcsec and round up
    xrms = numpy.round(xrms / plate_scale * 3600.0, 3)
    yrms = numpy.round(yrms / plate_scale * 3600.0, 3)
    rms = numpy.round(rms / plate_scale * 3600.0, 3)

    return (
        seq_no,
        camera_id,
        header["CONFIGID"],
        header["DESIGNID"],
        header["FIELDID"],
        field_ra,
        field_dec,
        field_pa,
        delta_ra,
        delta_dec,
        delta_rot,
        delta_scale,
        xrms,
        yrms,
        rms,
        header_proc["DELTARA"],
        header_proc["DELTADEC"],
        header_proc["DELTAROT"],
        header_proc["DELTASCL"],
        header_proc["RMS"],
    )


def check_internal_gfa_fit(mjd: int):

    gcam_data = DATA / str(MJD)

    gimg_paths = list(sorted(gcam_data.glob("gimg-gfa*[0-9].fits")))

    fit_data: list[tuple] = []

    with Progress() as progress:
        task_id = progress.add_task(f"[cyan]{MJD} ...", total=len(gimg_paths))

        with Pool(processes=4) as pool:
            for data in pool.imap_unordered(fit_one, gimg_paths):
                if data is None or data is False:
                    continue
                fit_data.append(data)
                progress.advance(task_id)

    fit_data = [fd for fd in fit_data if fd]

    df = pandas.DataFrame(
        fit_data,
        columns=[
            "seq_no",
            "camera_id",
            "configid",
            "designid",
            "fieldid",
            "field_ra",
            "field_dec",
            "field_pa",
            "delta_ra",
            "delta_dec",
            "delta_rot",
            "delta_scale",
            "xrms",
            "yrms",
            "rms",
            "delta_ra_proc",
            "delta_dec_proc",
            "delta_rot_proc",
            "delta_scale_proc",
            "rms_proc",
        ],
    )

    breakpoint()


if __name__ == "__main__":
    check_internal_gfa_fit(MJD)
