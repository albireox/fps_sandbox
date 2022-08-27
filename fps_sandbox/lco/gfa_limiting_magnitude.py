#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-26
# @Filename: gfa_limiting_magnitude.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import warnings
from multiprocessing import Pool

from typing import cast

import numpy
import pandas
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from tqdm import tqdm

from coordio.guide import gfa_to_radec
from sdssdb.connection import PostgresqlDatabase


warnings.filterwarnings("ignore", module="astropy.wcs.wcs")
warnings.filterwarnings("ignore", category=FITSFixedWarning)


def calculate_limiting_magnitude(file_: pathlib.Path | str):

    file_ = pathlib.Path(file_)

    if not file_.name.startswith("proc-"):
        raise NameError("Please use a proc- image.")

    header = fits.getheader(str(file_), 1)
    regions = fits.getdata(str(file_), 2)

    if len(regions) == 0:
        raise ValueError(f"No regions found for {file_!s}")

    regions = pandas.DataFrame(regions)

    if header["RAFIELD"] < 0:
        raise ValueError("RAFIELD not set.")

    if header["SOLVED"] is False:
        raise ValueError(f"Image {file_!s} was not solved.")

    offset_ra = (
        (header["AOFFRA"] + header["OFFRA"])
        / numpy.cos(numpy.deg2rad(header["DECFIELD"]))
        / 3600.0
    )
    offset_dec = (header["AOFFDEC"] + header["OFFDEC"]) / 3600.0
    offset_pa = (header["AOFFPA"] + header["OFFPA"]) / 3600.0

    ra, dec = gfa_to_radec(
        header["OBSERVAT"],
        1024,
        1024,
        int(header["CAMNAME"][3]),
        header["RAFIELD"] - offset_ra,
        header["DECFIELD"] - offset_dec,
        header["FIELDPA"] - offset_pa,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        with PostgresqlDatabase("sdss5db", port=5433) as db:
            gaia_sources = pandas.read_sql(
                "SELECT ra,dec,phot_g_mean_mag FROM gaia_dr2_source "
                f"WHERE q3c_radial_query(ra, dec, {ra}, {dec}, 0.25) AND "
                "phot_g_mean_mag < 21",
                db,
            )

    wcs = WCS(header)

    regions_astro = wcs.all_pix2world(regions.loc[:, ["x", "y"]].values, 0)
    regions.loc[:, ["ra", "dec"]] = regions_astro

    sky_gaia = SkyCoord(
        ra=gaia_sources.ra,
        dec=gaia_sources.dec,
        frame="icrs",
        unit="deg",
    )

    sky_regions = SkyCoord(
        ra=regions_astro[:, 0],
        dec=regions_astro[:, 1],
        frame="icrs",
        unit="deg",
    )

    idx, sep, _ = match_coordinates_sky(sky_regions, sky_gaia)

    regions = regions.loc[sep.deg < 1 / 3600.0, ["ra", "dec", "flux"]]

    matched_gaia = cast(pandas.DataFrame, gaia_sources.iloc[idx, :])
    matched_gaia = matched_gaia.rename(columns={"ra": "ra_gaia", "dec": "dec_gaia"})
    matched_gaia = matched_gaia.loc[sep.deg < 1 / 3600.0]
    matched_gaia.index = regions.index

    return pandas.concat([regions, matched_gaia], axis=1).reindex()


def _call_func(file_):
    try:
        return calculate_limiting_magnitude(file_)
    except Exception:
        return None


def mjd_run(mjd: int, path: str | None = None):

    procs = list((pathlib.Path(path or "/data/gcam") / f"{mjd}").glob("proc-*.fits"))

    with Pool(8) as pool:
        dataset = list(tqdm(pool.imap(_call_func, procs), total=len(procs)))

    return pandas.concat([d for d in dataset if d is not None], axis=0)
