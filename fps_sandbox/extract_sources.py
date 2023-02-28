#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-10-26
# @Filename: extract_sources.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import os
import pathlib
import re
import sys
import warnings
from functools import partial

import numpy
import pandas
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from tqdm import tqdm

from coordio.extraction import extract_marginal
from coordio.guide import gfa_to_radec


def _get_dark_APO(exptime: float, gfa_id: int):

    PATH = "/data/gcam/apo/calibration/gimg-gfa{gfa_id}n-{frame_no}.fits"

    if exptime == 15:
        files = [
            PATH.format(gfa_id=gfa_id, frame_no=frame_no)
            for frame_no in range(1047, 1053)
        ]
    else:
        files = [
            PATH.format(gfa_id=gfa_id, frame_no=frame_no)
            for frame_no in range(1053, 1056)
        ]

    stack = numpy.dstack(
        [fits.getdata(ff).astype("f8") for ff in files if os.path.exists(ff)]
    )
    return numpy.median(stack, -1)


def _get_dark_LCO_59878(exptime: float, gfa_id: int):

    PATH = "/data/gcam/lco/calibration/59878/gimg-gfa{gfa_id}s-{frame_no:04d}.fits"

    files = [
        PATH.format(gfa_id=gfa_id, frame_no=frame_no) for frame_no in range(15, 18)
    ]

    stack = numpy.dstack(
        [fits.getdata(ff).astype("f8") for ff in files if os.path.exists(ff)]
    )

    return numpy.median(stack, -1)


def _get_dark_LCO_59879(exptime: float, gfa_id: int):

    PATH = "/data/gcam/lco/calibration/59879/gimg-gfa{gfa_id}s-{frame_no:04d}.fits"

    if exptime == 15:
        files = [
            PATH.format(gfa_id=gfa_id, frame_no=frame_no) for frame_no in range(21, 24)
        ]
    else:
        files = [
            PATH.format(gfa_id=gfa_id, frame_no=frame_no) for frame_no in range(24, 27)
        ]

    stack = numpy.dstack(
        [fits.getdata(ff).astype("f8") for ff in files if os.path.exists(ff)]
    )

    return numpy.median(stack, -1)


def _process_one(output: pathlib.Path, file_: pathlib.Path | str, plot: bool = False):

    file_ = pathlib.Path(file_)

    data = fits.getdata(str(file_)).astype("f8")
    # header = fits.getheader(str(file_), 1)
    # gfa_id = int(re.search("gfa([1-6])", str(file_)).group(1))
    # dark = _get_dark_LCO_59879(header["EXPTIME"], gfa_id)

    out_path = str(output / file_.name)
    plot_path = out_path.replace(".fits", ".pdf") if plot else None

    try:
        dets = extract_marginal(data, 5, plot=plot_path, plot_title=file_.name)
    except Exception:
        return

    if len(dets) > 0:
        dets.index.name = "regions"
        dets.reset_index(inplace=True)
        dets.to_csv(out_path.replace(".fits", ".csv"), index=False)


def extract_sources(
    files: list[str] | list[pathlib.Path],
    output: str | pathlib.Path,
    plot: bool = True,
    chunksize: int = 20,
):
    """Extracts sources for a list of files."""

    output = pathlib.Path(output)
    output.mkdir(parents=True, exist_ok=True)

    files = list(sorted(files))

    _process = partial(_process_one, output, plot=plot)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        bar = tqdm(total=len(files))
        for ii in range(0, len(files), chunksize):
            fs = files[ii : ii + chunksize]
            with multiprocessing.Pool(chunksize) as pool:
                for _ in pool.imap_unordered(_process, fs):
                    bar.update(1)


def collate_data(path: pathlib.Path | str, mjd: int | None = None):
    """For each file in ``path`` calculates the average sigma."""

    path = pathlib.Path(path)
    files = path.glob("*.csv")

    data = []
    for file_ in files:

        if frame_no_group := re.search(r"\-([0-9]{4})\.csv", str(file_)):
            frame_no = int(frame_no_group.group(1))
        else:
            continue

        if gfa_id_group := re.search("gfa([1-6])", str(file_)):
            gfa_id = int(gfa_id_group.group(1))
        else:
            continue

        fdata = pandas.read_csv(str(file_))

        fdata = fdata.loc[
            (fdata.xstd > 0.75)
            & (fdata.xstd < 14)
            & (fdata.ystd > 0.75)
            & (fdata.ystd < 14),
            :,
        ]

        if len(fdata) == 0:
            continue

        xstd_10 = numpy.percentile(fdata.xstd, 10)
        xstd_90 = numpy.percentile(fdata.xstd, 90)
        iq = fdata.loc[(fdata.xstd > xstd_10) & (fdata.xstd < xstd_90)]

        if len(fdata) > 5 and len(iq) > 5:
            xstd_median = numpy.median(iq.xstd)
            ystd_median = numpy.median(iq.ystd)
            std_median = iq.loc[:, ["xstd", "ystd"]].stack().median()
            std_std = iq.loc[:, ["xstd", "ystd"]].stack().std()
            nn = len(iq)
        else:
            xstd_median = numpy.median(fdata.xstd)
            ystd_median = numpy.median(fdata.ystd)
            std_median = fdata.loc[:, ["xstd", "ystd"]].stack().median()
            std_std = fdata.loc[:, ["xstd", "ystd"]].stack().std()
            nn = len(fdata)

        data.append(
            (
                mjd,
                frame_no,
                gfa_id,
                xstd_median,
                ystd_median,
                std_median,
                std_std,
                nn,
            )
        )

    df = pandas.DataFrame(
        data,
        columns=[
            "mjd",
            "frame_no",
            "gfa_id",
            "xsigma",
            "ysigma",
            "sigma",
            "sigma_std",
            "n",
        ],
    )

    if mjd is None:
        df = df.drop(columns=["mjd"])

    df = df.sort_values(["frame_no", "gfa_id"])

    return df


def calculate_offset(data: pandas.DataFrame, gfa: int = 1):
    """Calculate camera focus offsets wrt a given GFA. Converts to FWHM from sigma."""

    data = data.copy()
    data = data.reset_index().set_index(["frame_no", "gfa_id"])

    sigma_diff = data.sigma - data.loc[pandas.IndexSlice[:, gfa], "sigma"].droplevel(1)
    sigma_diff = sigma_diff.dropna()

    fwhm_diff = sigma_diff * 2.355
    fwhm_diff.name = "FWHM"

    data["fwhm"] = data.sigma * 2.355
    data["fwhm_diff"] = fwhm_diff

    return data


def add_gaia_mags(
    csv_path: pathlib.Path | str,
    data_path: pathlib.Path | str,
    gaia_table: str = "catalogdb.gaia_dr2_source_g19",
    limit_mag: float = 19.0,
    max_sep: float = 1,
    max_rms: float = 1,
    database_string: str = "postgresql://sdss@localhost:5433/sdss5db",
):
    """Cross-match extractions with Gaia sources."""

    files = list(sorted(pathlib.Path(csv_path).glob("*.csv")))

    for file in tqdm(files):

        if "gaia" in file.name:
            continue

        match = re.search(r"gfa([1-6][ns])\-([0-9]{4})", str(file))
        if not match:
            continue
        cam_id, frame_no = match.groups()

        proc_file = pathlib.Path(data_path) / f"proc-gimg-gfa{cam_id}-{frame_no}.fits"
        if not proc_file.exists():
            continue

        header = fits.getheader(proc_file, 1)

        RMS = header.get("RMS", 999)
        solved = header.get("SOLVED", False)

        if RMS > max_rms or solved is False:
            continue

        field_ra = header.get("RAFIELD", None)
        field_dec = header.get("DECFIELD", None)
        field_pa = header.get("FIELDPA", None)

        offra = header.get("AOFFRA")
        offdec = header.get("AOFFDEC")
        offpa = header.get("AOFFPA")

        observatory = header["OBSERVAT"]
        obstime = Time(header["DATE-OBS"], format="iso", scale="tai")

        if field_ra is None or field_dec is None or field_pa is None:
            continue

        wcs = WCS(header)

        sources = pandas.read_csv(str(file))
        source_ra, source_dec = wcs.all_pix2world(sources.x1, sources.y1, 0)

        ccd_centre = gfa_to_radec(
            observatory,
            1024,
            1024,
            int(cam_id[-2]),
            field_ra,
            field_dec,
            field_pa,
            offra,
            offdec,
            offpa,
            obstime.jd,
            icrs=True,
        )

        gaia_search_radius = 0.05 if observatory == "LCO" else 0.09

        gaia_stars = pandas.read_sql(
            f"SELECT * FROM {gaia_table} "
            "WHERE q3c_radial_query(ra, dec, "
            f"{ccd_centre[0]}, {ccd_centre[1]}, {gaia_search_radius}) AND "
            f"phot_g_mean_mag < {limit_mag}",
            database_string,
        )

        idx, sep2d, _ = match_coordinates_sky(
            SkyCoord(ra=source_ra, dec=source_dec, frame="icrs", unit="deg"),
            SkyCoord(ra=gaia_stars.ra, dec=gaia_stars.dec, frame="icrs", unit="deg"),
        )

        gaia_cols = ["source_id", "ra", "dec", "phot_g_mean_mag"]

        stars = gaia_stars.loc[idx, gaia_cols]
        sources = pandas.concat([sources, stars.reset_index(drop=True)], axis=1)

        sources.loc[sep2d.arcsec > max_sep, gaia_cols] = numpy.nan
        sources = sources.dropna(subset=["source_id"])
        sources["source_id"] = sources["source_id"].values.astype("i8")

        new_file = str(file.resolve()).replace(".csv", "_gaia.csv")
        sources.to_csv(new_file, index=False)


if __name__ == "__main__":

    files = list(pathlib.Path(sys.argv[1]).glob("proc-gimg-*"))
    extract_sources(files, output="./59876_t5", plot=False)

    csv_path = "/home/gallegoj/software/fps_sandbox/results/lco/gcam_reprocess/59876_t5"
    data_path = "/data/gcam/lco/59876"
    add_gaia_mags(csv_path, data_path)

    # csv_path = "/home/gallegoj/software/fps_sandbox/results/apo/gcam_reprocess/59873_t5"
    # data_path = "/data/gcam/apo/59873"
    # add_gaia_mags(csv_path, data_path)
