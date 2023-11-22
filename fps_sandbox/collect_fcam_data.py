#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-11-21
# @Filename: collect_fcam_data.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib
import re

import pandas
from astropy.io import fits
from rich.progress import track


BASE_PATH = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/fcam"
OBSERVATORY = "lco"

OUTPATH = pathlib.Path(__file__).parent / f"./results/fcam_data_{OBSERVATORY}.parquet"


def get_fcam_data(path: pathlib.Path):
    """Returns fcam data."""

    match = re.match(r"proc-fimg-fvc[1-2][ns]-([0-9]+)\.fits", path.name)
    if match is None:
        return None

    try:
        seqno = int(match.group(1))
        mjd = fits.getval(path, "SJD", extname="RAW")
        date_obs = fits.getval(path, "DATE-OBS", extname="RAW")
    except Exception:
        return None

    return (mjd, seqno, date_obs, OBSERVATORY.upper())


def create_empty_file():
    """Creates and empty data frame."""

    df = pandas.DataFrame(
        {
            "mjd": pandas.Series([], dtype="Int32"),
            "seqno": pandas.Series([], dtype="Int32"),
            "date_obs": pandas.Series([], dtype="S30"),
            "observatory": pandas.Series([], dtype="S3"),
        }
    )

    df.to_parquet(OUTPATH)


def process_fcam_files():
    """Collects data from all fcam files."""

    create_empty_file()

    obspath = pathlib.Path(BASE_PATH) / OBSERVATORY
    mjds = sorted(obspath.glob("[5-6]*"))

    for mjd in track(mjds):
        proc_files = mjd.glob("proc-fimg*.fits")
        with multiprocessing.Pool(16) as pool:
            data = pool.map(get_fcam_data, proc_files)

        data = [tt for tt in data if tt is not None]

        df_orig = pandas.read_parquet(OUTPATH)

        df = pandas.DataFrame(data, columns=df_orig.columns)
        df = df.astype(df_orig.dtypes.to_dict())

        df_new = pandas.concat((df_orig, df))
        df_new.sort_values(["mjd", "seqno"], inplace=True)
        df_new.reset_index(drop=True, inplace=True)
        df_new.to_parquet(OUTPATH)


if __name__ == "__main__":
    process_fcam_files()
