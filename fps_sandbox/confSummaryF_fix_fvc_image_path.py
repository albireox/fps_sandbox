#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-11-21
# @Filename: confSummaryF_fix_fvc_image_path.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime
import multiprocessing
import pathlib
from functools import partial

import pandas
from rich.progress import track

from sdsstools import yanny


OBSERVATORY = "apo"
SDSSCORE_DIR = f"/home/gallegoj/software/sdsscore_test/{OBSERVATORY}/summary_files"
DATA_DIR = pathlib.Path(__file__).parents[1] / "data"


def update_confSummaryF(data: pandas.DataFrame, path: pathlib.Path):
    """Fixes the ``fvc_image_path`` keyword."""

    csF = yanny(str(path))

    if csF["fvc_image_path"] != "":
        return

    obstime = datetime.datetime.strptime(csF["obstime"], "%c")
    fimg = data.loc[data.date_obs < obstime].tail(1)

    if len(obstime - fimg.date_obs) == 0:
        return

    delta_seconds = (obstime - fimg.date_obs).iloc[0].total_seconds()
    if delta_seconds < 0 or delta_seconds > 60:
        return

    mjd = fimg.mjd.iloc[0]
    seq = fimg.seqno.iloc[0]
    cam = "fvc1n" if OBSERVATORY == "apo" else "fvc1s"

    csF["fvc_image_path"] = f"/data/fcam/{mjd}/proc-fimg-{cam}-{seq:04d}.fits"

    path.unlink()
    csF.write(str(path))


def process_confSummaryFs():
    """Updates all the confSummaryFs"""

    base_path = pathlib.Path(SDSSCORE_DIR)
    files = sorted(base_path.glob("**/**/confSummaryF*.par"))

    data = pandas.read_parquet(DATA_DIR / f"fcam_data_{OBSERVATORY}.parquet")
    data["date_obs"] = pandas.to_datetime(data["date_obs"])

    update_confSummaryF_partial = partial(update_confSummaryF, data)

    with multiprocessing.Pool(16) as pool:
        for _ in track(
            pool.imap_unordered(
                update_confSummaryF_partial,
                files,
            ),
            total=len(files),
        ):
            pass


if __name__ == "__main__":
    process_confSummaryFs()
