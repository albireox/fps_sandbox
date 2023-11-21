#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-05-16
# @Filename: get_gfa_dither_data.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import pandas
from astropy.io import fits


MJD_FIELD = [
    (60027, 20913),
    (60007, 20927),
    (60017, 20919),
    (60007, 20925),
    (60029, 20953),
    (60029, 20961),
    (60030, 20921),
    (60032, 20915),
    (60032, 20935),
    (60033, 20923),
    (60033, 20929),
    (60040, 20943),
    (60040, 20957),
    (60042, 20933),
    (60067, 20941),
    (60042, 20949),
    (60042, 20965),
    (60045, 20919),
    (60064, 20953),
    (60067, 20981),
    (60069, 20943),
    (60069, 20949),
    (60069, 20961),
    (60071, 20939),
    (60071, 20967),
    (60055, 20951),
]

GCAM_PATH = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/gcam/apo/"


def get_gfa_dither_data():
    """Collects GFA data for dither sequences."""

    data = []

    for mjd, field in MJD_FIELD:
        print(f"Doing {mjd}-{field}.")

        path = pathlib.Path(GCAM_PATH) / str(mjd)

        all_gimg = sorted(path.glob("gimg-gfa[1-6]n-[0-9][0-9][0-9][0-9].fits"))
        max_frame = int(all_gimg[-1].name.split("-")[-1].split(".")[0])

        n_solved = 0
        n_frames = 0
        n_astro = 0
        n_gaia = 0
        for frame in range(1, max_frame + 1):
            frame_gimgs = list(path.glob(f"proc-gimg-gfa[1-6]n-{frame:04d}.fits"))
            if len(frame_gimgs) == 0:
                continue

            header0 = fits.getheader(str(frame_gimgs[0]), 1)
            if header0["FIELDID"] != field:
                continue

            any_solved = False
            for gimg in frame_gimgs:
                header = fits.getheader(str(gimg), 1)
                if header["SOLVED"]:
                    n_solved += 1
                    any_solved = True
                if header["SOLVMODE"] == "astrometry.net":
                    n_astro += 1
                elif header["SOLVMODE"] == "gaia":
                    n_gaia += 1

            if any_solved:
                n_frames += 1

        data.append(
            (
                mjd,
                field,
                round(n_solved / n_frames, 2),
                round(n_astro / n_solved, 3),
                round(n_gaia / n_solved, 3),
            )
        )

    df = pandas.DataFrame(
        data,
        columns=["mjd", "field", "solved_cameras", "frac_astro", "frac_gaia"],
    )
    df.to_csv("dither_gfa_data.csv", index=False)


if __name__ == "__main__":
    get_gfa_dither_data()
