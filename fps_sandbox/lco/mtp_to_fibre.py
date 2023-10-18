#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-25
# @Filename: mtp_to_fibre.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import itertools
import pathlib
import sys

import pandas

from coordio import calibration


def mtp_to_fibre_lco():
    """Returns a table of MTP and MTP block to spectrograph fibre ID, for LCO."""

    mtp_block_ids = list(itertools.product("ABCDE", "123456"))

    fiber_id = 1

    assignments = []
    for mtp in range(1, 11):
        for ll, nn in mtp_block_ids:
            assignments.append((mtp, ll + nn, fiber_id))
            fiber_id += 1

    data = pandas.DataFrame(
        assignments,
        columns=["LongLinkMTP", "MTPFiber", "APOGEEFiber"],
    )

    return data


def update_fiberAssignments(file_: pathlib.Path | str | None = None):
    if file_ is None:
        fa = calibration.fiberAssignments.copy().reset_index()
    else:
        fa = pandas.read_csv(str(file_))

    mtp = mtp_to_fibre_lco()

    # Swap 26 and 27.
    idx_26 = int(mtp.index[mtp.APOGEEFiber == 26][0])
    idx_27 = int(mtp.index[mtp.APOGEEFiber == 27][0])
    mtp.loc[idx_26, "APOGEEFiber"] = 27
    mtp.loc[idx_27, "APOGEEFiber"] = 26

    for _, (mtp_id, mtp_fibre, apogee_fibre) in mtp.iterrows():
        fa.loc[
            (fa.LongLinkMTP == mtp_id) & (fa.MTPFiber == mtp_fibre), "APOGEEFiber"
        ] = apogee_fibre

    fa.to_csv(str(file_) + ".updated", index=False)

    return fa


if __name__ == "__main__":
    update_fiberAssignments(sys.argv[1])
