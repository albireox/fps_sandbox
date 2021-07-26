#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-07-25
# @Filename: check_layout.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

# Reads a layout CSV file and rotates all the positioners in alpha 60 degrees
# in order of decreasing row number and increasing column.
# POSITIONERS SHOULD BE FOLDED BEFORE RUNNING THIS SCRIPT.

import asyncio
import os
import sys

import numpy
import numpy.testing
import pandas

from jaeger import FPS


async def check_layout(data_file):

    data = pandas.read_csv(data_file)

    # Select positioners only.
    data = data.loc[data.Device.str.startswith("P")]

    data.Device = pandas.to_numeric(data.Device.str.slice(1))
    data.Row = pandas.to_numeric(data.Row.str.slice(1))
    data.Column = pandas.to_numeric(data.Column.str.slice(1))

    data = data.sort_values(["Row", "Column"], ascending=[False, True])
    data.set_index(data.Device, inplace=True)

    print(f"{len(data)} positioners identified in layout file.")

    fps = await FPS.create()

    n_positioners = len(fps)
    if n_positioners != len(data):
        raise RuntimeError(
            f"{n_positioners} positioners connected to FPS. "
            "Mismatch with number of expected positioners."
        )

    alpha = numpy.array([p.alpha for p in fps.values()])
    beta = numpy.array([p.beta for p in fps.values()])

    numpy.testing.assert_allclose(
        alpha,
        0,
        atol=1,
        err_msg="Alpha should be 0 for all positioners",
    )
    numpy.testing.assert_allclose(
        beta,
        180,
        atol=1,
        err_msg="Beta should be 180 for all positioners",
    )

    for pid, pid_data in data.iterrows():
        print(f"Rotating P{pid} (R{pid_data.Row}, C{pid_data.Column})")
        await fps[pid].goto(60, 180)
        await asyncio.sleep(0.5)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = os.path.join(
            os.path.dirname(__file__),
            "data/SloanFPS_HexArray_2021July23.csv",
        )

    asyncio.run(check_layout(data_file))
