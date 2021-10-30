#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-07-29
# @Filename: move_sequence.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio
import os
import sys

import numpy

from fps_sandbox.osu.check_layout import prepare_layout_data
from jaeger import FPS


async def move_sequence(data):

    fps = await FPS.create()

    # alpha = numpy.array([p.alpha for p in fps.values()])
    # beta = numpy.array([p.beta for p in fps.values()])

    # numpy.testing.assert_allclose(
    #     alpha,
    #     0,
    #     atol=1,
    #     err_msg="Alpha should be 0 for all positioners",
    # )
    # numpy.testing.assert_allclose(
    #     beta,
    #     180,
    #     atol=1,
    #     err_msg="Beta should be 180 for all positioners",
    # )

    data = data.sort_values(['Row', 'Column'], ascending=[True, True])

    for pid, pid_data in data.iterrows():
        if pid_data.Row < 11:
            continue
        print(f"Rotating P{pid} (R{pid_data.Row}, C{pid_data.Column})")
        await fps[pid].goto(90, 180)
        await asyncio.sleep(0.5)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = os.path.join(
            os.path.dirname(__file__),
            "../data/SloanFPS_Assignments_2021Oct22.csv",
        )

    asyncio.run(move_sequence(prepare_layout_data(data_file)))
