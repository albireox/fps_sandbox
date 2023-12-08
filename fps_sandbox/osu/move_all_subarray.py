#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-08-31
# @Filename: move_all_subarray.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio
import os

import numpy

from jaeger import FPS, log

from fps_sandbox.osu.check_layout import prepare_layout_data


log.sh.setLevel(20)


async def main():
    data_file = os.path.join(
        os.path.dirname(__file__),
        "../data/SloanFPS_HexArray_2021July23.csv",
    )
    data = prepare_layout_data(data_file)

    bad = [
        1255,
        768,
        794,
        267,
        732,
        500,
        537,
        717,
        1367,
        398,
        775,
        738,
        1003,
        981,
        545,
        688,
        474,
        769,
        652,
        703,
        878,
        313,
        1146,
        966,
        646,
        812,
        555,
        615,
        772,
        586,
    ]

    min_beta = 165
    max_beta = 195

    sub1 = data.loc[(data.Row + data.Column) % 2 != 0, :]
    sub2 = data.loc[(data.Row + data.Column) % 2 == 0, :]

    fps = await FPS.create()

    n_moves = 1
    while True:
        connected = list(fps.keys())

        for sub in [1, 2]:
            if sub == 1:
                positioners = sub1.Device.tolist()
            else:
                positioners = sub2.Device.tolist()

            print(f"Trajectory {n_moves}; subarray {sub}.")

            for pbad in bad:
                if pbad in positioners:
                    positioners.remove(pbad)

            positioners = [p for p in positioners if p in connected]

            await fps.stop_trajectory()

            n_pos = len(positioners)
            alpha = numpy.random.random(n_pos) * 360
            beta = numpy.random.random(n_pos) * (max_beta - min_beta) + min_beta

            await fps.goto(
                positioners,
                alpha,
                beta,
                speed=(2000, 2000),
                force=True,
                use_sync_line=False,
            )

            await asyncio.sleep(1)

            await fps.goto(
                positioners,
                0,
                180,
                speed=(2000, 2000),
                force=True,
                use_sync_line=False,
            )

            await asyncio.sleep(1)

        n_moves += 1


asyncio.run(main())
