#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-08-02
# @Filename: move_all_safe_random.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio

import numpy

from jaeger import FPS, log


log.sh.setLevel(20)


async def main():

    fps = await FPS.create()

    min_beta = 165
    max_beta = 195

    ii = 1
    while True:

        positioners = list(fps.keys())

        bad = []
         
        for pbad in bad:
            if pbad in positioners:
                positioners.remove(pbad)

        await fps.stop_trajectory()

        alpha = numpy.random.random(len(positioners)) * 360
        beta = numpy.random.random(len(positioners)) * (max_beta - min_beta) + min_beta

        print(f"Trajectory {ii}")

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

        ii += 1


asyncio.run(main())
