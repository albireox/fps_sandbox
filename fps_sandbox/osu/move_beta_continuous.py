#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-07-29
# @Filename: move_beta_continuous.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from jaeger import FPS
import asyncio


async def main():

    fps = await FPS.create()

    alpha = 360
    beta = 180

    while True:
        if beta == 180:
            alpha = 300
            beta = 160
        else:
            alpha = 360
            beta = 180

        await asyncio.gather(*[p.goto(alpha, beta) for p in fps.values()])

        await asyncio.sleep(1)


asyncio.run(main())
