#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-06
# @Filename: test_sextants.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio

from jaeger.commands.goto import goto
from jaeger.fps import FPS

from coordio import calibration


SEXTANT = 1
ALPHA = 10
BETA = 180


async def test_sextants():
    fps = await FPS.create()

    fa = calibration.fiberAssignments.loc["LCO"]
    fa = fa.loc[fa.Device == "Positioner"]

    pids = list(fa.loc[fa.Sextant == SEXTANT].positionerID)

    await goto(fps, {pid: (ALPHA, BETA) for pid in pids})


if __name__ == "__main__":
    asyncio.run(test_sextants())
