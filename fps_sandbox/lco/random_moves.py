#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-08-06
# @Filename: random_moves.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio

import numpy
from jaeger import FPS, config
from jaeger.exceptions import JaegerError, TrajectoryError
from jaeger.target.tools import create_random_configuration


N_MOVES = 20
SAFE = False


async def random_moves():
    seed = 19023

    fps = await FPS.create()

    for nn in range(N_MOVES):
        print("")
        print(f"Sending trajectory {nn+1}/{N_MOVES}")

        # Check that all positioners are folded.
        await fps.update_position()
        positions = fps.get_positions(ignore_disabled=True)

        if len(positions) == 0:
            raise RuntimeError("No positioners connected")

        alphaL: float
        betaL: float

        alphaL, betaL = config["kaiju"]["lattice_position"]
        if not numpy.allclose(positions[:, 1:] - [alphaL, betaL], 0, atol=1):
            raise RuntimeError("Not all the positioners are folded.")

        print("Creating random configuration.")

        try:
            configuration = await create_random_configuration(
                fps,
                seed=seed,
                safe=SAFE,
                collision_buffer=None,
                max_retries=10,
            )

            print("Getting trajectory.")
            trajectory = await configuration.get_paths(
                decollide=False,
                collision_buffer=None,
            )
        except JaegerError as err:
            raise RuntimeError(f"jaeger random failed: {err}")

        print("Executing random trajectory.")

        try:
            await fps.send_trajectory(trajectory)
        except TrajectoryError as err:
            raise RuntimeError(f"Trajectory failed with error: {err}")

        print("Reached destination")

        await asyncio.sleep(3)

        print("Reverting to folded")

        try:
            await fps.send_trajectory(configuration.to_destination)
        except TrajectoryError as err:
            raise RuntimeError(f"Trajectory failed with error: {err}")

        print("Folded.")

        seed += 1


if __name__ == "__main__":
    asyncio.run(random_moves())
