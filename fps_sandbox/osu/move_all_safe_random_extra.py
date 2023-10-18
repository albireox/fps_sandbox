#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-08-02
# @Filename: move_all_safe_random.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio
import os

import numpy
from jaeger import FPS, log


log.sh.setLevel(20)


async def main():
    traj_no_file = "/tmp/n_traj.dat"
    n_traj = 1
    if os.path.exists(traj_no_file):
        n_traj = int(open(traj_no_file, "r").read()) + 1

    fps = await FPS.create()

    positioners = list(fps.keys())

    bad = [683, 239]
    for pbad in bad:
        if pbad in positioners:
            positioners.remove(pbad)

    await fps.stop_trajectory()

    print("Making sure all positioners are folded.")

    await fps.goto(
        positioners,
        0,
        180,
        speed=(2000, 2000),
        force=True,
        use_sync_line=False,
    )

    await asyncio.sleep(1)

    min_beta = 165
    max_beta = 195

    while True:
        alpha = numpy.random.random(len(positioners)) * 360
        beta = numpy.random.random(len(positioners)) * (max_beta - min_beta) + min_beta

        print(f"Trajectory {n_traj}")

        print("Forward move. Destination positions follow:")
        for nn in range(len(alpha)):
            print(f"Pos. {positioners[nn]}: ({alpha[nn]:.2f}, {beta[nn]:.2f})")

        try:
            await fps.goto(
                positioners,
                alpha,
                beta,
                speed=(2000, 2000),
                force=True,
                use_sync_line=False,
            )

            await asyncio.sleep(1)

            print("Folding to (0, 180).")

            await fps.goto(
                positioners,
                0,
                180,
                speed=(2000, 2000),
                force=True,
                use_sync_line=False,
            )

            await asyncio.sleep(1)

        except Exception:
            print(
                f"An exception was raised during trajectory {n_traj}. "
                "Printing the position of the robots."
            )
            await fps.update_position()
            for pos in fps.values():
                print(f"Pos. {pos.positioner_id}: ({pos.alpha:.2f}, {pos.beta:.2f})")

            print("Now raising exception")
            raise

        with open(traj_no_file, "w") as f:
            f.write(str(n_traj))

        n_traj += 1


asyncio.run(main())
