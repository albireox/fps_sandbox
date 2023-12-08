#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-08-06
# @Filename: get_hall_calibration.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio

from astropy.table import Table

from jaeger import FPS


async def main():
    fps = await FPS.create()

    alpha_hall_calib = (await fps.send_command("GET_ALPHA_HALL_CALIB")).get_values()
    beta_hall_calib = (await fps.send_command("GET_BETA_HALL_CALIB")).get_values()
    hall_error = (await fps.send_command("GET_HALL_CALIB_ERROR")).get_values()

    rows = []
    for pid in alpha_hall_calib:
        rows.append(
            (
                pid,
                *alpha_hall_calib[pid],
                *beta_hall_calib[pid],
                *hall_error[pid],
            )
        )

    table = Table(
        rows=rows,
        names=[
            "pid",
            "maxA_alpha",
            "maxB_alpha",
            "minA_alpha",
            "minB_alpha",
            "maxA_beta",
            "maxB_beta",
            "minA_beta",
            "minB_beta",
            "error_alpha",
            "error_beta",
        ],
    )

    table.sort("pid")

    table.write("hall_calib.dat", format="ascii.fixed_width", delimiter="|")


asyncio.run(main())
