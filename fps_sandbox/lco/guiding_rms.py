#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-04-16
# @Filename: guiding_rms.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import pathlib

from cherno.guider import Guider


GCAM_PATH = "/data/gcam/lco"

MJD = 60050
FRAME_NO = 367


async def get_ast_solution(images: list[pathlib.Path]):
    """Returns an astrometric solution."""

    guider = Guider("LCO")
    return await guider.process(
        None,
        images,
        write_proc=False,
        correct=False,
        fit_focus=False,
    )


async def guiding_rms(mjd: int, frame_no: int):
    """Selects images for an MJD and frame number and gets the astrometric solution."""

    path = pathlib.Path(GCAM_PATH) / str(mjd)
    files = list(path.glob(f"gimg-gfa[1-6][sn]-{frame_no:04d}.fits"))
    if len(files) == 0:
        return

    await get_ast_solution(files)


if __name__ == "__main__":
    asyncio.run(guiding_rms(MJD, FRAME_NO))
    print()
