#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-02-09
# @Filename: test_focus_fit.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio
import pathlib
from time import time

from cherno import config
from cherno.acquisition import Acquisition


MJD = 59620
SEQ = 120


async def test_focus_fit():

    path = pathlib.Path(f"/data/gcam/{MJD}")

    t0 = time()

    for seq in range(200, 201):
        images = path.glob(f"gimg-*-{seq:04d}.fits")

        config["extraction"]["output_dir"] = "/home/gallegoj/tmp/extraction"
        config["acquisition"]["astrometry_dir"] = "/home/gallegoj/tmp/astrometry"

        acquisition = Acquisition("APO")
        await acquisition.process(
            None,
            list(images),
            write_proc=False,
            correct=False,
        )

        print(round(time() - t0, 1))
        t0 = time()

        print()

    # print(acq_data)


if __name__ == "__main__":
    asyncio.run(test_focus_fit())
