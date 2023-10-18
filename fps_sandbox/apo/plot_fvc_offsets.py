#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-08
# @Filename: process_fvc_offsets.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import sys

import pandas
from astropy.table import Table
from jaeger import log, logging


log.sh.setLevel(logging.INFO)


async def process_fvc_offsets(proc_image1: str, proc_image2: str, outdir: str):
    """Plot commanded and executed offsets in FVC loop."""

    offsets = Table.read(proc_image1, "OFFSETS").to_pandas()
    reported = Table.read(proc_image2, "POSANGLES").to_pandas()

    offsets.set_index("positioner_id", inplace=True)
    reported.set_index("positionerID", inplace=True)

    offsets = offsets.loc[:, ["alpha_new", "beta_new"]]
    reported = reported.loc[:, ["alphaReport", "betaReport"]]

    data = pandas.merge(reported, offsets, left_index=True, right_index=True)

    breakpoint()


if __name__ == "__main__":
    asyncio.run(process_fvc_offsets(sys.argv[1], sys.argv[2], sys.argv[3]))
