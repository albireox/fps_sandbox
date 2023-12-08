#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-08
# @Filename: reprocess_fvc.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile

import pandas
from astropy.io import fits

from jaeger import log, logging
from jaeger.fvc import FVC


log.sh.setLevel(logging.INFO)


async def reprocess_fvc(proc_image: str, outdir: str, proc_image2: str | None = None):
    """Reprocesses an FVC image."""

    fvc = FVC("APO")

    proc_orig = fits.open(proc_image)
    tmp_raw = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=proc_orig[1].data[:, ::-1],
                header=proc_orig[1].header,
            ),
        ]
    )
    tmp_dir = tempfile.mkdtemp()
    tmp_raw_filename = os.path.join(tmp_dir, os.path.basename(proc_image)[5:])
    tmp_raw.writeto(tmp_raw_filename)

    fibre_data = pandas.DataFrame(proc_orig["FIBERDATA"].data)

    for col in ["xwok_measured", "ywok_measured", "mismatched"]:
        if col in fibre_data:
            fibre_data.drop(columns=col, inplace=True)

    fibre_data.set_index(["positioner_id", "fibre_type"], inplace=True)

    fvc.process_fvc_image(
        tmp_raw_filename,
        fibre_data=fibre_data,
        outdir=outdir,
        plot=True,
    )

    offsets = pandas.DataFrame(proc_orig["OFFSETS"].data)

    fvc.calculate_offsets(
        offsets.loc[:, ["positioner_id", "alpha_reported", "beta_reported"]].to_numpy()
    )
    await fvc.write_proc_image(os.path.join(outdir, os.path.basename(proc_image)))

    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    asyncio.run(
        reprocess_fvc(
            sys.argv[1],
            sys.argv[2],
            proc_image2=sys.argv[3] if len(sys.argv) >= 4 else None,
        )
    )
