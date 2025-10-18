#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-03-07
# @Filename: add_extra_too_columns.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

import numpy
import numpy.lib.recfunctions as rfn
import polars

from sdsstools import yanny
from sdsstools._vendor.yanny import write_ndarray_to_yanny


DATA = pathlib.Path(__file__).parent / "../data"
RESULTS = pathlib.Path(__file__).parent / "../results"
SDSSCORE_DIR = pathlib.Path(os.environ["SDSSCORE_DIR"])


def add_extra_too_columns():
    """Adds ``too_id`` and ``too_program`` to confSummary files."""

    too_data = polars.read_parquet(DATA / "too_data.parquet")
    min_cids = {"APO": 18945, "LCO": 10012048}

    for obs in ["APO", "LCO"]:
        summ_files = SDSSCORE_DIR / obs.lower() / "summary_files"
        conf_files = list(summ_files.glob("**/confSummary-*.par"))

        replace_files: list[pathlib.Path] = []
        for conf_file in conf_files:
            name = conf_file.name
            mjd = int(name.split("-")[1].split(".")[0])
            if mjd >= min_cids[obs]:
                replace_files.append(conf_file)

        print(f"Found {len(replace_files)} files to replace for {obs}.")

        for conf_file in replace_files:
            for flavour in ["", "F"]:
                conf_file_flavour = conf_file.with_name(
                    conf_file.name.replace("confSummary", f"confSummary{flavour}")
                )

                if not conf_file_flavour.exists():
                    continue

                y_data = yanny(str(conf_file_flavour))
                cols = y_data["FIBERMAP"].dtype.names
                if "too_id" in cols or "too" not in cols:
                    continue

                conf_id = int(y_data["configuration_id"])

                # Delete parquet files allow for recreation.
                parquet_files = conf_file_flavour.parent.glob(
                    f"confSummary*-{conf_id}.parquet"
                )
                for file in parquet_files:
                    file.unlink()

                s_files = conf_file_flavour.parent.glob(
                    f"confSummary{flavour}S-{conf_id}.par"
                )
                for file in s_files:
                    file.unlink()

                fmap = y_data["FIBERMAP"]
                too_id = numpy.full(
                    fmap.shape[0],
                    -999,
                    dtype=[("too_id", "int64")],
                )
                too_program = numpy.full(
                    fmap.shape[0],
                    "",
                    dtype=[("too_program", "|U100")],
                )

                fmap = rfn.merge_arrays((fmap, too_id, too_program), flatten=True)

                for nn in range(fmap.shape[0]):
                    if not fmap[nn]["too"]:
                        continue

                    cid = fmap[nn]["catalogid"]
                    if cid < 0:
                        continue

                    too_cid_data = too_data.filter(polars.col.catalogid == cid)
                    if len(too_cid_data) == 0:
                        continue

                    too_id = too_cid_data["too_id"][0]
                    too_program = too_cid_data["too_program"][0]
                    if too_id == -999:
                        continue

                    fmap[nn]["too_id"] = too_id
                    fmap[nn]["too_program"] = too_program

                y_data.pop("FIBERMAP")
                conf_file_flavour.unlink()

                write_ndarray_to_yanny(
                    str(conf_file_flavour),
                    fmap,
                    structnames="FIBERMAP",
                    enums={
                        "fiberType": (
                            "FIBERTYPE",
                            ("BOSS", "APOGEE", "METROLOGY", "NONE"),
                        )
                    },
                    hdr=dict(y_data),
                )
