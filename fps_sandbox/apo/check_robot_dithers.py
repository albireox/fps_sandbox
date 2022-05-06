#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-05-03
# @Filename: check_robot_dithers.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import os
import pathlib

import pandas

from fps_sandbox.apo.check_confSummaryF import read_confSummary


SDSSCORE_DIR = pathlib.Path(os.environ["SDSSCORE_DIR"]) / "apo" / "summary_files"


def check_robot_dither_distribution():

    CONFIGURATION_IDS = [5111, 5112, 5113, 5114, 5093, 5092]

    ys = []
    for cid in CONFIGURATION_IDS:
        for f in [True, False]:
            cid_path_xx = SDSSCORE_DIR / f"{int(cid / 100):04d}XX"
            cid_path = cid_path_xx / f"confSummary{'F' if f else ''}-{cid}.par"
            # if not cid_path.exists():
            #     continue

            header, df = read_confSummary(cid_path)

            df.loc[:, "isFVC"] = int(f)
            df.loc[:, "configurationId"] = int(header["configuration_id"])
            df.loc[:, "parent_configuration"] = float(header["parent_configuration"])
            df.loc[:, "dither_radius"] = float(header["dither_radius"])

            ys.append(df)

    data = pandas.concat(ys)
    data.reset_index(inplace=True)
    data.set_index(["configurationId", "positionerId", "fiberType"], inplace=True)

    print(data)


if __name__ == "__main__":
    check_robot_dither_distribution()
