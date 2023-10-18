#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-04-11
# @Filename: fix_confSummary.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import os
import pathlib

import numpy
import pandas
from pydl.pydlutils.yanny import yanny
from rich.progress import track
from yaml import warnings


SDSSCORE_DIR = pathlib.Path(os.environ["SDSSCORE_DIR"])
RESULTS = pathlib.Path(os.path.dirname(__file__)) / "../results"


def fix_fvc_image_path():
    data = pandas.read_csv(RESULTS / "confSummaryF_images.csv")

    summary_files = pathlib.Path(SDSSCORE_DIR / "apo/summary_files")
    files = summary_files.glob("**/confSummaryF-*.par")

    for file in track(sorted(list(files))):
        y = yanny(str(file))

        if "fvc_image_path" in y:
            continue

        conf_id = int(y["configuration_id"])
        conf_data = data.loc[data.configuration_id == conf_id]

        if len(conf_data) == 0:
            filename = "NA"
        elif len(conf_data) > 1:
            filename = "NA"
            warnings.warn(f"Multiple entries for configuration ID {conf_id}")
        else:
            filename = conf_data.iloc[0].filename

        y["fvc_image_path"] = filename

        file.unlink()
        y.write(str(file))


def fix_swapped_fibres():
    summary_files = pathlib.Path(SDSSCORE_DIR / "apo/summary_files")
    files = summary_files.glob("**/confSummary*.par")

    for file in track(sorted(list(files))):
        y = yanny(str(file))
        f = y["FIBERMAP"]

        f_78 = numpy.where((f["fiberType"] == b"BOSS") & (f["fiberId"] == 78))[0][0]
        f_80 = numpy.where((f["fiberType"] == b"BOSS") & (f["fiberId"] == 80))[0][0]

        f[f_78]["fiberId"] = 80
        f[f_80]["fiberId"] = 78

        file.unlink()
        y.write(str(file))


if __name__ == "__main__":
    # fix_fvc_image_path()
    fix_swapped_fibres()
