#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-01-10
# @Filename: test_guider_data.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import pandas
import tqdm
from astropy.io.fits import getheader


RESULTS = pathlib.Path(__file__).parent / "../results"


def compile_guider_data():

    proc_files = pathlib.Path("/data/gcam/").glob("595[6-9][0-9]/proc-*.fits")

    data = {}
    for proc_file in tqdm.tqdm(list(proc_files)):
        mjd = int(str(proc_file.parts[-2]))
        frame = int(str(proc_file).split("-")[-1].split(".")[0])
        header = getheader(proc_file, 1)
        cam = int(header["CAMNAME"][3])

        data[(mjd, frame, cam)] = {k.lower(): v for k, v in dict(header).items()}

    df = pandas.DataFrame.from_dict(data, orient="index")
    df.index.set_names(["mjd", "frame", "camera"], inplace=True)

    # for col, dtype in df.dtypes.items():
    #     if dtype == object:
    #         df.loc[:, col] = df[col].str.decode("utf-8")

    fcols = df.select_dtypes("float").columns
    icols = df.select_dtypes("integer").columns

    df[fcols] = df[fcols].apply(pandas.to_numeric, downcast="float")
    df[icols] = df[icols].apply(pandas.to_numeric, downcast="integer")

    df.sort_index(inplace=True)

    outfile = RESULTS / "guider.hdf"
    outfile.unlink(missing_ok=True)

    df.to_hdf(outfile, "data", complevel=9)


if __name__ == "__main__":
    compile_guider_data()
