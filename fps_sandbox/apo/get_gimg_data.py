#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-01-28
# @Filename: get_gimg_data.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re

import pandas
import tqdm
from astropy.io import fits


RESULTS = pathlib.Path(__file__).parent / "../results"
GIMG_DATA = pathlib.Path("/data/gcam/59607")


def get_gimg_data():

    data = []
    for file in tqdm.tqdm(list(GIMG_DATA.glob("proc-*.fits"))):

        match = re.match(r"proc-gimg-gfa(\d)n-(\d+)\.fits", str(file.name))
        if match:
            camera = int(match.group(1))
            seq = int(match.group(2))
        else:
            continue

        f = fits.open(str(file))
        centroids = pandas.DataFrame(f["CENTROIDS"].data.newbyteorder())

        centroids.loc[:, "camera"] = camera
        centroids.loc[:, "seq"] = seq

        data.append(centroids)

    df = pandas.concat(data)
    df.set_index(["seq", "camera"])
    df.to_hdf(RESULTS / "gimg-59607.h5", "data")


if __name__ == "__main__":
    get_gimg_data()
