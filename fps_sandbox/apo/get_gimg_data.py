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


MJD = 59608
RESULTS = pathlib.Path(__file__).parent / "../results"
GIMG_DATA = pathlib.Path(f"/data/gcam/{MJD}")


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
    df = df.set_index(["seq", "camera"])
    df.to_hdf(RESULTS / f"gimg-{MJD}.h5", "data")


if __name__ == "__main__":
    get_gimg_data()
