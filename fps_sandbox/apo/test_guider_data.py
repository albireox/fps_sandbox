#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-01-10
# @Filename: test_guider_data.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import pandas
import seaborn
import tqdm
from astropy.io.fits import getheader
from astropy.time import Time


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


def plot_scale():

    data = pandas.read_hdf(RESULTS / "guider.hdf")

    data = data.dropna(subset=["deltascl", "date-obs", "airtemp"])
    data = data.loc[(data.deltascl > 0) & (data.capplied | data.e_radec)]

    g = data.groupby(["mjd", "frame"]).first()

    s = g.loc[:, ["date-obs", "deltascl", "airtemp"]]
    s.reset_index(inplace=True)
    s.loc[:, "date"] = Time(s["date-obs"].tolist(), format="iso").mjd - s.mjd
    s = s.sort_values("date")

    seaborn.set_style("darkgrid")

    ax = seaborn.lineplot(x="date", y="deltascl", data=s, hue="mjd")
    ax.set_ylim(0.9975, 1.0025)

    ax.figure.savefig(RESULTS / "guider_scale.pdf")

    plt.close(ax.figure)

    ax_temp = seaborn.scatterplot(x="airtemp", y="deltascl", data=s, s=3)
    ax_temp.set_ylim(0.9975, 1.0025)
    ax_temp.figure.savefig(RESULTS / "guider_scale_temp.pdf")


if __name__ == "__main__":

    # compile_guider_data()
    plot_scale()
