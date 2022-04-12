#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-04-04
# @Filename: analyse_fvc_rms.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import pandas
import seaborn


matplotlib.use("TkAgg")

seaborn.set_theme()


RESULTS = pathlib.Path(os.path.dirname(__file__)) / "../results"
DATA_FILE = pathlib.Path(RESULTS / "dither_data_fvc.h5")


def create_dataframe():

    data = pandas.read_hdf(DATA_FILE)

    on_target = data.groupby(["configuration_id", "positionerId"]).filter(
        lambda g: (g.on_target == 1).any() & (g.valid == 1).any()
    )

    met = on_target.loc[pandas.IndexSlice[:, :, "METROLOGY"], :]

    base = met.loc[met.isFVC == 0]
    fvc = met.loc[met.isFVC == 1]

    df = base.join(fvc, rsuffix="_fvc", how="inner")
    df = df.dropna(subset=["xwok", "ywok", "xwok_fvc", "ywok_fvc"])

    df["distance"] = (
        (df.xwok - df.xwok_fvc) ** 2 + (df.ywok - df.ywok_fvc) ** 2
    ) ** 0.5

    parent_confs = data.parent_configuration.dropna().unique().astype(int)

    df["is_parent"] = 0
    df.loc[df.index.get_level_values(0).isin(parent_confs), "is_parent"] = 1

    filename = RESULTS / "dither_data_fvc_parent.h5"
    if filename.exists():
        filename.unlink()

    df.to_hdf(filename, "data")


def plot_all():

    data = pandas.read_hdf(RESULTS / "dither_data_fvc_parent.h5").reset_index()
    data["distance"] *= 1000.0

    parent = data.loc[data.is_parent == 1]

    fig, ax = plt.subplots(figsize=(20, 8))

    ax.scatter(
        data.configuration_id,
        data.distance.values,
        color="g",
        alpha=0.1,
        ec="None",
        label="All configurations",
    )
    ax.scatter(
        parent.configuration_id,
        parent.distance.values,
        color="y",
        alpha=0.1,
        ec="None",
        label="Parent configurations",
    )

    seaborn.lineplot(x="configuration_id", y="distance", data=data, ax=ax)
    seaborn.lineplot(x="configuration_id", y="distance", data=parent, ax=ax)

    ax.set_yscale("log")

    ax.set_xlabel("Configuration ID")
    ax.set_ylabel(r"Distance [$\mu$m]")

    fig.savefig(RESULTS / "configuration_vs_distance.pdf")


if __name__ == "__main__":
    # create_dataframe()
    plot_all()
