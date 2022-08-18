#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-06-10
# @Filename: analyse_trajectories.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import json
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import pandas
import seaborn


matplotlib.use("Agg")
# matplotlib.use("MacOSX")

seaborn.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})


TRAJECTORIES = pathlib.Path(__file__).parents[1] / "data" / "trajectories"


def parse_trajectory(trajectory_file: pathlib.Path):

    data = json.loads(open(trajectory_file, "r").read())
    if data["success"]:
        return
    rows = []
    for robot_id in data["trajectory"]:
        for axis in data["trajectory"][robot_id]:
            axis_data = data["trajectory"][robot_id][axis]
            for angle, time in axis_data:
                rows.append((int(robot_id), axis, time, angle))

    df = pandas.DataFrame(data=rows, columns=["positioner_id", "axis", "time", "angle"])
    df = df.sort_values(["positioner_id", "axis", "time"])
    df = df.set_index(["positioner_id", "axis"])

    g = df.groupby(["positioner_id", "axis"])
    df["velocity"] = g["angle"].diff() / g["time"].diff() / 6.0

    def zeroer(x):
        x.iloc[0] = 0
        return x

    # The first velocity per robot trajectory is NaN, set it to zero.
    g = df.groupby(["positioner_id", "axis"])
    df["velocity"] = g["velocity"].transform(zeroer)

    max_velocity = max(min(df.velocity), max(df.velocity), key=abs)
    if max_velocity > 3.0:
        print(str(trajectory_file.name), abs(max_velocity), data["success"])

    return data, df


def plot_velocity(data: dict, df: pandas.DataFrame):

    df = df.reset_index()

    vel_g = df.groupby(["positioner_id", "axis"])["velocity"]
    df["velocity_norm"] = vel_g.transform(lambda x: x / max(x.min(), x.max(), key=abs))

    g = df.groupby(["positioner_id"])
    df["velocity_off"] = g["velocity_norm"].transform(
        lambda x: x + g.positioner_id.first()
    )

    alpha = df.loc[df.axis == "alpha"]

    pal = seaborn.cubehelix_palette(
        len(alpha.positioner_id.unique()),
        rot=-0.25,
        light=0.7,
    )

    fig, ax = plt.subplots(1, 1, figsize=(20, 50))

    lp = seaborn.lineplot(
        data=alpha,
        x="time",
        y="velocity_off",
        hue="positioner_id",
        palette=pal,
        ax=ax,
    )

    fig = lp.get_figure()
    fig.savefig("test.pdf")


if __name__ == "__main__":

    files = sorted(TRAJECTORIES.glob("**/*.json"))

    for file in files:
        try:
            parse_trajectory(file)
        except Exception:
            pass
