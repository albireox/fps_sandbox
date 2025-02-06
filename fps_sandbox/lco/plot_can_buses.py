#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-01-02
# @Filename: plot_can_buses.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re

from typing import TypedDict

import matplotlib.pyplot as plt
import polars
import seaborn
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from coordio.defaults import calibration

import fps_sandbox


class MapPositionerToHole(TypedDict):
    positioner: int
    sextant: int
    bus: int


ROOT = pathlib.Path(fps_sandbox.__file__).parent / ".."


def parse_log_data(file: pathlib.Path):
    """Parses the ``positioner_status`` lines and maps positioner to hole and bus."""

    REGEX = re.compile(
        r"^.+positioner_status="
        r"(?P<positioner_id>\d+),"
        r"(?P<alpha_pos>[\d\.]+),"
        r"(?P<beta_pos>[\d\.]+),"
        r"(?P<status>.+?),"
        r"(?P<initialised>.+?),"
        r"(?P<disabled>.+?),"
        r"(?P<offline>.+?),"
        r"(?P<firmware>.+?),"
        r"(?P<sextant>\d+),"
        r"(?P<bus>\d+),"
        r"(?P<n_trajs_pid>[\d\?]+).*$"
    )

    with open(file, "r") as f:
        lines = f.readlines()

    positioner_to_bus: dict[int, MapPositionerToHole] = {}

    for line in lines:
        match = REGEX.match(line)
        if match:
            data = match.groupdict()
            positioner_to_bus[int(data["positioner_id"])] = {
                "positioner": int(data["positioner_id"]),
                "sextant": int(data["sextant"]),
                "bus": int(data["bus"]),
            }

    return positioner_to_bus


def get_fibre_assignments():
    """Returns the fibre assignments as a Polars dataframe."""

    return (
        polars.DataFrame(calibration.fiberAssignments.reset_index())
        .select(["positionerID", "holeID", "Device", "Sextant", "CAN", "Row", "Column"])
        .rename(
            {
                "positionerID": "positioner_id",
                "holeID": "hole_id",
                "Device": "device",
                "Sextant": "sextant",
                "CAN": "bus",
                "Row": "row",
                "Column": "column",
            }
        )
        .cast(
            {
                "positioner_id": polars.Int32,
                "hole_id": polars.String,
                "device": polars.String,
                "sextant": polars.Int16,
                "bus": polars.Int16,
                "row": polars.String,
                "column": polars.String,
            }
        )
        .sort("positioner_id")
        .with_row_index(offset=0)
    )


def check_fiber_assignments(silent: bool = False):
    """Checks the sextant-to-bus assignments."""

    INPUT_DATA = ROOT / "data" / "lco" / "positioner_status.dat"

    positioner_to_bus_fps = parse_log_data(INPUT_DATA)

    fiber_assignments = get_fibre_assignments()
    positioners = fiber_assignments.filter(polars.col.device == "Positioner")

    assert len(positioners) == len(positioner_to_bus_fps)

    for p_id, data in positioner_to_bus_fps.items():
        fa_data = positioners.filter(polars.col.positioner_id == p_id).to_dicts()[0]

        if data["sextant"] != fa_data["sextant"]:
            raise ValueError(
                f"Positioner {p_id} has different sextant: "
                f"jaeger={data['sextant']} vs FA={fa_data['sextant']}."
            )

        if data["bus"] != fa_data["bus"]:
            if not silent:
                print(
                    f"Positioner {p_id} (sextant {data['sextant']}) has different bus: "
                    f"FPS={data['bus']} vs FA={fa_data['bus']}. "
                    "Updating fibre assignments."
                )

            index = fa_data["index"]
            fiber_assignments[index, "bus"] = data["bus"]

    return fiber_assignments


def plot_can_buses(fiber_assignments: polars.DataFrame, from_above: bool = True):
    """Plots the CAN buses for each sextant."""

    OUTPUT = ROOT / "results" / "lco"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    BUS_TO_COLOUR = {1: "g", 2: "r", 3: "b", 4: "y"}

    seaborn.set_theme(style="white", font_scale=1.2)
    plt.ioff()

    wok_coords = (
        polars.DataFrame(calibration.wokCoords.reset_index())
        .select(["holeID", "xWok", "yWok"])
        .rename({"holeID": "hole_id", "xWok": "x_wok", "yWok": "y_wok"})
        .cast({"x_wok": polars.Float32, "y_wok": polars.Float32})
    )

    fiber_assignments = fiber_assignments.join(wok_coords, on="hole_id", how="inner")

    fig, ax = plt.subplots(figsize=(15, 15))
    fig.tight_layout(rect=(0.025, 0.025, 0.975, 0.975))

    for row in fiber_assignments.to_dicts():
        if row["device"] != "Positioner":
            ax.add_patch(
                Circle(
                    (row["x_wok"], row["y_wok"]),
                    3,
                    edgecolor="0.5",
                    facecolor="0.5",
                    alpha=0.5,
                )
            )
            ax.add_patch(
                Circle(
                    (row["x_wok"], row["y_wok"]),
                    6,
                    edgecolor="0.5",
                    fill=False,
                    alpha=0.5,
                )
            )
            continue

        ax.add_patch(
            Circle(
                (row["x_wok"], row["y_wok"]),
                8,
                edgecolor="0.5",
                facecolor=BUS_TO_COLOUR[row["bus"]],
                alpha=0.5,
            )
        )

        ax.text(
            row["x_wok"],
            row["y_wok"],
            f"{row['hole_id']}",
            fontsize=5,
            ha="center",
            va="center",
            color="k",
        )

    # Add the legend
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markeredgecolor="0.5",
                markerfacecolor=BUS_TO_COLOUR[bus],
                markersize=15,
                alpha=0.5,
            )
            for bus in BUS_TO_COLOUR
        ],
        labels=[f"CAN {bus}" for bus in BUS_TO_COLOUR],
        loc="lower right",
        fontsize=14,
        frameon=False,
    )

    # Add sextant labels
    ax.text(
        -260,
        140,
        "Sextant 1",
        fontsize=24,
        ha="center",
        va="center",
        color="0.2",
        rotation=60 if from_above else -60,
    )
    ax.text(
        0,
        290,
        "Sextant 2",
        fontsize=24,
        ha="center",
        va="center",
        color="0.2",
        rotation=0,
    )
    ax.text(
        260,
        140,
        "Sextant 3",
        fontsize=24,
        ha="center",
        va="center",
        color="0.2",
        rotation=-60 if from_above else 60,
    )
    ax.text(
        260,
        -140,
        "Sextant 4",
        fontsize=24,
        ha="center",
        va="center",
        color="0.2",
        rotation=60 if from_above else -60,
    )
    ax.text(
        0,
        -290,
        "Sextant 5",
        fontsize=24,
        ha="center",
        va="center",
        color="0.2",
        rotation=0,
    )
    ax.text(
        -260,
        -140,
        "Sextant 6",
        fontsize=24,
        ha="center",
        va="center",
        color="0.2",
        rotation=-60 if from_above else 60,
    )
    # -145.600006 ┆ -252.1866
    ax.plot([-300, 5], [10, 10], "m--")  # Border Sextant 6-1
    ax.plot([-5, 300], [-10, -10], "m--")  # Border Sextant 3-4
    ax.plot([-145.6 - 17.5, -5], [-252.2 - 10, 10], "m--")  # Border Sextant 5-6
    ax.plot([145.6 - 5, -11], [-252.2 - 10, 0], "m--")  # Border Sextant 4-5
    ax.plot([145.6 + 17.5, 5], [252.2 + 10, -10], "m--")  # Border Sextant 2-3
    ax.plot([-145.6 + 5, 11], [252.2 + 10, 0], "m--")  # Border Sextant 1-2

    # breakpoint()
    ax.set_xlabel(r"$x_{\rm wok}\,\rm [mm]$", fontdict={"size": 18})
    ax.set_ylabel(r"$y_{\rm wok}\,\rm [mm]$", fontdict={"size": 18})

    if from_above:
        ax.set_xlim(-320, 320)
    else:
        ax.set_xlim(320, -320)
    ax.set_ylim(-320, 320)

    fig.savefig(OUTPUT / f"can_buses_{'above' if from_above else 'below'}.pdf")


if __name__ == "__main__":
    fa = check_fiber_assignments(silent=True)
    plot_can_buses(fa, from_above=False)
