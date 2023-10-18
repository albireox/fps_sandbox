#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-05-03
# @Filename: check_robot_dithers.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from jaeger.target.tools import read_confSummary


matplotlib.use("MacOSX")
seaborn.set_theme()


RESULTS = pathlib.Path(__file__).parent / "../results/dither_distribution"
SDSSCORE_DIR = pathlib.Path(os.environ["SDSSCORE_DIR"]) / "apo" / "summary_files"


def check_robot_dither_distribution(
    parent_id: int,
    dither_id: int,
    flavour_parent: str = "",
    flavour_dither: str = "",
):
    RESULTS.mkdir(exist_ok=True, parents=True)

    header, parent = read_confSummary(parent_id, flavour=flavour_parent)
    header_d, dither = read_confSummary(dither_id, flavour=flavour_dither)

    radius = header_d["dither_radius"] * 1000.0

    wok_delta = dither.loc[:, ["xwok", "ywok"]] - parent.loc[:, ["xwok", "ywok"]]
    wok_delta *= 1000.0

    met = wok_delta.loc[pandas.IndexSlice[:, "METROLOGY"], :]

    xwok_perc = numpy.round((numpy.abs(met.xwok) <= radius).sum() / len(met) * 100.0, 2)
    ywok_perc = numpy.round((numpy.abs(met.ywok) <= radius).sum() / len(met) * 100.0, 2)

    g = seaborn.jointplot(
        x="xwok",
        y="ywok",
        data=met,
        height=10,
        xlim=(-200, 200),
        ylim=(-200, 200),
    )
    g.plot_joint(seaborn.kdeplot, color="g", zorder=10, levels=8)
    g.plot_marginals(seaborn.histplot, kde=True)

    g.ax_joint.text(
        -190,
        185,
        f"{parent_id}{'(F)' if flavour_parent else ''} vs "
        f"{dither_id}{'(F)' if flavour_dither else ''}",
        size="large",
    )
    g.ax_joint.text(
        -190,
        -190,
        f"Design {header['design_id']}. "
        + ((f"Dither {radius} " + r"$\mu{\rm m}$.") if radius > -999.0 else ""),
        size="large",
    )

    if radius > -999.0:
        g.ax_joint.text(
            190,
            185,
            r"$x_{\%}=" + f"{xwok_perc}" + r"\% \quad $"
            r"$y_{\%}=" + f"{ywok_perc}" + r"\%$",
            ha="right",
            size="large",
        )

    g.ax_joint.set_xlabel(r"$\Delta x_{\rm wok}$")
    g.ax_joint.set_ylabel(r"$\Delta y_{\rm wok}$")

    g.fig.savefig(
        RESULTS
        / f"dither_{parent_id}{flavour_parent}_{dither_id}{flavour_dither}_wok.pdf"
    )

    # cos_dec = numpy.cos(numpy.deg2rad(header["decCen"]))
    # sky_delta = dither.loc[:, ["ra", "dec"]] - parent.loc[:, ["ra", "dec"]]
    # sky_delta.ra *= cos_dec
    # sky_delta *= 3600.0

    # g = seaborn.jointplot(
    #     x="ra",
    #     y="dec",
    #     data=sky_delta,
    #     hue="fiberType",
    #     height=10,
    #     xlim=(-3, 3),
    #     ylim=(-3, 3),
    # )
    # g.plot_joint(seaborn.kdeplot, color="r", zorder=10, levels=8)

    # g.ax_joint.text(
    #     -2.8,
    #     2.8,
    #     f"{parent_id}{'(F)' if flavour_parent else ''} vs "
    #     f"{dither_id}{'(F)' if flavour_dither else ''}",
    #     size="large",
    # )
    # g.ax_joint.text(
    #     -2.8,
    #     -2.8,
    #     f"Design {header['design_id']}. Dither {header_d['dither_radius']}.",
    #     size="large",
    # )

    # g.fig.savefig(
    #     RESULTS
    #     / f"dither_{parent_id}{flavour_parent}_{dither_id}{flavour_dither}_sky.pdf"
    # )

    plt.close("all")


def check_fibre_separation(parent_id: int, dither_id: int):
    for config_id in [parent_id, dither_id]:
        _, data = read_confSummary(config_id)

        met = data.loc[(slice(None), "METROLOGY"), :].reset_index(1)
        boss = data.loc[(slice(None), "BOSS"), :].reset_index(1)
        apog = data.loc[(slice(None), "APOGEE"), :].reset_index(1)

        met["dboss"] = numpy.hypot(met.xwok - boss.xwok, met.ywok - boss.ywok)
        met["dapog"] = numpy.hypot(met.xwok - apog.xwok, met.ywok - apog.ywok)

        print()
        print(f"Configuration {config_id}")
        print("dboss")
        print(met.dboss.describe())
        print("dapog")
        print(met.dapog.describe())

    _, parent = read_confSummary(parent_id)
    pmet = parent.loc[(slice(None), "METROLOGY"), :].reset_index(1)
    pboss = parent.loc[(slice(None), "BOSS"), :].reset_index(1)
    papog = parent.loc[(slice(None), "APOGEE"), :].reset_index(1)

    _, dither = read_confSummary(dither_id)
    dmet = dither.loc[(slice(None), "METROLOGY"), :].reset_index(1)
    dboss = dither.loc[(slice(None), "BOSS"), :].reset_index(1)
    dapog = dither.loc[(slice(None), "APOGEE"), :].reset_index(1)

    pmet["delta_dboss"] = (
        numpy.hypot(
            (pmet.xwok - dmet.xwok) - (pboss.xwok - dboss.xwok),
            (pmet.ywok - dmet.ywok) - (pboss.ywok - dboss.ywok),
        )
        * 1000
    )
    pmet["delta_dapog"] = (
        numpy.hypot(
            (pmet.xwok - dmet.xwok) - (papog.xwok - dapog.xwok),
            (pmet.ywok - dmet.ywok) - (papog.ywok - dapog.ywok),
        )
        * 1000.0
    )

    breakpoint()


if __name__ == "__main__":
    # check_fibre_separation(5203, 5204)

    for parent_id in range(5203, 5222, 2):
        dither_id = parent_id + 1
        check_robot_dither_distribution(
            parent_id,
            parent_id,
            "",
            "F",
        )
        check_robot_dither_distribution(
            dither_id,
            dither_id,
            "",
            "F",
        )
        for flavour_parent in ["", "F"]:
            for flavour_dither in ["", "F"]:
                check_robot_dither_distribution(
                    parent_id,
                    dither_id,
                    flavour_parent,
                    flavour_dither,
                )
