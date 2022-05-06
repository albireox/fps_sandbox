#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-05-03
# @Filename: plot_fvc_distances.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import os
import pathlib

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn

from fps_sandbox.apo.check_confSummaryF import read_confSummary


seaborn.set_theme()


CONFIGURATION_IDS = [
    5121,
    5122,
    5123,
    5124,
    5125,
    5126,
    5127,
    5128,
    5129,
    5130,
    5132,
    5133,
]

SDSSCORE_DIR = pathlib.Path(os.environ["SDSSCORE_DIR"]) / "apo" / "summary_files"
RESULTS = pathlib.Path(__file__).parent / "../results/fvc_results"


def _plot_wok_distance(
    data_F: pandas.DataFrame,
    header_F: dict,
    ax: plt.Axes,
    is_dither: bool = False,
):

    colours = ["g", "r", "b"]
    for ii, fibre in enumerate(["Metrology", "APOGEE", "BOSS"]):

        data_fibre = data_F.loc[pandas.IndexSlice[:, fibre.upper()], :].copy()

        if fibre != "Metrology" and is_dither is False:
            data_fibre = data_fibre.loc[
                (data_fibre.assigned == 1) & (data_fibre.on_target == 1), :
            ]
        if len(data_fibre) == 0:
            continue

        wok_distance = data_fibre.wok_distance * 1000.0
        wok_distance_bin = numpy.histogram(wok_distance, bins=numpy.arange(0, 105, 5))

        perc_90 = numpy.percentile(wok_distance, 90)
        ax.plot(
            wok_distance_bin[1][1:] - wok_distance_bin[1][1] / 2.0,  # type: ignore
            wok_distance_bin[0],
            linestyle="-",
            color=colours[ii],
            label=rf"{fibre}: {perc_90:.1f} $\mu$m",
            zorder=10,
        )

        if fibre == "Metrology":
            for perc_q in [90]:
                perc = numpy.percentile(wok_distance, perc_q)
                ax.axvline(x=perc, color="k", linestyle="--", linewidth=0.5, zorder=0)

        ax.set_xlim(0, 100)

        ax.set_xlabel("Wok distance [microns]")
        ax.set_ylabel("Number")

        perc_90_header = float(header_F["fvc_90_perc"]) * 1000.0
        ax.set_title(rf"Wok 90\% percentile: {perc_90_header:.1f} $\mu$m")

        ax.legend()


def _plot_sky_distance(
    data_F: pandas.DataFrame,
    ax: plt.Axes,
    column: str,
    is_dither: bool = False,
    plot_metrology: bool = True,
    title: str = "Sky distance",
):

    colours = ["g", "r", "b"]
    for ii, fibre in enumerate(["Metrology", "APOGEE", "BOSS"]):
        if fibre == "Metrology" and not plot_metrology:
            continue

        data_fibre = data_F.loc[pandas.IndexSlice[:, fibre.upper()], :].copy()

        if fibre != "Metrology" and is_dither is False:
            data_fibre = data_fibre.loc[
                (data_fibre.assigned == 1) & (data_fibre.on_target == 1), :
            ]
        if len(data_fibre) == 0:
            continue

        sky_distance = data_fibre[column]
        sky_distance_bin = numpy.histogram(sky_distance, bins=numpy.arange(0, 1.5, 0.1))

        perc_90 = numpy.percentile(sky_distance, 90)
        ax.plot(
            sky_distance_bin[1][1:] - sky_distance_bin[1][1] / 2.0,  # type: ignore
            sky_distance_bin[0],
            linestyle="-",
            color=colours[ii],
            label=f"{fibre}: {perc_90:.2f} arcsec",
            zorder=10,
        )

        if fibre == "Metrology":
            for perc_q in [90]:
                perc = numpy.percentile(sky_distance, perc_q)
                ax.axvline(x=perc, color="k", linestyle="--", linewidth=0.5, zorder=0)

        ax.set_xlim(0, 1.5)

        ax.set_xlabel("Sky distance [arcsec]")
        ax.set_ylabel("Number")

        ax.set_title(title)

        ax.legend()


def _plot_sky_quiver(data_F: pandas.DataFrame, ax: plt.Axes, is_dither: bool = False):

    colours = ["r", "b"]
    key = False
    for ii, fibre in enumerate(["APOGEE", "BOSS"]):

        data_fibre = data_F.loc[pandas.IndexSlice[:, fibre.upper()], :].copy()

        if is_dither is False:
            data_fibre = data_fibre.loc[
                (data_fibre.assigned == 1) & (data_fibre.on_target == 1), :
            ]
        if len(data_fibre) == 0:
            continue

        q = ax.quiver(
            data_fibre.ra,
            data_fibre.dec,
            data_fibre.ra_distance,
            data_fibre.dec_distance,
            color=colours[ii],
            label=fibre,
        )

        if key is False:
            ax.quiverkey(
                q,
                X=0.05,
                Y=0.05,
                U=0.2,
                label="0.2 arcsec",
                labelpos="E",
            )
            key = True

        ax.set_xlabel("Right Ascension")
        ax.set_ylabel("Declination")

        ax.set_title("confSummary(ra/dec) - confSummaryF(ra/dec)")

        ax.legend()


def plot_fvc_distances(configuration_id: int):

    cid_path_xx = SDSSCORE_DIR / f"{int(configuration_id / 100):04d}XX"

    cid_path = cid_path_xx / f"confSummary-{configuration_id}.par"
    data = read_confSummary(cid_path)[1]

    cid_path_F = cid_path_xx / f"confSummaryF-{configuration_id}.par"
    header_F, data_F = read_confSummary(cid_path_F)

    is_dither = True if int(header_F["parent_configuration"]) != -999 else False

    if not is_dither:
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        data_F = data_F.groupby("positionerId").filter(
            lambda g: g.assigned.any() & g.on_target.any() & g.valid.all()
        )

    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))

    assert isinstance(axes, numpy.ndarray)

    data = data.loc[data_F.index, :]

    data_F["xwok_distance"] = data.xwok - data_F.xwok
    data_F["ywok_distance"] = data.ywok - data_F.ywok
    data_F["wok_distance"] = numpy.hypot(data_F.xwok_distance, data_F.ywok_distance)

    cos_dec = numpy.cos(numpy.deg2rad(float(header_F["decCen"])))
    data_F["ra_distance"] = (data.ra - data_F.ra) * cos_dec * 3600.0
    data_F["dec_distance"] = (data.dec - data_F.dec) * 3600.0
    data_F["sky_distance"] = numpy.hypot(data_F.ra_distance, data_F.dec_distance)

    if not is_dither:
        data_F["racat_distance"] = (data_F.racat - data_F.ra) * cos_dec * 3600.0
        data_F["deccat_distance"] = (data_F.deccat - data_F.dec) * 3600.0
        data_F["skycat_distance"] = numpy.hypot(
            data_F.racat_distance, data_F.deccat_distance
        )

    _plot_wok_distance(data_F, header_F, axes[0, 0])

    _plot_sky_distance(
        data_F,
        axes[0, 1],
        "sky_distance",
        is_dither=is_dither,
        plot_metrology=True,
        title="Sky distance (ra/dec vs ra/dec)",
    )

    if not is_dither:
        _plot_sky_distance(
            data_F,
            axes[1, 0],
            "skycat_distance",
            is_dither=is_dither,
            plot_metrology=False,
            title="Sky distance (ra/dec vs racat/deccat)",
        )

    _plot_sky_quiver(data_F, axes[1, 1], is_dither=is_dither)

    fig.suptitle(
        f"Configuration ID: {configuration_id}" + (" (dithered)" if is_dither else "")
    )

    RESULTS.mkdir(exist_ok=True)
    plt.tight_layout()
    fig.savefig(RESULTS / f"fvc_distance_{configuration_id}.pdf")


if __name__ == "__main__":

    for configuration_id in CONFIGURATION_IDS:
        plot_fvc_distances(configuration_id)
