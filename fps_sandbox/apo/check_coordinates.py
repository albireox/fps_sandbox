#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-01-05
# @Filename: check_coordinates.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib
import re

from typing import cast

import numpy
import pandas
import seaborn
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from pydl.pydlutils.yanny import yanny
from rich.progress import track

from coordio.utils import radec2wokxy, wokxy2radec


RESULTS = pathlib.Path(os.path.dirname(__file__)) / "../results"
SDSSCORE_DIR = pathlib.Path(os.environ["SDSSCORE_DIR"])


def create_dataframe(start_mjd: int = 59564, onlyF: bool = False):
    """Creates a dataframe with all dither sequences."""

    summaryF_files = sorted(SDSSCORE_DIR.glob("apo/**/confSummaryF*"))
    summary_files = sorted(SDSSCORE_DIR.glob("apo/**/confSummary-*"))

    path = os.path.join(RESULTS, "dither_data_fvc.h5")

    if os.path.exists(path):
        existing = pandas.read_hdf(path)
        existing.reset_index(inplace=True)
    else:
        existing = None

    dithers = []

    for summary_file in track(summaryF_files + summary_files):

        isFVC = 1 if "confSummaryF" in str(summary_file) else 0
        if isFVC == 0 and onlyF:
            continue

        cid = int(str(summary_file).split("-")[1].split(".")[0])

        with open(summary_file, "r") as f:
            mjd = None
            for line in f:
                if (match := re.search(r"MJD\s+([0-9]+)", line)) is not None:
                    mjd = int(match.group(1))
                    break
            if mjd is not None and mjd < start_mjd:
                continue

        if existing is not None:
            entries = existing.loc[
                (existing.isFVC == isFVC) & (existing.configuration_id == cid)
            ]
            if len(entries) > 0:
                continue

        summary = yanny(str(summary_file))

        if "MJD" in summary and float(summary["MJD"]) < start_mjd:
            continue

        fibermap = summary["FIBERMAP"]
        names = list(fibermap.dtype.names)
        names.remove("mag")
        fibermap = fibermap[names]

        df = pandas.DataFrame(fibermap)

        df.loc[:, "configuration_id"] = cid
        df.loc[:, "isFVC"] = isFVC

        if "is_dithered" not in summary or summary["is_dithered"] == "0":
            df.loc[:, "parent_configuration"] = numpy.nan

        if summary["is_dithered"] == "1":
            df.loc[:, "parent_configuration"] = int(summary["parent_configuration"])
            df.loc[:, "dither_radius"] = float(summary["dither_radius"])

        df.loc[:, "mjd"] = int(summary["MJD"])
        df.loc[:, "epoch"] = float(summary["epoch"])
        df.loc[:, "racen"] = float(summary["raCen"])
        df.loc[:, "deccen"] = float(summary["decCen"])

        df.loc[:, ["focal_scale", "temperature"]] = numpy.nan
        if "focal_scale" in summary:
            df.loc[:, "focal_scale"] = float(summary["focal_scale"])
        if "temperature" in summary:
            df.loc[:, "temperature"] = float(summary["temperature"])

        df_numeric = df.select_dtypes(["number"])
        df_numeric["fiberType"] = df["fiberType"]

        dithers.append(df_numeric)

    dithers_df = pandas.concat(dithers)

    for col, dtype in dithers_df.dtypes.items():
        if dtype == object:
            dithers_df.loc[:, col] = dithers_df[col].str.decode("utf-8")

    if existing is not None:
        dithers_df = pandas.concat([existing, dithers_df])

    fcols = dithers_df.select_dtypes("float").columns
    icols = dithers_df.select_dtypes("integer").columns

    dithers_df[fcols] = dithers_df[fcols].apply(pandas.to_numeric, downcast="float")
    dithers_df[icols] = dithers_df[icols].apply(pandas.to_numeric, downcast="integer")

    dithers_df = dithers_df.set_index(["configuration_id", "positionerId", "fiberType"])

    if os.path.exists(path):
        os.remove(path)

    dithers_df.to_hdf(path, "data")


def radec_radeccat(data: pandas.DataFrame):
    """Plots RA/Dec - RA/Dec from the catalogue."""

    data = data.copy()

    delta_ra = (data.ra - data.racat) * numpy.cos(numpy.deg2rad(data.deccen))
    delta_dec = (data.dec - data.deccat) * numpy.cos(numpy.deg2rad(data.deccen))
    data.loc[:, "delta_cat"] = numpy.sqrt(delta_ra**2 + delta_dec**2) * 3600.0

    _, ax = plt.subplots()

    cmap = cast(Colormap, seaborn.dark_palette("#69d", reverse=True, as_cmap=True))
    hex = ax.hexbin(
        x=data.xFocal,
        y=data.yFocal,
        C=data.delta_cat,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,  # type: ignore
    )
    cmap = plt.colorbar(hex, ax=ax)
    cmap.set_label(r"${\rm RA/Dec - RA/Dec_{\rm cat}} {\rm\ [arcsec]}$")
    ax.set_xlabel(r"$x_{\rm wok}$")
    ax.set_ylabel(r"$y_{\rm wok}$")

    seaborn.despine()

    plt.close("all")

    return ax


def cycle_radec_wok(data: pandas.DataFrame, file1: str = None, file2: str = None):
    """Cycles between RA/Dec and wok coordinates."""

    def apply_radec2wok(g):
        xwok2, ywok2, *_ = radec2wokxy(
            g.ra,
            g.dec,
            None,
            g.fiberType.str.capitalize(),
            g.racen.values[0],
            g.deccen.values[0],
            0.0,
            "APO",
            g.epoch.values[0],
            pmra=None,
            pmdec=None,
            parallax=None,
        )

        return pandas.DataFrame(
            {
                "configuration_id": g.configuration_id,
                "positioner_id": g.positionerId,
                "fibre_type": g.fiberType,
                "epoch": g.epoch,
                "racat": g.racat,
                "deccat": g.deccat,
                "ra": g.ra,
                "dec": g.dec,
                "racen": g.racen,
                "deccen": g.deccen,
                "xwok": g.xFocal,
                "ywok": g.yFocal,
                "xwok2": xwok2,
                "ywok2": ywok2,
            }
        )

    def apply_wok2radec(g):
        ra2, dec2, *_ = wokxy2radec(
            g.xwok2,
            g.ywok2,
            g.fibre_type.str.capitalize(),
            g.racen.values[0],
            g.deccen.values[0],
            0.0,
            "APO",
            g.epoch.values[0],
        )

        g.loc[:, "ra2"] = ra2
        g.loc[:, "dec2"] = dec2

        return g

    data = data.copy()
    data.reset_index(inplace=True)

    # Group by configuration ID
    wok = data.groupby("configuration_id").apply(apply_radec2wok)
    cycled = wok.groupby("configuration_id").apply(apply_wok2radec)

    # Plot xywok vs wxwok2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = seaborn.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

    hex_xwok = axes[0].hexbin(
        x=cycled.xwok,
        y=cycled.ywok,
        C=cycled.xwok - cycled.xwok2,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )
    cmap_xwok = plt.colorbar(hex_xwok, ax=axes[0])
    cmap_xwok.set_label(r"$\Delta x_{\rm wok}\ {\rm [mm]}$")
    axes[0].set_xlabel(r"$x_{\rm wok}$")
    axes[0].set_ylabel(r"$y_{\rm wok}$")

    hex_ywok = axes[1].hexbin(
        x=cycled.xwok,
        y=cycled.ywok,
        C=cycled.ywok - cycled.ywok2,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )
    cmap_ywok = plt.colorbar(hex_ywok, ax=axes[1])
    cmap_ywok.set_label(r"$\Delta y_{\rm wok}\ {\rm [mm]}$")
    axes[1].set_xlabel(r"$x_{\rm wok}$")
    axes[1].set_ylabel(r"$y_{\rm wok}$")

    fig.suptitle("Wok vs radec2xywok")

    seaborn.despine()
    plt.tight_layout()

    fig.savefig(os.path.join(RESULTS, file1 or "cycle_xywok_fvc.pdf"))

    plt.close("all")

    # Plot radec vs radec2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    hex_ra = axes[0].hexbin(
        x=cycled.xwok,
        y=cycled.ywok,
        C=(cycled.racat - cycled.ra2) * numpy.cos(numpy.deg2rad(cycled.deccen)) * 3600,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )
    cmap_ra = plt.colorbar(hex_ra, ax=axes[0])
    cmap_ra.set_label(r"$\Delta\alpha\ {\rm [arcsec]}$")
    axes[0].set_xlabel(r"$x_{\rm wok}$")
    axes[0].set_ylabel(r"$y_{\rm wok}$")

    hex_dec = axes[1].hexbin(
        x=cycled.xwok,
        y=cycled.ywok,
        C=(cycled.deccat - cycled.dec2) * 3600,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )
    cmap_dec = plt.colorbar(hex_dec, ax=axes[1])
    cmap_dec.set_label(r"$\Delta\delta\ {\rm [arcsec]}$")
    axes[1].set_xlabel(r"$x_{\rm wok}$")
    axes[1].set_ylabel(r"$y_{\rm wok}$")

    fig.suptitle("RA/Dec vs radec2xywok+xywok2radec")

    seaborn.despine()
    plt.tight_layout()

    fig.savefig(os.path.join(RESULTS, file2 or "cycle_radec_fvc.pdf"))

    plt.close("all")


def plot_cycle():

    data = pandas.read_hdf(os.path.join(RESULTS, "dither_data_fvc.h5"))

    conf_ids = range(500, max(data.index.get_level_values(0)))
    data = data.loc[data.index.get_level_values(0).isin(conf_ids)]
    data = data.loc[
        (data.assigned == 1)
        & (data.on_target == 1)
        & (data.xFocal > -999)
        & (data.parent_configuration.isna())
    ]

    data_no_fvc = data.loc[data.isFVC == 0].copy()
    ax = radec_radeccat(data_no_fvc)
    ax.figure.suptitle("All confSummary (assigned=1, on_target=1)")
    ax.figure.savefig(os.path.join(RESULTS, "radec_radeccat.pdf"))

    data_fvc = data.loc[data.isFVC == 1].copy()
    ax = radec_radeccat(data_fvc)
    ax.figure.suptitle("All confSummaryF (assigned=1, on_target=1)")
    ax.figure.savefig(os.path.join(RESULTS, "radec_radeccat_fvc.pdf"))

    cycle_radec_wok(data_no_fvc)
    cycle_radec_wok(data_fvc)

    # data_1446 = data.loc[[1446]]
    # data_1446_no_fvc = data_1446.loc[data_1446.isFVC == 0]
    # ax = radec_radeccat(data_1446_no_fvc)
    # ax.figure.suptitle("Configuration 1446")
    # ax.figure.savefig(os.path.join(RESULTS, "radec_radeccat_1446.pdf"))
    # cycle_radec_wok(data_1446_no_fvc, "cycle_xywok_1446.pdf", "cycle_radec_1446.pdf")

    # data_1446_fvc = data_1446.loc[data_1446.isFVC == 1]
    # ax = radec_radeccat(data_1446_fvc)
    # ax.figure.suptitle("Configuration 1446")
    # ax.figure.savefig(os.path.join(RESULTS, "radec_radeccat_1446_fvc.pdf"))
    # cycle_radec_wok(
    #     data_1446_fvc,
    #     "cycle_xywok_1446_fvc.pdf",
    #     "cycle_radec_1446_fvc.pdf",
    # )


def simulate_radec(racen: float = 105.0, deccen: float = 30.0, wave: str = "Apogee"):
    """Generate random RA/Dec positions and check the conversion to wok coordinates."""

    deccen_rad = numpy.deg2rad(deccen)

    ra = numpy.random.uniform(-1.5, 1.5, size=10000)
    dec = numpy.random.uniform(-1.5, 1.5, size=10000)

    valid = numpy.where((ra**2 + dec**2) < 1.5**2)
    ra = (racen + ra[valid] / numpy.cos(deccen_rad)) % 360.0
    dec = deccen + dec[valid]

    xwok, ywok, zwok, *_ = radec2wokxy(
        ra,
        dec,
        2459591.7,
        wave,
        racen,
        deccen,
        0.0,
        "APO",
        2459591.7,
        pmra=1,
        pmdec=1,
        parallax=1,
    )

    ra_cycle, dec_cycle, *_ = wokxy2radec(
        xwok,
        ywok,
        zwok,
        wave,
        racen,
        deccen,
        0.0,
        "APO",
        2459591.7,
    )

    ra_diff = (ra - ra_cycle) * numpy.cos(deccen_rad) * 3600.0
    dec_diff = (dec - dec_cycle) * 3600.0

    seaborn.set_style("white")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    hexra = axes[0].hexbin(
        xwok,
        ywok,
        C=ra_diff,
        gridsize=30,
        reduce_C_function=numpy.median,  # type: ignore
    )

    cmra = plt.colorbar(hexra, ax=axes[0])
    cmra.set_label(r"$\Delta\alpha\ $[arcsec]")

    axes[0].set_xlim(-350, 350)
    axes[0].set_xlabel(r"$x_{\rm wok}$")
    axes[0].set_ylabel(r"$y_{\rm wok}$")

    hexdec = axes[1].hexbin(
        xwok,
        ywok,
        C=dec_diff,
        gridsize=30,
        reduce_C_function=numpy.median,  # type: ignore
    )

    cmdec = plt.colorbar(hexdec, ax=axes[1])
    cmdec.set_label(r"$\Delta\delta\ $[arcsec]")

    axes[1].set_xlim(-350, 350)
    axes[1].set_xlabel(r"$x_{\rm wok}$")
    axes[1].set_ylabel(r"$y_{\rm wok}$")

    fig.savefig(RESULTS / "radec_wok_simulated.pdf")

    # fig, ax = plt.subplots()

    # ax.quiver(ra, dec, ra_cycle, dec_cycle, units="xy", scale=400, width=0.005)

    # plt.show()


def check_fvc_coords():

    data = pandas.read_hdf(RESULTS / "dither_data_fvc.h5")

    # Override xywok from xyFocal
    wokna = data.xwok.isna()

    data.loc[wokna, ("xwok", "ywok")] = data.loc[wokna, ("xFocal", "yFocal")].values

    data = data.loc[data.mjd >= 59594]
    data = data.loc[data.xwok > -999]
    data = data.loc[data.parent_configuration.isna()]
    data = data.loc[data.assigned == 1, :]

    diff = (
        data.loc[data.isFVC == 0, ["xwok", "ywok", "ra", "dec"]]
        - data.loc[data.isFVC == 1, ["xwok", "ywok", "ra", "dec"]]
    )

    print(diff.groupby("fiberType").describe())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = seaborn.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

    hex_xwok = axes[0].hexbin(
        x=data.loc[data.isFVC == 0].xwok,
        y=data.loc[data.isFVC == 0].ywok,
        C=diff.xwok,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )
    cmap_xwok = plt.colorbar(hex_xwok, ax=axes[0])
    cmap_xwok.set_label(r"$\Delta x_{\rm wok}\ {\rm [mm]}$")
    axes[0].set_xlabel(r"$x_{\rm wok}$")
    axes[0].set_ylabel(r"$y_{\rm wok}$")

    hex_ywok = axes[1].hexbin(
        x=data.loc[data.isFVC == 0].xwok,
        y=data.loc[data.isFVC == 0].ywok,
        C=diff.ywok,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )
    cmap_ywok = plt.colorbar(hex_ywok, ax=axes[1])
    cmap_ywok.set_label(r"$\Delta y_{\rm wok}\ {\rm [mm]}$")
    axes[1].set_xlabel(r"$x_{\rm wok}$")
    axes[1].set_ylabel(r"$y_{\rm wok}$")

    fig.suptitle("Wok vs wok difference")

    seaborn.despine()
    plt.tight_layout()

    fig.savefig(RESULTS / "fvc_xywok_diff.pdf")

    plt.close("all")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = seaborn.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

    hex_xwok = axes[0].hexbin(
        x=data.loc[data.isFVC == 0].xwok,
        y=data.loc[data.isFVC == 0].ywok,
        C=diff.ra * numpy.cos(numpy.deg2rad(data.loc[data.isFVC == 0].deccen)) * 3600.0,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )
    cmap_xwok = plt.colorbar(hex_xwok, ax=axes[0])
    cmap_xwok.set_label(r"$\Delta\alpha\ {\rm [arcsec]}$")
    axes[0].set_xlabel(r"$x_{\rm wok}$")
    axes[0].set_ylabel(r"$y_{\rm wok}$")

    hex_ywok = axes[1].hexbin(
        x=data.loc[data.isFVC == 0].xwok,
        y=data.loc[data.isFVC == 0].ywok,
        C=diff.dec * 3600.0,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )
    cmap_ywok = plt.colorbar(hex_ywok, ax=axes[1])
    cmap_ywok.set_label(r"$\Delta\delta\ {\rm [arcsec]}$")
    axes[1].set_xlabel(r"$x_{\rm wok}$")
    axes[1].set_ylabel(r"$y_{\rm wok}$")

    fig.suptitle("Wok vs RA/Dec difference")

    seaborn.despine()
    plt.tight_layout()

    fig.savefig(RESULTS / "fvc_radec_diff.pdf")

    plt.close("all")


if __name__ == "__main__":

    create_dataframe()
    # plot_cycle()

    # simulate_radec()

    # check_fvc_coords()
