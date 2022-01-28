#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-01-13
# @Filename: test_fibre_table.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn

from jaeger.target.configuration import AssignmentData
from jaeger.target.design import Design
from jaeger.target.tools import wok_to_positioner


RESULTS = pathlib.Path(__file__).parent / "../results"


def plot_residuals(
    raw_data: pandas.DataFrame,
    fvc_data: pandas.DataFrame,
    deccen: float,
    filename: pathlib.Path,
    fibre: str | None = None,
):

    idx = pandas.IndexSlice
    if fibre:
        raw_data = raw_data.loc[idx[:, fibre], :].copy()
        fvc_data = fvc_data.loc[idx[:, fibre], :].copy()

    deccen_rad = numpy.deg2rad(deccen)

    ra_diff = (raw_data.ra_epoch - fvc_data.ra_epoch) * numpy.cos(deccen_rad) * 3600.0
    dec_diff = (raw_data.dec_epoch - fvc_data.dec_epoch) * 3600.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = seaborn.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

    hexra = axes[0].hexbin(
        raw_data.xwok,
        raw_data.ywok,
        C=ra_diff,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )

    cmra = plt.colorbar(hexra, ax=axes[0])
    cmra.set_label(r"$\Delta\alpha\ $[arcsec]")

    axes[0].set_xlim(-350, 350)
    axes[0].set_xlabel(r"$x_{\rm wok}$")
    axes[0].set_ylabel(r"$y_{\rm wok}$")

    hexdec = axes[1].hexbin(
        raw_data.xwok,
        raw_data.ywok,
        C=dec_diff,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )

    cmdec = plt.colorbar(hexdec, ax=axes[1])
    cmdec.set_label(r"$\Delta\delta\ $[arcsec]")

    axes[1].set_xlim(-350, 350)
    axes[1].set_xlabel(r"$x_{\rm wok}$")
    axes[1].set_ylabel(r"$y_{\rm wok}$")

    seaborn.despine()
    plt.tight_layout()

    fig.savefig(filename)


def simulate_fvc_data(assignment_data: AssignmentData, wok_sigma: float = 0.001):
    """Simulates FVC measured wok coordinates and remeasured data."""

    fibre_table = assignment_data.fibre_table
    fdata = fibre_table.reset_index().set_index(["positioner_id", "fibre_type"]).copy()

    idx = pandas.IndexSlice

    metrology = fdata.loc[idx[:, "Metrology"], :].copy()

    xwok_delta = numpy.random.uniform(-wok_sigma, wok_sigma, len(metrology))
    ywok_delta = numpy.random.uniform(-wok_sigma, wok_sigma, len(metrology))

    metrology.loc[:, "xwok_measured"] = metrology.xwok.values + xwok_delta
    metrology.loc[:, "ywok_measured"] = metrology.ywok.values + ywok_delta

    cols = ["hole_id", "xwok_measured", "ywok_measured"]
    measured = metrology.loc[:, cols].dropna()

    for (pid, ftype), row in measured.iterrows():
        (alpha, beta), _ = wok_to_positioner(
            row.hole_id,
            "APO",
            "Metrology",
            row.xwok_measured,
            row.ywok_measured,
        )
        if not numpy.isnan(alpha):
            fdata.loc[idx[pid, :], ["alpha", "beta"]] = (alpha, beta)

    assignment_data.fibre_table = fdata.copy()

    for pid in measured.index.get_level_values(0).tolist():
        alpha, beta = fdata.loc[(pid, "Metrology"), ["alpha", "beta"]]
        for ftype in ["APOGEE", "BOSS", "Metrology"]:
            assignment_data.positioner_to_icrs(
                pid,
                ftype,
                alpha,
                beta,
                update=True,
            )

    return assignment_data.fibre_table.copy()


def test_fibre_table(design_id: int, epoch: float):

    design = Design(design_id, epoch=epoch)

    raw_data = design.configuration.assignment_data.fibre_table.copy()
    fvc_data = simulate_fvc_data(design.configuration.assignment_data)

    # raw_data.to_hdf(RESULTS / "raw_data.hdf", "data")
    # fvc_data.to_hdf(RESULTS / "fvc_data.hdf", "data")

    plot_residuals(
        raw_data,
        fvc_data,
        design.field.deccen,
        RESULTS / "raw_vs_fvc.pdf",
        fibre="APOGEE",
    )


def plot_wok_difference():

    raw_data = pandas.read_hdf(RESULTS / "raw_data.hdf")
    raw_data_orig = pandas.read_hdf(RESULTS / "raw_data_orig.hdf")

    diff = raw_data.loc[:, ["xwok", "ywok"]] - raw_data_orig.loc[:, ["xwok", "ywok"]]
    print(diff.groupby("fibre_type").describe())

    diff = diff.loc[pandas.IndexSlice[:, ["APOGEE"]], :]
    raw_data_apogee = raw_data.loc[pandas.IndexSlice[:, ["APOGEE"]], :]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = seaborn.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

    hexxwok = axes[0].hexbin(
        raw_data_apogee.xwok,
        raw_data_apogee.ywok,
        C=diff.xwok,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )

    cmra = plt.colorbar(hexxwok, ax=axes[0])
    cmra.set_label(r"$\Delta\ x_{\rm wok}\ $[mm]")

    axes[0].set_xlim(-350, 350)
    axes[0].set_xlabel(r"$x_{\rm wok}$")
    axes[0].set_ylabel(r"$y_{\rm wok}$")

    hexywok = axes[1].hexbin(
        raw_data_apogee.xwok,
        raw_data_apogee.ywok,
        C=diff.ywok,
        gridsize=15,
        cmap=cmap,
        reduce_C_function=numpy.median,
    )

    cmdec = plt.colorbar(hexywok, ax=axes[1])
    cmdec.set_label(r"$\Delta\ y_{\rm wok}\ $[mm]")

    axes[1].set_xlim(-350, 350)
    axes[1].set_xlabel(r"$x_{\rm wok}$")
    axes[1].set_ylabel(r"$y_{\rm wok}$")

    seaborn.despine()
    plt.tight_layout()

    fig.savefig(RESULTS / "wok_diff.pdf")


if __name__ == "__main__":
    # test_fibre_table(35891, 2459592.5733)
    plot_wok_difference()
