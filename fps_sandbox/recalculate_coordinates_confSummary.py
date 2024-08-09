#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-08-08
# @Filename: recalculate_coordinates_confSummary.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import numpy
import polars
import seaborn
from astropy.table import Table
from matplotlib import pyplot as plt

from jaeger.target.coordinates import icrs_from_positioner_dataframe
from jaeger.target.tools import read_confSummary
from sdsstools._vendor.yanny import write_ndarray_to_yanny


seaborn.set_theme(style="white", font_scale=1.2)


def recalculate_coordinates_confSummary(path: str | pathlib.Path):
    """Recalculates the coordinates in a confSummary file."""

    path = pathlib.Path(path)
    header, df_orig = read_confSummary(path)

    df_ab = (
        df_orig.select(
            polars.col(
                [
                    "positionerId",
                    "holeId",
                    "fiberType",
                    "lambda_eff",
                    "alpha",
                    "beta",
                ]
            )
        )
        .rename(
            {
                "positionerId": "positioner_id",
                "holeId": "hole_id",
                "fiberType": "fibre_type",
                "lambda_eff": "wavelength",
            }
        )
        .cast({"positioner_id": polars.Int32})
        .with_columns(
            fibre_type=polars.col.fibre_type.str.replace("METROLOGY", "Metrology")
        )
    )

    new_coords = icrs_from_positioner_dataframe(
        df_ab,
        site=header["observatory"],
        boresight=(header["raCen"], header["decCen"]),
        epoch=header["epoch"],
        position_angle=header["pa"],
        focal_plane_scale=header["focal_scale"],
    )

    df_new = df_orig.with_columns(
        ra=new_coords["ra_epoch"],
        dec=new_coords["dec_epoch"],
        xFocal=new_coords["xfocal"],
        yFocal=new_coords["yfocal"],
        ra_observed=new_coords["ra_observed"],
        dec_observed=new_coords["dec_observed"],
        alt_observed=new_coords["alt_observed"],
        az_observed=new_coords["az_observed"],
    )

    new_filename = pathlib.Path(path).with_stem(f"{path.stem}_fix")
    new_filename.unlink(missing_ok=True)

    fibermap = Table.from_pandas(df_new.to_pandas())
    fibermap["mag"] = fibermap["mag"].astype(numpy.dtype(("f4", 5)))
    write_ndarray_to_yanny(
        str(new_filename),
        [fibermap],
        structnames=["FIBERMAP"],
        hdr=header,
        enums={"fiberType": ("FIBERTYPE", ("BOSS", "APOGEE", "METROLOGY", "NONE"))},
    )

    df_orig = df_orig.with_row_count("n")
    df_new = df_new.with_row_count("n")

    do_plots(df_new, df_orig, path)

    # Recalculate the coordinates here
    return df_new, header, df_orig


def do_plots(df_new: polars.DataFrame, df_orig: polars.DataFrame, path: pathlib.Path):
    """Creates QA plots."""

    fig_delta, ax_delta = plt.subplots(1, 2, figsize=(16, 6))

    for ii, df in enumerate([df_orig, df_new]):
        on_target = df.filter(polars.col.on_target == 1)
        dra = on_target["ra"] - on_target["racat"]
        dra *= numpy.cos(numpy.radians(on_target["dec"]))
        ddec = on_target["dec"] - on_target["deccat"]
        dt = (dra**2 + (ddec) ** 2) ** 0.5

        sc = ax_delta[ii].scatter(
            on_target["xwok"],
            on_target["ywok"],
            c=dt * 3600,
            vmax=2,
        )

        ax_delta[ii].set_xlabel("xwok")
        ax_delta[ii].set_ylabel("ywok")

        fig_delta.colorbar(sc, label="Delta [arcsec]", ax=ax_delta[ii])

    fig_delta.savefig(
        path.with_stem(f"{path.stem}_delta").with_suffix(".png"),
        dpi=300,
    )

    fig_quiver, ax_quiver = plt.subplots(1, 1)
    ax_quiver.quiver(
        df_orig["ra"],
        df_orig["dec"],
        (df_new["ra"] - df_orig["ra"]) * numpy.cos(numpy.radians(df_orig["dec"])),
        df_new["dec"] - df_orig["dec"],
        angles="xy",
        scale_units="inches",
        scale=0.002,
    )

    ax_quiver.set_xlabel("RA")
    ax_quiver.set_ylabel("Dec")

    fig_quiver.savefig(
        path.with_stem(f"{path.stem}_quiver").with_suffix(".png"),
        dpi=300,
    )
