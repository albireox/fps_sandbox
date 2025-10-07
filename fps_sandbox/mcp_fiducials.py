#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-09-01
# @Filename: mcp_fiducials.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime
import pathlib
import re

from typing import Literal

import numpy
import polars
import seaborn
from matplotlib import pyplot as plt


def read_fiducials(file_: str | pathlib.Path, simple: bool = True) -> polars.DataFrame:
    if isinstance(file_, str):
        file_ = pathlib.Path(file_)

    file_data = open(file_).read()

    # Example line
    # ROT_FIDUCIAL 1731506354  -79    -612460   -612460    -981608   -981608    -5.812    -612791   -198878  0 0     # noqa
    matches = re.findall(
        r"^ROT_FIDUCIAL\s+(?P<time>\d+)\s+(?P<fiducial_id>-?\d+)\s+(?P<true1>-?\d+)\s+"
        r"(?P<true2>-?\d+)\s+(?P<pos1>-?\d+)\s+(?P<pos2>-?\d+)\s+(?P<deg>-?\d+.\d+)\s+"
        r"(?P<rot_latch>-?\d+)\s+(?P<velocity>-?\d+)\s+(?P<encoder_error1>\d+)\s+(?P<encoder_error2>\d+)",
        file_data,
        re.MULTILINE,
    )

    df = polars.DataFrame(
        matches,
        schema={
            "time": polars.Int64,
            "fiducial_id": polars.Int16,
            "true1": polars.Int32,
            "true2": polars.Int32,
            "pos1": polars.Int32,
            "pos2": polars.Int32,
            "deg": polars.Float32,
            "rot_latch": polars.Int32,
            "velocity": polars.Int32,
            "encoder_error1": polars.Int32,
            "encoder_error2": polars.Int32,
        },
        orient="row",
    )

    pos_diff = polars.col.pos1 - polars.col.rot_latch
    df = df.with_columns(
        pos_diff=pos_diff,
        h_index=(pos_diff / 800).round().abs().cast(polars.Int32),
        big=polars.col.fiducial_id >= 0,
        ccw=polars.col.deg > 90,
    )

    if simple:
        return df.select(
            "fiducial_id",
            "pos1",
            "rot_latch",
            "deg",
            "pos_diff",
            "h_index",
            "big",
            "ccw",
        )

    return df


def calculate_fiducial_id_orig(latch_pos_diff: int, rot_dir_ccw: bool = True) -> int:
    """Calculate the fiducial index based on the latch positions. Copied from MCP."""

    N_ROT_FIDUCIALS = 160

    fididx = int(abs(round(latch_pos_diff / 800)))
    fididx -= 500

    if fididx > 0:
        big = True
    else:
        big = False
        fididx = -fididx

    fididx += 45

    if fididx > 80 and rot_dir_ccw:
        fididx -= 76
    elif fididx < 48 and not rot_dir_ccw:
        fididx += 76

    if fididx <= 0 or fididx >= N_ROT_FIDUCIALS:
        fididx = -1 - 5
    else:
        fididx = fididx - 5

    if fididx < 0:
        return -1

    if (big and latch_pos_diff >= 0) or (not big and latch_pos_diff <= 0):
        pos_is_mark = 0
    else:
        pos_is_mark = 1

    return fididx if pos_is_mark == 1 else -fididx


def calculate_fiducial_id_new(latch_pos_diff: int, rot_dir_ccw: bool = True) -> int:
    """Recalculate the fiducial index based on the latch positions."""

    N_ROT_FIDUCIALS = 160
    OFFSET = 3

    fididx = int(abs(round(latch_pos_diff / 800)))
    fididx -= 500

    if fididx > 0:
        big = True
    else:
        big = False
        fididx = -fididx

    if fididx < 40 and rot_dir_ccw:
        fididx += 76
    elif fididx > 73 and not rot_dir_ccw:
        fididx -= 76

    fididx += OFFSET

    if fididx <= 0 or fididx >= N_ROT_FIDUCIALS:
        return -1

    if (big and latch_pos_diff >= 0) or (not big and latch_pos_diff <= 0):
        pos_is_mark = 0
    else:
        pos_is_mark = 1

    return fididx if pos_is_mark == 1 else -fididx


def plot_fiducial_table(
    df: polars.DataFrame | str | pathlib.Path,
    outpath: str | pathlib.Path | None = None,
    recalculate_fid_mode: Literal["original", "new"] = "new",
) -> polars.DataFrame:
    """Produces several plots for the fiducial data."""

    file: pathlib.Path | None
    if isinstance(df, (str, pathlib.Path)):
        file = pathlib.Path(df)
        if str(df).endswith(".parquet"):
            df = polars.read_parquet(df)
        else:
            df = read_fiducials(df, simple=True)

    new_fids: list[int] = []
    for row in df.iter_rows(named=True):
        pos_diff = row["pos_diff"]
        ccw = row["ccw"]

        if recalculate_fid_mode == "original":
            new_fids.append(calculate_fiducial_id_orig(pos_diff, ccw))
        elif recalculate_fid_mode == "new":
            new_fids.append(calculate_fiducial_id_new(pos_diff, ccw))
        else:
            raise ValueError(f"Unknown recalculate_fid_mode: {recalculate_fid_mode}")

    df = df.with_columns(fiducial_id_new=polars.Series(new_fids))

    seaborn.set_theme(
        style="darkgrid",
        palette="deep",
        font_scale=1.2,
        color_codes=True,
    )
    plt.ioff()

    fig, axes = plt.subplots(4, 1, figsize=(12, 20))

    seaborn.scatterplot(
        data=df.to_pandas(),
        x="deg",
        y="fiducial_id",
        style="big",
        hue="big",
        ax=axes[0],
    )

    seaborn.scatterplot(
        data=df.to_pandas(),
        x="deg",
        y="fiducial_id_new",
        style="big",
        hue="ccw",
        ax=axes[1],
    )

    seaborn.scatterplot(
        data=df.to_pandas(),
        x="deg",
        y="h_index",
        style="big",
        hue="big",
        ax=axes[2],
    )
    axes[2].set_ylabel("Heidenhain index")
    axes[2].set_ylim(400, 600)

    seaborn.scatterplot(
        data=df.to_pandas(),
        x="deg",
        y="pos1",
        ax=axes[3],
    )

    if file:
        axes[0].set_title(f"Fiducials - {file.name}", fontsize=16)
        plt.savefig(outpath or file.with_suffix(".pdf"), bbox_inches="tight")
    else:
        axes[0].set_title("Fiducials", fontsize=16)
        if outpath is None:
            raise ValueError("Output path must be specified.")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return df


def calculate_scale(data: polars.DataFrame | str | pathlib.Path) -> list[float]:
    """Calculates the scale of the Heidenhain encoder in arcsec per count."""

    if isinstance(data, (str, pathlib.Path)):
        if str(data).endswith(".parquet"):
            data = polars.read_parquet(data)
        else:
            data = read_fiducials(data, simple=True)

    data = data.select("fiducial_id", "deg", "pos1")

    scales: list[float] = []

    for marker in ["big", "small"]:
        data_m = data.filter(
            polars.col.fiducial_id >= 0
            if marker == "big"
            else polars.col.fiducial_id < 0
        )

        diff = (data_m.shift(1) - data_m).with_columns(polars.all().abs()).drop_nulls()
        sigma = 3

        pos1_std = diff["pos1"].std()
        assert isinstance(pos1_std, float)
        diff = diff.filter(
            (polars.col.pos1 - polars.col.pos1.mean()).abs() < sigma * pos1_std
        )

        scale = (diff["deg"] / 800_000 * 3600).mean()
        assert isinstance(scale, float)

        scales.append(scale)

    return scales


def create_fiducials_table(
    file: str | pathlib.Path,
    version: str,
    canonical: int = 41,
    outpath: str | pathlib.Path | None = None,
    skip_fids: list[int] = [],
):
    """Create a rot.dat file from the fiducials mapping data."""

    HEADER = """#
# rotator fiducials
#
# $Name: {version} $
#
# Creator:                 observer
# Time:                    {date}
# Input file:              {file}
# Scales:                  {scale} {scale}
# Canonical fiducial:      {canonical}
#
# Fiducial Encoder1 +-   error  npoint  Encoder2 +-   error  npoint
"""

    file = pathlib.Path(file)
    fids = read_fiducials(file, simple=True)

    scale = numpy.mean(calculate_scale(fids))

    text = HEADER.format(
        version=version,
        date=datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%SZ"),
        file=file.name,
        scale=f"{scale:.18f}",
        canonical=canonical,
    )

    for row in fids.sort("deg").iter_rows(named=True):
        fid = row["fiducial_id"]
        if fid in skip_fids:
            continue

        ref_id = canonical if fid > 0 else -canonical
        ref_data = fids.filter(polars.col.fiducial_id == ref_id)
        ref_deg = 0.0
        pos1_diff = ref_data["pos1"][0] - row["pos1"]
        new_deg = scale / 3600 * pos1_diff

        print(fid, ref_data["deg"][0], new_deg)

    if outpath is None:
        print(text)
