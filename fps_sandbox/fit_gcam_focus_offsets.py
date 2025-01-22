#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-08-21
# @Filename: fit_gcam_focus_offsets.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
from multiprocessing import get_context
from os import PathLike

import polars
from astropy.io import fits
from astropy.table import Table
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)


def _get_fwhm_data(proc_gimg: pathlib.Path) -> polars.DataFrame | None:
    """Returns the FWHM data from a ``proc-gimg`` file."""

    header = fits.getheader(proc_gimg, 1)
    centroids = Table.read(proc_gimg, hdu=2)

    if "xfitvalid" not in centroids.colnames:
        return None

    data = polars.DataFrame(centroids.as_array()).filter(
        polars.col.xfitvalid == 1,
        polars.col.yfitvalid == 1,
        polars.col.fwhm_valid == 1,
    )

    df = data.select(
        mjd=polars.lit(header["SJD"], dtype=polars.UInt16),
        camera=polars.lit(header["CAMNAME"][3], dtype=polars.Int8),
        fwhm_mean=polars.col.fwhm.mean().cast(polars.Float32),
        fwhm_median=polars.col.fwhm.median().cast(polars.Float32),
        fwhm_stddev=polars.col.fwhm.std().cast(polars.Float32),
        fwhm_perc10=polars.col.fwhm.quantile(0.1).cast(polars.Float32),
        fwhm_perc25=polars.col.fwhm.quantile(0.25).cast(polars.Float32),
        fwhm_perc75=polars.col.fwhm.quantile(0.75).cast(polars.Float32),
    )

    return df


def collect_fwhm_data(path: PathLike | pathlib.Path) -> polars.DataFrame | None:
    """Collects the FWHM data from a list of ``proc-gimg`` files."""

    files = list(pathlib.Path(path).glob("proc-gimg-*.fits"))

    # Prepare progress bar.
    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        auto_refresh=True,
    )
    task = progress.add_task("[green]Processing ...", total=len(files))

    progress.start()

    fwhm_data: list[polars.DataFrame] = []

    with get_context("spawn").Pool(processes=12) as pool:
        for fwhm_df in pool.imap(_get_fwhm_data, files):
            progress.update(task, advance=1)
            if fwhm_df is None:
                continue
            fwhm_data.append(fwhm_df)

    progress.stop()
    progress.console.clear_live()

    if len(fwhm_data) == 0:
        return None

    df = polars.concat(fwhm_data).sort(["mjd", "camera"])
    return df


if __name__ == "__main__":
    GCAM_DIRS = pathlib.Path("/data/gcam/lco").glob("6*")
    for gcam_dir in GCAM_DIRS:
        fwhm_data = collect_fwhm_data(gcam_dir)
        if fwhm_data is None:
            continue
        fwhm_data.write_parquet(gcam_dir / f"fwhm_{gcam_dir.name}.parquet")
