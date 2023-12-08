#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-05-29
# @Filename: check_gimg_data.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re
import warnings
from multiprocessing import Pool

from typing import Any, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from astropy.io.fits import getheader
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS, FITSFixedWarning
from matplotlib.backends.backend_pdf import PdfPages
from rich.progress import Progress

from cherno.utils import gfa_to_wok, umeyama
from coordio import calibration
from coordio.defaults import PLATE_SCALE
from coordio.utils import radec2wokxy


warnings.filterwarnings("ignore", module="astropy.wcs.wcs")
warnings.filterwarnings("ignore", category=FITSFixedWarning)


# matplotlib.use("TkAgg")
matplotlib.use("MacOSX")
seaborn.set_theme()


def fit_one(
    gimg_path: pathlib.Path,
    grid: tuple[int, int] = (10, 10),
    scale_rms: bool = True,
    internal_fit: bool = True,
) -> tuple | None:
    """Fits one camera with internal rotation."""

    name = gimg_path.name
    parent = gimg_path.parent

    proc_path = parent / ("proc-" + name)
    if not proc_path.exists():
        return None

    solved = parent / "astrometry" / name.replace("fits", "solved")
    if not solved.exists():
        return None

    wcs_path = parent / "astrometry" / name.replace("fits", "wcs")
    if not wcs_path.exists():
        return None

    wcs = WCS(open(wcs_path, "r").read())

    header = getheader(str(gimg_path), 1)

    header_proc = getheader(str(proc_path), 1)
    if "DELTARA" not in header_proc:
        return None

    match = re.match(r"gimg-gfa[1-6][ns]-([0-9]{4})\.fits", name)
    if not match:
        return None

    seq_no = int(match.group(1))

    field_ra = header["RAFIELD"]
    field_dec = header["DECFIELD"]
    field_pa = header["FIELDPA"]

    if "AOFFRA" in header_proc:
        offset_ra = header_proc["AOFFRA"]
        offset_dec = header_proc["AOFFDEC"]
        offset_pa = header_proc["AOFFPA"]
    elif "OFFRA" in header_proc:
        offset_ra = header_proc["OFFRA"]
        offset_dec = header_proc["OFFDEC"]
        offset_pa = header_proc["OFFPA"]
    else:
        return None

    obstime = Time(header["DATE-OBS"], format="iso", scale="tai")
    obstime += TimeDelta(header["EXPTIMEN"] / 2.0, format="sec")

    camera_id = int(header["CAMNAME"][3:4])

    observatory = header["OBSERVAT"]

    gfa_data = calibration.gfaCoords.loc[(observatory, camera_id), :]

    xwok_gfa: list[float] = []
    ywok_gfa: list[float] = []
    xwok_astro: list[float] = []
    ywok_astro: list[float] = []

    xidx, yidx = numpy.meshgrid(
        numpy.linspace(0, 2048, grid[0]),
        numpy.linspace(0, 2048, grid[1]),
    )
    xidx = xidx.flatten()
    yidx = yidx.flatten()

    coords: Any = wcs.pixel_to_world(xidx, yidx)
    ra = coords.ra.value
    dec = coords.dec.value

    for x, y in zip(xidx, yidx):
        xw, yw, _ = gfa_to_wok(x, y, camera_id)
        xwok_gfa.append(cast(float, xw))
        ywok_gfa.append(cast(float, yw))

    cos_dec = numpy.cos(numpy.deg2rad(field_dec))
    offset_ra_deg = offset_ra / cos_dec / 3600.0

    _xwok_astro, _ywok_astro, *_ = radec2wokxy(
        ra,
        dec,
        None,
        "GFA",
        field_ra - offset_ra_deg,
        field_dec - offset_dec / 3600.0,
        field_pa - offset_pa / 3600.0,
        observatory,
        obstime.jd,
    )

    xwok_astro += _xwok_astro.tolist()
    ywok_astro += _ywok_astro.tolist()

    X = numpy.array([xwok_gfa, ywok_gfa])
    Y = numpy.array([xwok_astro, ywok_astro])

    if internal_fit:
        X[0, :] -= gfa_data.xWok
        X[1, :] -= gfa_data.yWok

        Y[0, :] -= gfa_data.xWok
        Y[1, :] -= gfa_data.yWok

    try:
        c, R, t = umeyama(X, Y)
    except ValueError:
        return None

    plate_scale = PLATE_SCALE[observatory]  # mm/deg

    # delta_x and delta_y only align with RA/Dec if PA=0. Otherwise we need to
    # project using the PA.
    pa_rad = numpy.deg2rad(field_pa)
    delta_xwok = t[0] * numpy.cos(pa_rad) + t[1] * numpy.sin(pa_rad)
    delta_ywok = -t[0] * numpy.sin(pa_rad) + t[1] * numpy.cos(pa_rad)

    # Convert to arcsec and round up
    delta_ra = numpy.round(delta_xwok / plate_scale * 3600.0, 3)
    delta_dec = numpy.round(delta_ywok / plate_scale * 3600.0, 3)

    delta_rot = numpy.round(-numpy.rad2deg(numpy.arctan2(R[1, 0], R[0, 0])) * 3600.0, 1)
    delta_scale = numpy.round(c, 6)

    if scale_rms:
        xwok_astro /= delta_scale
        ywok_astro /= delta_scale

    delta_x = (numpy.array(xwok_gfa) - numpy.array(xwok_astro)) ** 2  # type: ignore
    delta_y = (numpy.array(ywok_gfa) - numpy.array(ywok_astro)) ** 2  # type: ignore

    xrms = numpy.sqrt(numpy.sum(delta_x) / len(delta_x))
    yrms = numpy.sqrt(numpy.sum(delta_y) / len(delta_y))
    rms = numpy.sqrt(numpy.sum(delta_x + delta_y) / len(delta_x))

    # Convert to arcsec and round up
    xrms = numpy.round(xrms / plate_scale * 3600.0, 3)
    yrms = numpy.round(yrms / plate_scale * 3600.0, 3)
    rms = numpy.round(rms / plate_scale * 3600.0, 3)

    return (
        seq_no,
        camera_id,
        header["CONFIGID"],
        header["DESIGNID"],
        header["FIELDID"],
        field_ra,
        field_dec,
        field_pa,
        delta_ra,
        delta_dec,
        delta_rot,
        delta_scale,
        delta_xwok,
        delta_ywok,
        xrms,
        yrms,
        rms,
        header_proc["DELTARA"],
        header_proc["DELTADEC"],
        header_proc["DELTAROT"],
        header_proc["DELTASCL"],
        header_proc["RMS"],
    )


RESULTS = pathlib.Path(__file__).parents[1] / "results" / "gimg_fits"
DATA = pathlib.Path("/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/gcam/apo")
N_CORES = 16
MJDS = [59714, 59728]


def check_internal_gfa_fit(mjds: list[int]):
    with Progress() as progress:
        for mjd in range(mjds[0], mjds[1] + 1):
            gcam_data = DATA / str(mjd)

            gimg_paths = list(sorted(gcam_data.glob("gimg-gfa*[0-9].fits")))

            fit_data: list[tuple] = []

            task_id = progress.add_task(f"[cyan]{mjd} ...", total=len(gimg_paths))

            with Pool(processes=N_CORES) as pool:
                for data in pool.imap_unordered(fit_one, gimg_paths):
                    progress.advance(task_id)
                    if data is None or data is False:
                        continue
                    fit_data.append(data)

            fit_data = [fd for fd in fit_data if fd]

            df = pandas.DataFrame(
                fit_data,
                columns=[
                    "seq_no",
                    "camera_id",
                    "configid",
                    "designid",
                    "fieldid",
                    "field_ra",
                    "field_dec",
                    "field_pa",
                    "delta_ra",
                    "delta_dec",
                    "delta_rot",
                    "delta_scale",
                    "delta_xwok",
                    "delta_ywok",
                    "xrms",
                    "yrms",
                    "rms",
                    "delta_ra_proc",
                    "delta_dec_proc",
                    "delta_rot_proc",
                    "delta_scale_proc",
                    "rms_proc",
                ],
            )

            df.set_index(["seq_no", "camera_id"], inplace=True)

            RESULTS.mkdir(exist_ok=True)

            outpath = RESULTS / (str(mjd) + ".hdf")
            if outpath.exists():
                outpath.unlink()

            df.to_hdf(outpath, "data")


def plot_fits(MJDS: list[int]):
    with PdfPages(RESULTS / "gimg_mjd_fit.pdf") as pdf:
        for mjd in range(MJDS[0], MJDS[1] + 1):
            data: pandas.DataFrame = pandas.read_hdf(RESULTS / (str(mjd) + ".hdf"))
            data = data.loc[data.rms < 10.0]
            data = data.sort_index()

            axes: Any
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 8))

            for ii, ax in enumerate(axes):
                if ii == 0:
                    y = "delta_ra"
                elif ii == 1:
                    y = "delta_dec"
                else:
                    y = "delta_rot"

                seaborn.lineplot(
                    x="seq_no",
                    y=y,
                    data=data,
                    hue="camera_id",
                    ax=ax,
                    zorder=10,
                    palette="deep",
                )

                seq_no_field = data.reset_index().groupby("fieldid").first().seq_no - 1

                for seq_no in seq_no_field:
                    ax.axvline(
                        x=seq_no, zorder=0, color="k", alpha=0.5, linestyle="dashed"
                    )

                if ii != 0:
                    ax.legend().set_visible(False)

                ax.set_xlim(-75, None)
                ax.set_ylabel(ax.get_ylabel() + " [arcsec]")

            plt.tight_layout(rect=(0, 0, 1, 0.97))
            fig.suptitle(str(mjd))

            pdf.savefig(fig)
            plt.close(fig)


def analyse_fits(MJDS: list[int]):
    dataset: list[pandas.DataFrame] = []
    for mjd in range(MJDS[0], MJDS[1] + 1):
        data: pandas.DataFrame = pandas.read_hdf(RESULTS / (str(mjd) + ".hdf"))
        data = data.loc[data.rms < 10.0]
        data["mjd"] = mjd
        dataset.append(data.sort_index().reset_index())

    data = pandas.concat(dataset)

    avg = data.groupby(["mjd", "camera_id", "fieldid"]).mean()
    avg = avg.loc[:, ["delta_ra", "delta_dec", "delta_rot"]].reset_index()

    for ii, (name, group) in enumerate(avg.groupby("fieldid")):
        avg.loc[avg.fieldid == name, "fieldid_seq"] = ii + 1

    axes: Any
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 8))

    for ii, ax in enumerate(axes):
        if ii == 0:
            y = "delta_ra"
        elif ii == 1:
            y = "delta_dec"
        else:
            y = "delta_rot"

        seaborn.lineplot(
            x="fieldid_seq",
            y=y,
            data=avg,
            hue="camera_id",
            ax=ax,
            zorder=10,
            palette="deep",
            markers=True,
            ci=None,  # type: ignore
            ms=5.0,
            marker="o",
        )

        if ii != 0:
            ax.legend().set_visible(False)

        # ax.set_xlim(-75, None)
        ax.set_ylabel(ax.get_ylabel() + " [arcsec]")

    plt.tight_layout()

    fig.savefig(RESULTS / "gimg_fit.pdf")

    plt.close(fig)


if __name__ == "__main__":
    # check_internal_gfa_fit(MJDS)
    plot_fits(MJDS)
    # analyse_fits(MJDS)
