#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-10-10
# @Filename: gcam_reprocess.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import os
import pathlib
from io import StringIO

import matplotlib.pyplot as plt
import numpy
import pandas
from astropy.io import fits
from tqdm import tqdm

from cherno import config, set_observatory
from cherno.acquisition import Acquisition
from coordio import calibration
from coordio.guide import umeyama


OUTPUT = pathlib.Path(__file__).parent / "../results"


async def gcam_reprocess(
    mjd: int,
    step: int = 1,
    gcam_root: pathlib.Path | str = "/data/gcam",
    observatory: str | None = None,
):
    gcam_root = pathlib.Path(gcam_root)
    path = gcam_root / f"{mjd}"

    procs = path.glob("proc-gimg-*.fits")
    max_frame_no = int(str(list(sorted(procs))[-1]).split("-")[-1].split(".")[0])

    results = []

    observatory = observatory or os.environ["OBSERVATORY"]

    acq = Acquisition(observatory)
    acq.command.log.setLevel(100)  # type: ignore

    for frame_no in tqdm(range(1, max_frame_no + 1, step)):
        files = list(path.glob(f"proc-gimg-*-{frame_no:04d}.*"))

        if len(files) < 3:
            continue

        header = fits.getheader(str(files[0]), 1)

        offra = header.get("OFFRA", 0)
        offdec = header.get("OFFDEC", 0)
        offpa = header.get("OFFPA", 0)
        if offra != 0 or offdec != 0 or offpa != 0:
            continue

        frame_no = int(str(files[0]).split("-")[-1].split(".")[0])

        try:
            astronet = await acq.process(
                None,
                list(files),
                write_proc=False,
                correct=False,
                use_gaia=False,
            )
        except Exception:
            continue

        n_astronet = sum([1 if ad.solved else 0 for ad in astronet.acquisition_data])

        try:
            hybrid = await acq.process(
                None,
                list(files),
                write_proc=False,
                correct=False,
                use_gaia=True,
            )
        except Exception:
            continue

        n_hybrid = sum([1 if ad.solved else 0 for ad in hybrid.acquisition_data])

        try:
            gaia = await acq.process(
                None,
                files,
                write_proc=False,
                correct=False,
                use_astrometry_net=False,
            )
        except Exception:
            continue

        n_gaia = sum([1 if ad.solved else 0 for ad in gaia.acquisition_data])

        result = (
            frame_no,
            header.get("DELTARA", numpy.nan),
            header.get("DELTADEC", numpy.nan),
            header.get("DELTAROT", numpy.nan),
            header.get("DELTASCL", numpy.nan),
            header.get("RMS", numpy.nan),
            astronet.delta_ra,
            astronet.delta_dec,
            astronet.delta_rot,
            astronet.delta_scale,
            astronet.rms,
            n_astronet,
            hybrid.delta_ra,
            hybrid.delta_dec,
            hybrid.delta_rot,
            hybrid.delta_scale,
            hybrid.rms,
            n_hybrid,
            gaia.delta_ra,
            gaia.delta_dec,
            gaia.delta_rot,
            gaia.delta_scale,
            gaia.rms,
            n_gaia,
        )
        results.append(result)

    data = pandas.DataFrame(
        results,
        columns=[
            "frame_no",
            "delta_ra",
            "delta_dec",
            "delta_rot",
            "delta_scale",
            "rms",
            "astronet_delta_ra",
            "astronet_delta_dec",
            "astronet_delta_rot",
            "astronet_delta_scale",
            "astronet_rms",
            "n_astronet",
            "hybrid_delta_ra",
            "hybrid_delta_dec",
            "hybrid_delta_rot",
            "hybrid_delta_scale",
            "hybrid_rms",
            "n_hybrid",
            "gaia_delta_ra",
            "gaia_delta_dec",
            "gaia_delta_rot",
            "gaia_delta_scale",
            "gaia_rms",
            "n_gaia",
        ],
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    data.to_hdf(OUTPUT / f"{mjd}.hdf", "data")


async def get_wok_coordinates(
    observatory: str,
    mjds: int | list[int],
    rms: float = 2.5,
    gcam_root: pathlib.Path | str = "/data/gcam",
    use_astrometry_net: bool = True,
    use_gaia: bool = True,
):
    observatory = observatory.upper()
    gcam_root = pathlib.Path(gcam_root)

    if isinstance(mjds, (int, float)):
        mjds = [int(mjds)]
    else:
        mjds = list(mjds)

    results = []

    set_observatory(observatory)

    acq = Acquisition(observatory)
    acq.command.log.setLevel(100)  # type: ignore

    assert config

    if config["observatory"] != observatory:
        raise ValueError("Observatory not correctly set.")

    config["extraction"]["plot"] = False
    config["acquisition"]["plot_focus"] = False

    data = []
    for mjd in mjds:
        path = gcam_root / str(mjd)

        procs = path.glob("proc-gimg-*.fits")
        max_frame_no = int(str(list(sorted(procs))[-1]).split("-")[-1].split(".")[0])

        for frame_no in tqdm(range(1, max_frame_no + 1)):
            files = list(path.glob(f"proc-gimg-*-{frame_no:04d}.*"))

            if len(files) < 4:
                continue

            header = fits.getheader(str(files[0]), 1)

            offra = header.get("OFFRA", 0)
            offdec = header.get("OFFDEC", 0)
            offpa = header.get("OFFPA", 0)
            if offra != 0 or offdec != 0 or offpa != 0:
                continue

            if header.get("RMS", 999) > rms:
                continue

            frame_no = int(str(files[0]).split("-")[-1].split(".")[0])

            try:
                astrometry = await acq.process(
                    None,
                    list(files),
                    write_proc=False,
                    correct=False,
                    use_astrometry_net=use_astrometry_net,
                    use_gaia=use_gaia,
                    fit_all_detections=True,
                    fit_focus=False,
                )
            except Exception:
                continue

            if astrometry.guider_fit is not None:
                gfa_wok = astrometry.guider_fit.gfa_wok
                astro_wok = astrometry.guider_fit.astro_wok

                result = gfa_wok.join(astro_wok, lsuffix="_gfa", rsuffix="_astro")

                result = result.drop(columns=["gfa_id_astro"])
                result = result.rename(columns={"gfa_id_gfa": "gfa_id"})

                result.loc[:, "region_id"] = result.index.values + 1
                result.loc[:, "frame"] = frame_no
                result.loc[:, "PA"] = round(header.get("FIELDPA", -999.0), 1)
                result.loc[:, "RMS_proc"] = round(header.get("RMS", -999.0), 2)
                result.loc[:, "RMS_recalc"] = round(astrometry.guider_fit.rms, 2)

                results.append(result)

        data_mjd = pandas.concat(results, axis=0)
        data_mjd.loc[:, "mjd"] = mjd

        data.append(data_mjd)

    data_mjds = pandas.concat(data, axis=0)
    data_mjds = data_mjds.reset_index(drop=True)

    return data_mjds


def generate_gfa_coords(
    observatory: str,
    data: str | pathlib.Path | pandas.DataFrame,
    plot: bool = True,
):
    """Generates a new set to gfaCoords.

    To use this function first generate a data frame of wok coordinates using
    ``get_wok_coordinates``. You can generate several days of data and then
    combine them in a single data frame. Make sure you have the fps_calibrations
    that were used to get that data in your $WOK_CALIBS.

    This function will plot the average translation between GFA wok coordinates
    and those derived from astrometric solutions, for each camera. Then, for each
    camera, it will fit translation, rotation, and scale, and will ouput the
    measured translations and rotations, and a new set of GFA coordinates.

    """

    if not isinstance(data, pandas.DataFrame):
        data = pandas.read_csv(str(data))

    # Offset between expected camera centres and astrometric position in wok coordinates
    data.loc[:, "xwok_off"] = data.xwok_astro - data.xwok_gfa
    data.loc[:, "ywok_off"] = data.ywok_astro - data.ywok_gfa

    # Calculate averages.
    xoff = data.groupby(["gfa_id"]).apply(lambda d: float(numpy.mean(d.xwok_off)))
    yoff = data.groupby(["gfa_id"]).apply(lambda d: float(numpy.mean(d.ywok_off)))

    # Quiver plot of offsets.
    if plot:
        x_mean = data.groupby(["gfa_id"])["xwok_astro"].median()
        y_mean = data.groupby(["gfa_id"])["ywok_astro"].median()
        plt.quiver(
            x_mean.values,
            y_mean.values,
            xoff.values,
            yoff.values,
            scale=0.001,
            units="xy",
            angles="xy",
            scale_units="xy",
        )

    gfa_coords = calibration.gfaCoords.loc[observatory].copy()
    new_gfa_coords = gfa_coords.copy()

    if len(gfa_coords) == 0:
        raise ValueError("No current GFA coordinates found.")

    for gfa_id in sorted(data.gfa_id.unique()):
        cam_coords = gfa_coords.loc[gfa_id]
        current_cam_pos = cam_coords.loc[["xWok", "yWok"]].values[numpy.newaxis].T

        data_cam = data.loc[data.gfa_id == gfa_id]

        # Fit translation, rotation, and scale with the cameras centred at zero.
        c, R, t = umeyama(
            data_cam.loc[:, ["xwok_gfa", "ywok_gfa"]].values.T - current_cam_pos,
            data_cam.loc[:, ["xwok_astro", "ywok_astro"]].values.T - current_cam_pos,
        )

        # Modify the current camera centres with the translation we just measured.
        new_gfa_coords.loc[gfa_id, "xWok"] += t[0]  # type: ignore
        new_gfa_coords.loc[gfa_id, "yWok"] += t[1]  # type: ignore

        # The j vector in gfaCoords has jx and jy aligned with wok xy coordinates
        # so it's the easiest to use. We apply the rotation from the fit.
        j = numpy.array([cam_coords.loc["jx"], cam_coords.loc["jy"]])
        new_j = numpy.matmul(R, j)

        # Update the unitary j vector (jz is always 1)
        new_gfa_coords.loc[gfa_id, "jx"] = new_j[0]  # type: ignore
        new_gfa_coords.loc[gfa_id, "jy"] = new_j[1]  # type: ignore

        # Keep the unitary i vector perpendicular.
        new_gfa_coords.loc[gfa_id, "ix"] = new_j[1]  # type: ignore
        new_gfa_coords.loc[gfa_id, "iy"] = -new_j[0]  # type: ignore

        # Calculate the rotation in degrees and output some information.
        rot = numpy.rad2deg(numpy.arctan2(R[0][1], R[0][0]))

        print(f"GFA{gfa_id:.0f}: translation {t}; rotation {rot:.3f}; scale {c:.6f}")

    # Recentre the cameras as a block so that their average in x and y is (0,0)
    new_gfa_coords.xWok -= new_gfa_coords.xWok.mean()
    new_gfa_coords.yWok -= new_gfa_coords.yWok.mean()

    # Some DF gymnastics to get the gfaCoords with the right columns and order.
    new_gfa_coords.reset_index(inplace=True)
    new_gfa_coords["site"] = observatory
    new_gfa_coords.set_index(["site", "id"], inplace=True)
    new_gfa_coords.reset_index(inplace=True)

    # Output the new coordinates in CSV
    io = StringIO()
    new_gfa_coords.to_csv(io, index=True)

    io.seek(0)
    print()
    print(io.read())


if __name__ == "__main__":
    data_mjds = asyncio.run(
        get_wok_coordinates(
            "APO",
            [59873],
            rms=1.0,
            gcam_root="/data/gcam/apo/",
        )
    )

    outpath = OUTPUT / "apo/gcam_reprocess"
    outpath.mkdir(parents=True, exist_ok=True)
    data_mjds.to_csv(outpath / "59873.csv", index=False)

    generate_gfa_coords("APO", data_mjds)
