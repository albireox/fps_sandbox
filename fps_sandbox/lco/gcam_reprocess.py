#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-10-10
# @Filename: gcam_reprocess.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
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


OUTPUT = pathlib.Path(__file__).parent / "../results/lco/gcam_reprocess"


async def gcam_reprocess(mjd: int, step: int = 1):

    path = pathlib.Path(f"/data/gcam/lco/{mjd}")

    procs = path.glob("proc-gimg-*.fits")
    max_frame_no = int(str(list(sorted(procs))[-1]).split("-")[-1].split(".")[0])

    results = []

    acq = Acquisition("LCO")
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
):

    if isinstance(mjds, (int, float)):
        mjds = [int(mjds)]
    else:
        mjds = list(mjds)

    results = []

    set_observatory(observatory)

    acq = Acquisition(observatory.upper())
    acq.command.log.setLevel(100)  # type: ignore

    if config["observatory"] != observatory.upper():
        raise ValueError("Observatory not correctly set.")

    config["extraction"]["plot"] = False
    config["acquisition"]["plot_focus"] = False

    data = []
    for mjd in mjds:

        path = pathlib.Path(f"/data/gcam/lco/{mjd}")

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
                    use_gaia=True,
                    use_astrometry_net=False,
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
                result.loc[:, "RMS"] = round(header.get("RMS", -999.0), 2)

                results.append(result)

        data_mjd = pandas.concat(results, axis=0)
        data_mjd.loc[:, "mjd"] = mjd

        data.append(data_mjd)

    data_mjds = pandas.concat(data, axis=0)
    data_mjds = data_mjds.reset_index(drop=True)

    return data_mjds


def generate_gfa_coords(file_: str | pathlib.Path, plot: bool = True):
    """Generates a new set to gfaCoords."""

    data = pandas.read_csv(str(file_))

    # Offset between expected camera centres and astrometric position in wok coordinates
    data.loc[:, "xwok_off"] = data.xwok_astro - data.xwok_gfa
    data.loc[:, "ywok_off"] = data.ywok_astro - data.ywok_gfa

    # Calculate averages, weighted by RMS.
    xoff = data.groupby(["gfa_id"]).apply(
        lambda d: numpy.average(d.xwok_off, weights=1 / d.RMS)
    )
    yoff = data.groupby(["gfa_id"]).apply(
        lambda d: numpy.average(d.ywok_off, weights=1 / d.RMS)
    )

    # Print offsets in x and ywok coordinates.
    for gfa_id in sorted(data.gfa_id.unique()):
        xoff_gfa = xoff.loc[gfa_id]
        yoff_gfa = yoff.loc[gfa_id]
        print(f"GFA{gfa_id:.0f}: ({xoff_gfa:.2f}, {yoff_gfa:.2f})")

    # Quiver plot of offsets.
    if plot:
        x_mean = data.groupby(["gfa_id"]).xwok_astro.median()
        y_mean = data.groupby(["gfa_id"]).ywok_astro.median()
        plt.quiver(x_mean.values, y_mean.values, xoff.values * 20, yoff.values * 20)

    gfa_coords = calibration.gfaCoords.loc["LCO"].copy()
    if len(gfa_coords) == 0:
        raise ValueError("No current GFA coordinates found.")

    # Correct the GFA positions by the measured offsets.
    gfa_coords.xWok += xoff
    gfa_coords.yWok += yoff

    # Recentre the cameras as a block so that their average in x and y is (0,0)
    gfa_coords.xWok -= gfa_coords.xWok.mean()
    gfa_coords.yWok -= gfa_coords.yWok.mean()

    # Some DF gymnastics to get the gfaCoords with the right columns and order.
    gfa_coords.reset_index(inplace=True)
    gfa_coords["site"] = "LCO"
    gfa_coords.set_index(["site", "id"], inplace=True)
    gfa_coords.reset_index(inplace=True)

    io = StringIO()
    gfa_coords.to_csv(io, index=True)

    io.seek(0)
    print()
    print(io.read())


if __name__ == "__main__":
    asyncio.run(gcam_reprocess(59864))
