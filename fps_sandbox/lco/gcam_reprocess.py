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

import numpy
import pandas
from astropy.io import fits
from tqdm import tqdm

from cherno import config, set_observatory
from cherno.acquisition import Acquisition


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


if __name__ == "__main__":
    asyncio.run(gcam_reprocess(59864))
