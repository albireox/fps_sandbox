#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-04-21
# @Filename: fix_confSummaryF.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

from pydl.pydlutils.sdss import yanny

from coordio import ICRS, Field, FocalPlane, Observed, Site, Wok
from coordio.defaults import INST_TO_WAVE


CONFIGURATION_IDS = [
    4296,
    4373,
    4413,
    4455,
    4465,
    4532,
    4537,
    4593,
    4613,
    4617,
    4894,
    4908,
    4928,
    4946,
]

OUTPUT = pathlib.Path(os.environ["HOME"]) / "Downloads"
SDSSCORE_DIR = pathlib.Path(os.environ["SDSSCORE_DIR"])


def fix_confSummaryF(configuration_ids: list[int], replace_original: bool = False):
    """Fix confSummaryF files to correctly use the scale factor."""

    for configuration_id in configuration_ids:
        configuration_file = (
            SDSSCORE_DIR
            / "apo/summary_files"
            / f"{int(configuration_id/100):04d}XX"
            / f"confSummaryF-{configuration_id}.par"
        )

        y = yanny(str(configuration_file))

        fibermap = y["FIBERMAP"]

        obs_epoch = float(y["epoch"])

        site = Site("APO")
        site.set_time(obs_epoch)
        assert site.time

        icrs_bore = ICRS([[float(y["raCen"]), float(y["decCen"])]])
        ics_bore = icrs_bore.to_epoch(site.time.jd)
        obs_bore = Observed(ics_bore, site=site, wavelength=INST_TO_WAVE["GFA"])

        scale = float(y["focal_scale"])

        for row in fibermap:

            # Up to wok coordinates everything is AOK.
            xwok = row["xwok"]
            ywok = row["ywok"]
            zwok = row["zwok"]

            ftype = row["fiberType"]

            wavelength = INST_TO_WAVE["Apogee" if ftype == "APOGEE" else "Boss"]

            wok = Wok([[xwok, ywok, zwok]], site=site, obsAngle=float(y["pa"]))
            focal = FocalPlane(wok, wavelength=wavelength, site=site, fpScale=scale)
            field = Field(focal, field_center=obs_bore)
            obs = Observed(field, site=site, wavelength=wavelength)
            icrs = ICRS(obs, epoch=site.time.jd)

            row["ra"] = icrs[0][0]
            row["dec"] = icrs[0][1]

        if replace_original:
            outpath = configuration_file
        else:
            outpath = OUTPUT / configuration_file.name

        outpath.unlink(missing_ok=True)
        y.write(str(outpath))


if __name__ == "__main__":
    fix_confSummaryF(CONFIGURATION_IDS)
