#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-01-22
# @Filename: fix_coordinates.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

import numpy
from pydl.pydlutils.yanny import write_ndarray_to_yanny, yanny

from coordio import ICRS, Field, FocalPlane, Observed, Site, Wok
from coordio.defaults import INST_TO_WAVE


def fix_coordinates():

    mjds = [59600, 59601]

    sdsscore_dir = pathlib.Path(os.environ["SDSSCORE_DIR"]) / "apo/summary_files"
    files = (
        list((sdsscore_dir / "0024XX").glob("*"))
        + list((sdsscore_dir / "0025XX").glob("*"))
        + list((sdsscore_dir / "0026XX").glob("*"))
    )

    outpath = pathlib.Path(__file__).parent / "../results/summary_files"
    outpath.mkdir(exist_ok=True, parents=True)

    for file in files:
        y = yanny(str(file))
        if (int(y["MJD"])) not in mjds:
            continue

        header = y.copy()
        header.pop("FIBERMAP")

        fibermap = y["FIBERMAP"]

        site = Site("APO")
        site.set_time(float(y["epoch"]))

        assert site.time

        boresight_icrs = ICRS([[float(y["raCen"]), float(y["decCen"])]])
        boresight = Observed(
            boresight_icrs,
            site=site,
            wavelength=INST_TO_WAVE["GFA"],
        )

        for ii in range(len(fibermap)):
            if fibermap["ra"][ii] != -999:
                continue

            if fibermap["fiberType"][ii] == "APOGEE":
                wavelength = INST_TO_WAVE["Apogee"]
            elif fibermap["fiberType"][ii] == "BOSS":
                wavelength = INST_TO_WAVE["Boss"]
            else:
                wavelength = INST_TO_WAVE["GFA"]

            wok = numpy.array([fibermap[["xwok", "ywok", "zwok"]][ii].tolist()])

            focal = FocalPlane(
                Wok(wok, site=site, obsAngle=float(y["pa"])),
                wavelength=wavelength,
                site=site,
                fpScale=float(y.get("focal_scale", "1")),
            )

            field = Field(focal, field_center=boresight)
            obs = Observed(field, site=site, wavelength=wavelength)
            icrs = ICRS(obs, epoch=site.time.jd)

            fibermap[ii]["ra"] = icrs[0, 0]
            fibermap[ii]["dec"] = icrs[0, 1]

        outfile = outpath / file.name
        if outfile.exists():
            os.unlink(outfile)

        write_ndarray_to_yanny(
            str(outfile),
            [fibermap],
            structnames=["FIBERMAP"],
            hdr=header,
            enums={"fiberType": ("FIBERTYPE", ("BOSS", "APOGEE", "METROLOGY", "NONE"))},
        )

        print(file.name)


if __name__ == "__main__":
    fix_coordinates()
