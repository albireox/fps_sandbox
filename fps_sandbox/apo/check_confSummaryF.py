#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-04-18
# @Filename: check_confSummaryF.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pandas
import pandas.plotting
import seaborn
from astropy.time import Time
from pydl.pydlutils.sdss import yanny

from coordio import ICRS, Field, FocalPlane, Observed, Site, Wok
from coordio.defaults import INST_TO_WAVE
from jaeger.target.tools import positioner_to_wok, wok_to_positioner


matplotlib.use("MacOSX")
seaborn.set_theme()

RESULTS = pathlib.Path(os.path.dirname(__file__)) / "../results"
SDSSCORE_DIR = pathlib.Path(os.environ["SDSSCORE_DIR"])


def read_confSummary(path: str | pathlib.Path, return_yanny: bool = False) -> tuple:
    y = yanny(str(path))
    header = dict(y)

    fibermap = header.pop("FIBERMAP")
    fibermap = fibermap[[col for col in fibermap.dtype.names if col != "mag"]]

    df = pandas.DataFrame(fibermap)

    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.decode("utf-8")

    for key, value in header.items():
        try:
            header[key] = int(value)
        except ValueError:
            try:
                header[key] = float(value)
            except ValueError:
                pass

    if return_yanny:
        return header, df.set_index(["positionerId", "fiberType"]), y

    return header, df.set_index(["positionerId", "fiberType"])


def check_confSummaryF():
    path = SDSSCORE_DIR / "apo" / "summary_files" / "0048XX"
    file_ = path / "confSummary-4888.par"
    fileF_ = path / "confSummaryF-4888.par"

    fm = yanny(str(file_))["FIBERMAP"]
    fmF = yanny(str(fileF_))["FIBERMAP"]

    cs = pandas.DataFrame(fm[[col for col in fm.dtype.names if col != "mag"]])
    cs.holeId = cs.holeId.str.decode("utf-8")
    cs.fiberType = cs.fiberType.str.decode("utf-8")

    csF = pandas.DataFrame(fmF[[col for col in fmF.dtype.names if col != "mag"]])
    csF.holeId = csF.holeId.str.decode("utf-8")
    csF.fiberType = csF.fiberType.str.decode("utf-8")

    met = cs.loc[cs.fiberType == "METROLOGY"]
    metF = csF.loc[csF.fiberType == "METROLOGY"]

    wavelength = INST_TO_WAVE["GFA"]
    position_angle = 16.263046

    site = Site("APO")
    site.set_time(2459683.683394076)
    assert site.time

    icrs_bore = ICRS([[193.703227069447, 83.8629516188865]])
    boresight = Observed(
        icrs_bore,
        site=site,
        wavelength=INST_TO_WAVE["GFA"],
    )

    scale = 0.999641012363888

    recalculated = []
    for _, row in metF.iterrows():
        wok, _ = positioner_to_wok(
            row.holeId,
            "APO",
            "Metrology",
            row.alpha,
            row.beta,
        )

        focal = FocalPlane(
            Wok([wok], site=site, obsAngle=position_angle),
            wavelength=wavelength,
            site=site,
            fpScale=scale,
        )

        field = Field(focal, field_center=boresight)
        obs = Observed(field, site=site, wavelength=wavelength)
        icrs = ICRS(obs, epoch=site.time.jd)[0]

        recalculated.append((row.positionerId, wok[0], wok[1], icrs[0], icrs[1]))

    recalculated = pandas.DataFrame(
        recalculated,
        columns=["positionerId", "xwok", "ywok", "ra", "dec"],
    )
    recalculated.set_index("positionerId", inplace=True)

    met.set_index("positionerId", inplace=True)
    metF.set_index("positionerId", inplace=True)

    ra_diff = (met.ra - recalculated.ra) * numpy.cos(numpy.deg2rad(83.8629516188865))
    dec_diff = met.dec - recalculated.dec

    dist = numpy.sqrt(ra_diff**2 + dec_diff**2) * 3600
    dist = dist.loc[dist < 2]

    pandas.plotting.hist_series(dist, bins=50)
    plt.show()


def check_coordinates():
    OUTPUT_DIR = RESULTS / "fvc_coordinates"
    OUTPUT_DIR.mkdir(exist_ok=True)

    path = SDSSCORE_DIR / "apo" / "summary_files" / "0048XX"
    file_ = path / "confSummary-4889.par"

    header, cs = read_confSummary(file_)

    cos_dec = numpy.cos(numpy.deg2rad(header["decCen"]))

    assigned: pandas.DataFrame = cs.loc[
        (cs.assigned == 1) & (cs.on_target == 1) & (cs.valid == 1)
    ]
    plt.scatter(
        (assigned.ra - assigned.racat)
        * numpy.cos(numpy.deg2rad(header["decCen"]))
        * 3600,
        3600 * (assigned.dec - assigned.deccat),
    )

    recalc = []

    obs_epoch = header["epoch"]

    site = Site("APO")
    site.set_time(obs_epoch)
    assert site.time

    icrs_bore = ICRS([[header["raCen"], header["decCen"]]])
    ics_bore = icrs_bore.to_epoch(site.time.jd)
    obs_bore = Observed(ics_bore, site=site, wavelength=INST_TO_WAVE["GFA"])

    for idx, row in assigned.iterrows():
        positionerId, fiberType = idx  # type: ignore

        pmra = row.pmra if row.pmra > -999.0 else 0
        pmdec = row.pmdec if row.pmdec > -999.0 else 0
        parallax = row.parallax if row.parallax > -999.0 else 0
        epoch = row.coord_epoch

        icrs = ICRS(
            [[row.racat, row.deccat]],
            pmra=numpy.nan_to_num(pmra, nan=0),
            pmdec=numpy.nan_to_num(pmdec, nan=0),
            parallax=numpy.nan_to_num(parallax),
            epoch=Time(epoch, format="jyear").jd,
        )
        icrs = icrs.to_epoch(site.time.jd, site=site)

        wavelength = INST_TO_WAVE["Apogee" if fiberType == "APOGEE" else "Boss"]

        observed = Observed(icrs, wavelength=wavelength, site=site)
        field = Field(observed, field_center=obs_bore)
        focal = FocalPlane(
            field,
            wavelength=wavelength,
            site=site,
            fpScale=header["focal_scale"],
        )
        wok = Wok(focal, site=site, obsAngle=header["pa"])

        positioner, _ = wok_to_positioner(
            row.holeId,
            site.name,
            fiberType,
            wok[0][0],
            wok[0][1],
            wok[0][2],
        )

        recalc.append(
            (
                positionerId,
                fiberType,
                icrs[0][0],
                icrs[0][1],
                wok[0][0],
                wok[0][1],
                positioner[0],
                positioner[1],
            )
        )

    recalc = pandas.DataFrame(
        recalc,
        columns=[
            "positionerId",
            "fiberType",
            "ra",
            "dec",
            "xwok",
            "ywok",
            "alpha",
            "beta",
        ],
    )
    recalc.set_index(["positionerId", "fiberType"], inplace=True)

    radec_dist = recalc.loc[:, ["ra", "dec"]] - assigned.loc[:, ["ra", "dec"]]
    radec_dist = 3600 * numpy.sqrt((radec_dist.ra * cos_dec) ** 2 + radec_dist.dec**2)

    ax_radec = pandas.plotting.hist_series(radec_dist, bins=20)
    ax_radec.set_xlabel("Distance [arcsec]")
    ax_radec.figure.savefig(OUTPUT_DIR / "down_radec_all.pdf")
    plt.close("all")

    wok_dist = recalc.loc[:, ["xwok", "ywok"]] - assigned.loc[:, ["xwok", "ywok"]]
    wok_dist = 1000 * numpy.sqrt(wok_dist.xwok**2 + wok_dist.ywok**2)

    ax_wok = pandas.plotting.hist_series(wok_dist, bins=20)
    ax_wok.set_xlabel("Distance [microns]")
    ax_wok.figure.savefig(OUTPUT_DIR / "down_wok_all.pdf")
    plt.close("all")

    # wok_apogee = wok_dist.loc[pandas.IndexSlice[:, "APOGEE"]]
    # ax_wok = pandas.plotting.hist_series(wok_apogee.loc[wok_apogee < 100], bins=20)
    # ax_wok.set_xlabel("Distance [microns]")
    # ax_wok.figure.savefig(OUTPUT_DIR / "down_wok_apogee.pdf")
    # plt.close("all")

    # wok_boss = wok_dist.loc[pandas.IndexSlice[:, "BOSS"]]
    # ax_wok = pandas.plotting.hist_series(wok_boss.loc[wok_boss < 100], bins=20)
    # ax_wok.set_xlabel("Distance [microns]")
    # ax_wok.figure.savefig(OUTPUT_DIR / "down_wok_boss.pdf")
    # plt.close("all")

    _, ax_wok_scatter = plt.subplots(1)
    ax_wok_scatter.quiver(
        assigned.xwok,
        assigned.ywok,
        recalc.xwok - assigned.xwok,
        recalc.ywok - assigned.ywok,
        wok_dist,
        cmap="Reds",
        angles="xy",
        units="xy",
    )
    ax_wok_scatter.figure.savefig(OUTPUT_DIR / "down_wok_scatter.pdf")

    _, ax_radec_radeccat = plt.subplots(1)
    ax_radec_radeccat.scatter(
        (assigned.ra - assigned.racat)
        * numpy.cos(numpy.deg2rad(header["decCen"]))
        * 3600,
        3600 * (assigned.dec - assigned.deccat),
    )
    ax_radec_radeccat.figure.savefig(OUTPUT_DIR / "down_radec_scatter.pdf")

    # Now up.
    # recalc_up = recalc.copy()

    # for idx, row in assigned.iterrows():

    #     positionerId, fiberType = idx  # type: ignore

    #     for ftype in ["APOGEE", "BOSS", "Metrology"]:
    #         if ftype == fiberType:
    #             continue

    #         icrs_data = self.positioner_to_icrs(
    #             pid,
    #             ftype,
    #             positioner_data["alpha"],
    #             positioner_data["beta"],
    #             position_angle=self.position_angle,
    #             update=False,
    #         )
    #         data[(pid, ftype)] = icrs_data
    breakpoint()


if __name__ == "__main__":
    check_confSummaryF()
    # check_coordinates()
