#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-02-28
# @Filename: confSummaryF_reprocess.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

# Fixes the bug introduced when we started trimming the FVC images.
# Conor produced a file ptm-sub.csv that recalculates the xyWokMetrology
# values that the FVC camera should have read. Based on that we update the
# confSummaryF files for that period.

from __future__ import annotations

import pathlib
import re
import warnings

import numpy
import pandas
from jaeger.target.tools import positioner_to_wok, wok_to_positioner
from pydl.pydlutils.yanny import yanny

from coordio import ICRS, Field, FocalPlane, Observed, Site, Wok, calibration
from coordio.defaults import INST_TO_WAVE


COMMON_DIR = "/uufs/chpc.utah.edu/common/home/"
SDSSCORE_DIR = COMMON_DIR + "/u0931042/sdss09/sdsscore"
PTM_DIR = COMMON_DIR + "/sdss50/sdsswork/users/u0449727/fvcResize/ptm-sub.csv"

MJD_RANGE = {"APO": (59941, 59997), "LCO": (59936, 59997)}

SITE = "LCO"

ptm = pandas.read_csv(PTM_DIR)


def get_confSummaryFs():
    """Returns a list of affected confSummaryF to reprocess."""

    summary_files = pathlib.Path(SDSSCORE_DIR) / SITE.lower() / "summary_files"
    all_files = summary_files.glob("**/confSummaryF*.par")

    mjd_range = MJD_RANGE[SITE]

    to_fix = []
    for file_ in all_files:
        rr = open(file_, "r").read()
        mjd_match = re.search(r"MJD (\d+)", rr)
        if mjd_match:
            mjd = int(mjd_match.group(1))
            if mjd >= mjd_range[0] and mjd <= mjd_range[1]:
                to_fix.append(file_)

    return to_fix


def get_data(confSummaryF: pathlib.Path):
    """Recovers data needed to recalculate coordinates."""

    rr = open(confSummaryF, "r").read()
    fimg_match = re.search("fvc_image_path (.+)", rr)
    if fimg_match:
        fimg = fimg_match.group(1)

        mm = re.match(r"/data/fcam/(\d+)/.+?(\d+)\.fits", fimg)
        if mm is None:
            return

        mjd = int(mm.group(1))
        fimg_no = int(mm.group(2))

        return mjd, fimg_no


def positioner_to_icrs(
    site: Site,
    hole_id: str,
    fibre_type: str,
    boresight: Observed,
    alpha: float,
    beta: float,
    position_angle: float,
    focal_scale: float,
):
    """Converts from positioner to ICRS coordinates."""

    wavelength = INST_TO_WAVE.get(fibre_type.capitalize(), INST_TO_WAVE["GFA"])

    assert site.time

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)

        wok, _ = positioner_to_wok(
            hole_id,
            site.name,
            fibre_type,
            alpha,
            beta,
        )

        focal = FocalPlane(
            Wok([wok], site=site, obsAngle=position_angle),
            wavelength=wavelength,
            site=site,
            fpScale=focal_scale,
        )

        if boresight is not None:
            field = Field(focal, field_center=boresight)
            obs = Observed(field, site=site, wavelength=wavelength)
            icrs = ICRS(obs, epoch=site.time.jd)
        else:
            field = obs = icrs = None

    row = {
        "hole_id": hole_id,
        "fibre_type": fibre_type.upper(),
        "ra_epoch": icrs[0, 0] if icrs is not None else numpy.nan,
        "dec_epoch": icrs[0, 1] if icrs is not None else numpy.nan,
        "xfocal": focal[0, 0],
        "yfocal": focal[0, 1],
        "xwok": wok[0],
        "ywok": wok[1],
        "zwok": wok[2],
        "alpha": alpha,
        "beta": beta,
    }

    return row


def recalculate_coordinates(confSummaryF: pathlib.Path, ptm_data: pandas.DataFrame):
    """Recalculate coordinates using updated xyWokMetrology."""

    csF = yanny(str(confSummaryF))

    focal_scale = float(csF["focal_scale"])
    epoch = float(csF["epoch"])
    raCen = float(csF["raCen"])
    decCen = float(csF["decCen"])
    position_angle = float(csF["pa"])

    site = Site(SITE)
    site.set_time(epoch)

    pT = calibration.positionerTable.loc[SITE]

    icrs_bore = ICRS([[raCen, decCen]])
    boresight = Observed(icrs_bore, site=site, wavelength=INST_TO_WAVE["GFA"])

    all_rows = []

    for _, row in ptm_data.iterrows():
        pID = row.positionerID
        xWokMeasMetrology = row.xWokMeasMetrology
        yWokMeasMetrology = row.yWokMeasMetrology

        hole_id = pT.loc[pT.positionerID == pID].index[0]

        (alpha, beta), _ = wok_to_positioner(
            hole_id,
            SITE,
            "Metrology",
            xWokMeasMetrology,
            yWokMeasMetrology,
        )

        for fibre_type in ["APOGEE", "BOSS", "Metrology"]:
            row = positioner_to_icrs(
                site,
                hole_id,
                fibre_type,
                boresight,
                alpha,
                beta,
                position_angle,
                focal_scale,
            )
            all_rows.append(row)

            # if fibre_type == "Metrology":
            #     print(xWokMeasMetrology, yWokMeasMetrology, row["xwok"], row["ywok"])

    fixed_data = pandas.DataFrame(
        all_rows,
        columns=[
            "hole_id",
            "fibre_type",
            "ra_epoch",
            "dec_epoch",
            "xfocal",
            "yfocal",
            "xwok",
            "ywok",
            "zwok",
            "alpha",
            "beta",
        ],
    )

    return fixed_data


def update_csF(confSummaryF: pathlib.Path, rec_coords: pandas.DataFrame):
    """Update confSummaryF file with update coordinates."""

    csF = yanny(str(confSummaryF))

    fm = csF["FIBERMAP"]

    for irow in range(len(fm)):
        holeId = fm[irow]["holeId"].decode()
        fiberType = fm[irow]["fiberType"].decode()

        cond = (rec_coords.hole_id == holeId) & (rec_coords.fibre_type == fiberType)
        rec_row = rec_coords.loc[cond]

        fm[irow]["xwok"] = rec_row.xwok
        fm[irow]["ywok"] = rec_row.ywok
        fm[irow]["zwok"] = rec_row.zwok

        fm[irow]["xFocal"] = rec_row.xfocal
        fm[irow]["yFocal"] = rec_row.yfocal

        fm[irow]["ra"] = rec_row.ra_epoch
        fm[irow]["dec"] = rec_row.dec_epoch

        fm[irow]["alpha"] = rec_row.alpha
        fm[irow]["beta"] = rec_row.beta

    confSummaryF.unlink()
    csF.write(str(confSummaryF))


def process_file(confSummaryF: pathlib.Path):
    """Processes a confSummaryF that needs fixing."""

    data = get_data(confSummaryF)

    if data:
        mjd, fimg_no = data
        ptm_data = ptm.loc[
            (ptm.mjd == mjd)
            & (ptm.imgNum == fimg_no)
            & (ptm.site == SITE.lower())
            & (ptm.adjusted)
        ]

        if len(ptm_data) != 500:
            print(mjd, fimg_no, SITE, len(ptm_data))
            print(f"{str(confSummaryF)} failed (number of rows does not match)")
            return

        rec_coords = recalculate_coordinates(confSummaryF, ptm_data)
        update_csF(confSummaryF, rec_coords)

    else:
        print(f"{str(confSummaryF)} failed (data not found)")


if __name__ == "__main__":
    files_to_fix = get_confSummaryFs()
    for file_ in sorted(files_to_fix):
        process_file(file_)
