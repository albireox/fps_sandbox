#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-07-04
# @Filename: apply_plate_holes_corrections.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import TYPE_CHECKING

import numpy
import polars
import requests
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table

from sdsstools import get_logger, yanny

import fps_sandbox


if TYPE_CHECKING:
    from sdsstools._vendor.yanny import Yanny


def get_plate_holes_sorted(plate_id: int, output_dir: pathlib.Path):
    """Downloads a ``plateHolesSorted`` file from the platelist."""

    BASE_URL = "https://svn.sdss.org/public/data/sdss/platelist/trunk/plates/"

    plate_XX = (f"{plate_id:06d}")[0:4] + "XX"
    ph_name = f"plateHolesSorted-{plate_id:06d}.par"

    url = f"{BASE_URL}{plate_XX}/{plate_id:06d}/{ph_name}"
    with open(output_dir / ph_name, "wb") as f:
        f.write(requests.get(url).content)


def apply_corrections(corr_data: polars.DataFrame, orig_data: Yanny) -> Yanny:
    """Applies the corrections and returns an updated Yanny file."""

    # The correction data is not ordered in the same way as the original data.
    # We do a sky coordinates match and reorder the corr_data dataframe.
    corr_data_coords = SkyCoord(
        ra=corr_data["RA"],
        dec=corr_data["Dec"],
        unit="deg",
    )

    orig_data_science = orig_data["STRUCT1"]
    orig_data_science = orig_data_science[
        numpy.isin(orig_data_science["holetype"], ["BOSS_SHARED", "APOGEE_SHARED"])
    ]

    orig_data_coords = SkyCoord(
        ra=orig_data_science["target_ra"],
        dec=orig_data_science["target_dec"],
        unit="deg",
    )
    assert len(orig_data_coords) == 800, "Invalid number of science targets"

    idx, sep2d, _ = match_coordinates_sky(orig_data_coords, corr_data_coords)
    if any(sep2d.arcsec > 1):
        raise ValueError("Some corrections do not match any original data.")

    # Reorder the corr_data dataframe to match the original data (excluding traps,
    # guide fibres, etc.)
    corr_data_sorted = corr_data[idx]

    # Now iterate over each row in the original data and apply the corrections.
    # The Corrected_values file only modifies certain columns and indicates whether
    # the value has changed.

    corr_data_idx: int = -1

    for orig_data_idx in range(len(orig_data["STRUCT1"])):
        struct1 = orig_data["STRUCT1"]
        row = struct1[orig_data_idx]

        hole_type = row["holetype"]
        if hole_type not in ["BOSS_SHARED", "APOGEE_SHARED"]:
            continue

        corr_data_idx += 1
        corr_row = corr_data_sorted[corr_data_idx]

        corr_original_cid = int(corr_row["Original_CatalogID"][0])
        if struct1["catalogid"][orig_data_idx] != corr_original_cid:
            raise ValueError(f"catalogid mismatch in row {orig_data_idx}.")

        corr_new_cid = int(corr_row["Final_CatalogID"][0])
        struct1["catalogid"][orig_data_idx] = corr_new_cid

        if corr_row["Mag_Change"][0]:
            new_mag = (
                struct1["mag"][orig_data_idx][0],
                corr_row["gmag"][0],
                corr_row["rmag"][0],
                corr_row["imag"][0],
                corr_row["zmag"][0],
            )
            struct1["mag"][orig_data_idx] = new_mag

        if corr_row["APOGEE_Flag_Change"][0]:
            new_apogee_flag = corr_row["APOGEE_Flag"][0]
            struct1["sdssv_apogee_target0"][orig_data_idx] = new_apogee_flag

        if corr_row["BOSS_Flag_Change"][0]:
            new_boss_flag = corr_row["BOSS_Flag"][0]
            struct1["sdssv_boss_target0"][orig_data_idx] = new_boss_flag

        if corr_row["Transformation_Flag_Change"][0]:
            new_gri_gaia_transform = corr_row["Transformation_Flag"][0]
            struct1["gri_gaia_transform"][orig_data_idx] = new_gri_gaia_transform

        if corr_row["Carton_Change"][0]:
            new_firstcarton = corr_row["First_Carton"][0]
            struct1["firstcarton"][orig_data_idx] = new_firstcarton

    return orig_data


def apply_plate_holes_corrections(
    corrections_path: pathlib.Path | str,
    cache_plate_holes_sorted: pathlib.Path | str,
    output_dir: pathlib.Path | str,
    overwrite: bool = False,
):
    """Applies corrections to ``plateHolesSorted`` files.

    Over the course of the 15XXX plate series (SDSS-V plates) a number of mistakes
    were made which now appear in the ``plateHolesSorted`` files. At some point in
    time Felipe generated a series of ``Corrected_values_plateXXX`` files that match
    from catalogid in the ``plateHolesSorted`` files to the new catalogid and
    associated corrected values. Those files have generally lived in the ``idlspec2d``
    product.

    This function applies the corrections and generates a new set of
    ``plateHolesSorted`` files. A new file is generated for each plate ID, even for
    plates that do not have corrections.

    Parameters
    ----------
    corrections_path
        The path to the directory containing the ``Corrected_values_plateXXX`` files.
    cache_plate_holes_sorted
        The path to the directory where the original ``plateHolesSorted`` files are
        stored. If the files do not exist, they will be downloaded.
    output_dir
        The path to the directory where the corrected ``plateHolesSorted`` files will
        be saved.
    overwrite
        Whether to overwrite the corrected ``plateHolesSorted`` if already exists.

    """

    log = get_logger("fps_sandbox.apply_plate_holes_corrections")
    log.set_level(5)
    log.sh.setLevel(5)

    corrections_path = pathlib.Path(corrections_path)
    cache_plate_holes_sorted = pathlib.Path(cache_plate_holes_sorted)
    output_dir = pathlib.Path(output_dir)

    assert corrections_path.exists(), "corrections_path does not exist."

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    if not cache_plate_holes_sorted.exists():
        cache_plate_holes_sorted.mkdir(parents=True)

    PLATE_RANGE = range(15000, 15420)
    for plate_id in PLATE_RANGE:
        ph_name = f"plateHolesSorted-{plate_id:06d}.par"

        log.info(f"Processing {ph_name}.")

        output_phsorted = output_dir / ph_name
        if output_phsorted.exists():
            if not overwrite:
                log.debug(f"{output_phsorted.name} already exists. Skipping.")
                continue
            else:
                output_phsorted.unlink()

        ph_original = cache_plate_holes_sorted / ph_name
        if not ph_original.exists():
            log.debug(f"Downloading {ph_name} from platelist.")
            get_plate_holes_sorted(plate_id, output_dir=cache_plate_holes_sorted)
        else:
            log.debug(f"Using cached {ph_name}.")

        corr_file_glob = list(corrections_path.glob(f"*_plate{plate_id}_design*.fits"))
        if len(corr_file_glob) != 1:
            log.error(f"Cannot find Corrected_values file for plate {plate_id}.")
            continue

        corr_file = corr_file_glob[0]
        corr_data_table = Table.read(corrections_path / corr_file)
        corr_data_table.convert_bytestring_to_unicode()
        corr_data = polars.from_pandas(corr_data_table.to_pandas())

        ph_orig_data = yanny(str(ph_original))

        new_plate_holes_yn = apply_corrections(corr_data, ph_orig_data)
        new_plate_holes_yn.write(str(output_phsorted))
        log.debug("Wrote corrected plateHolesSorted file.")


if __name__ == "__main__":
    FPS_SANDBOX_ROOT = pathlib.Path(fps_sandbox.__file__).parents[1]

    PLATE_FIX_FILES_PATH = FPS_SANDBOX_ROOT / "data" / "plate_fix_files"
    CORRECTED_VALUES_PATH = PLATE_FIX_FILES_PATH / "corrected_values"
    CACHE_PLATE_HOLES_SORTED_PATH = PLATE_FIX_FILES_PATH / "original_plate_holes_sorted"
    OUTPUT_PATH = PLATE_FIX_FILES_PATH / "corrected_plate_holes"

    apply_plate_holes_corrections(
        corrections_path=CORRECTED_VALUES_PATH,
        cache_plate_holes_sorted=CACHE_PLATE_HOLES_SORTED_PATH,
        output_dir=OUTPUT_PATH,
    )
