#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-04-01
# @Filename: confSummary_fix_fibre_swap.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re


SDSSCORE_TEST_DIR = "/Users/gallegoj/Code/sdss5/sdsscore_test/"


def fix_fibre_swap(
    confSummary: pathlib.Path,
    assignments: dict[int, tuple[str, int]],
    min_mjd: int | None = None,
):
    """Fixes the fibre swap in a confSummary file."""

    data = open(confSummary).read()

    mjd_match = re.search(r"MJD\s+([0-9]+)", data, re.MULTILINE)
    if mjd_match is None:
        return

    mjd = int(mjd_match.group(1))
    if min_mjd and mjd < min_mjd:
        return

    for fibre_id, (fibre_type, positioner_id) in assignments.items():
        data = re.sub(
            rf"(FIBERMAP\s+{positioner_id}\s+R[+-].+\s+{fibre_type}\s+.+?)[0-9]+(\s{{.+)",
            r"\g<1>{fibre_id}\g<2>".format(fibre_id=fibre_id),
            data,
        )

    print(confSummary)
    with open(confSummary, "w") as f:
        f.write(data)


def process_files(
    observatory: str,
    min_mjd: int | None = None,
    assignments: dict[int, tuple[str, int]] = {},
):
    """Processes all confSummary files in the directory."""

    obs_path = pathlib.Path(SDSSCORE_TEST_DIR) / observatory.lower() / "summary_files"
    conf_files = sorted(obs_path.glob("**/confSummary*.par"))

    for file in conf_files:
        fix_fibre_swap(file, assignments, min_mjd=min_mjd)


if __name__ == "__main__":
    # ASSIGNMENTS is a mappint of fibre_id to fibre tye and positioner id
    # (this is the final assignment that we want)

    # OBSERVATORY = "APO"
    # MIN_MJD = 59860
    # ASSIGNMENTS = {208: ("BOSS", 898), 209: ("BOSS", 1017)}

    OBSERVATORY = "LCO"
    MIN_MJD = 59860
    ASSIGNMENTS = {26: ("BOSS", 121), 27: ("BOSS", 499)}

    process_files(OBSERVATORY, min_mjd=MIN_MJD, assignments=ASSIGNMENTS)
