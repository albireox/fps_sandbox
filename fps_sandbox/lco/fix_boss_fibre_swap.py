#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-01-21
# @Filename: fix_boss_fibre_swap.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re


def fix_file(file: pathlib.Path) -> None:
    """Fixes the BOSS fibre swap in a confSummary file."""

    data = file.read_text()

    mjd = re.search(r"MJD\s(\d+)", data, re.IGNORECASE | re.MULTILINE)
    if mjd is None:
        return

    mjd = int(mjd.group(1))
    if mjd < 60689 or mjd >= 60697:
        return

    data = re.sub(
        r"(FIBERMAP 1375 R\+7C1 BOSS.+)\s86\s\{(.+)\n",
        r"\1 36 {\2\n",
        data,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    file.write_text(data)


def fix_boss_fibre_swap(path: pathlib.Path | str) -> None:
    """Fixes the BOSS fibre swap in a directory."""

    path = pathlib.Path(path)
    files = sorted(list(path.glob("confSummary*.par")))

    for file in files:
        fix_file(file)
