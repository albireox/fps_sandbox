#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-10-07
# @Filename: pos_1353_60955.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

# This file includes functions used to debug the issue with positioner 1353
# failing to initialise on MJD 60955.

from __future__ import annotations

import json
import pathlib

import polars


DATA_DIR = pathlib.Path(__file__).parent / "../../data"


def read_jaeger_positions():
    """Read the jaeger reported positions."""

    FILE = DATA_DIR / "pos_1353_60955/jaeger_status.dat"

    lines = FILE.read_text().splitlines()

    data: list[tuple[int, float, float]] = []
    for line in lines:
        idx = line.find("=")
        if idx != -1:
            line_data = line[idx + 1 :]
            parts = line_data.strip().split(",")
            positioner_id = int(parts[0])
            alpha = float(parts[1])
            beta = float(parts[2])
            data.append((positioner_id, alpha, beta))

    assert len(data) == 500

    df = polars.DataFrame(data, schema=["positioner_id", "alpha", "beta"])
    df = df.sort("positioner_id")

    return df


def compare_positions():
    """Compare expected final positioner positions with jaeger reported positions."""

    EXPECTED_FILE = DATA_DIR / "pos_1353_60955/trajectory-60955-0014.json"

    data_expected = json.loads(EXPECTED_FILE.read_text())

    final_positions: list[tuple[int, float, float]] = []
    for pos in data_expected["final_positions"]:
        positioner_id = int(pos)
        alpha = float(data_expected["final_positions"][pos][0])
        beta = float(data_expected["final_positions"][pos][1])
        final_positions.append((positioner_id, alpha, beta))

    df_expected = polars.DataFrame(
        final_positions,
        schema=["positioner_id", "alpha", "beta"],
    ).sort("positioner_id")
    assert len(df_expected) == 500

    df_reported = read_jaeger_positions()

    df = df_expected.join(df_reported, on="positioner_id", suffix="_reported")

    df = (
        df.with_columns(
            diff_alpha=(polars.col.alpha - polars.col.alpha_reported).abs(),
            diff_beta=(polars.col.beta - polars.col.beta_reported).abs(),
        )
        .with_columns(diff=(polars.col.diff_alpha**2 + polars.col.diff_beta**2).sqrt())
        .sort("diff", descending=True)
    )

    return df
