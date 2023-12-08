#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-07
# @Filename: read_confsummary.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import pandas

from sdsstools import yanny


__all__ = ["read_confsummary"]


def read_confsummary(path: str | pathlib.Path):
    """Reads a confSummary file and returns the ``FIBERMAP`` table and headers."""

    yy = yanny(str(path))

    fm = yy.pop("FIBERMAP")
    cols = [col for col in fm.dtype.names if col != "mag"]

    df = pandas.DataFrame(fm[cols])
    for col in df.select_dtypes(["object"]):
        df[col] = df[col].str.decode("utf-8")

    return df, dict(yy)
