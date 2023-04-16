#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-04-16
# @Filename: guiding_rms.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from cherno.guider import Guider


async def get_ast_solution(images: list[pathlib.Path]):
    """Returns an astrometric solution."""

    guider = Guider("LCO")
    return await guider.process(None, images, write_proc=False, correct=False)
