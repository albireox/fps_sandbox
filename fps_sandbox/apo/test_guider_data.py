#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-01-10
# @Filename: test_guider_data.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib


def compile_guider_data():

    proc_files = pathlib.Path("/data/gcam/595[6-9][0-9]/").glob("proc-*.fits")

    for proc_file in proc_files:
        print(proc_file)


if __name__ == "__main__":
    compile_guider_data()
