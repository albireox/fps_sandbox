#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-07-29
# @Filename: check_buses.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio
import os
import sys

from fps_sandbox.osu.check_layout import prepare_layout_data
from jaeger import FPS


async def check_buses(data):

    fps = await FPS.create()

    n_bus = {ii: [0, 0, 0, 0] for ii in range(1, 7)}
    for pid in fps.positioners:
        iface, bus = fps[pid].get_bus()
        iface += 1

        n_bus[iface][bus - 1] += 1

    for sextant in sorted(n_bus):
        for ibus, bus in enumerate(n_bus[sextant]):
            print(f"Sextant {sextant}, bus {ibus+1}: {n_bus[sextant][ibus]}.")

    for pid in fps.positioners:
        p_data = data.loc[pid]
        iface, bus = fps[pid].get_bus()

        if p_data.Sextant != iface + 1:
            print(f"Positioner {pid}. Mismatch layout (sextant, bus) "
                  f"({p_data.Sextant}, {p_data.CAN}) with measured "
                  f"({iface+1}, {bus}).")
        elif p_data.CAN != bus:
            print(f"Positioner {pid}. Mismatch layout (sextant, bus) "
                  f"({p_data.Sextant}, {p_data.CAN}) with measured "
                  f"({iface+1}, {bus}).")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = os.path.join(
            os.path.dirname(__file__),
            "../data/SloanFPS_HexArray_2021July23.csv",
        )

    asyncio.run(check_buses(prepare_layout_data(data_file)))
