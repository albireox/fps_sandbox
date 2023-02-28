#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-09-07
# @Filename: boss_calibs.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio

from clu.legacy import TronConnection


async def boss_calibs():

    tron = await TronConnection("LCO.jose_script", "sdss5-hub").start()

    for nbias in range(1, 26):
        print(f"\nTaking bias {nbias}/25 ...")

        cmd = tron.send_command("yao", "expose --bias")
        await cmd
        if cmd.status.did_fail:
            raise RuntimeError("Bias failed!")

        for reply in cmd.replies:
            if "filename" in reply.message and "b2" in reply.message["filename"][0]:
                print(reply.message["filename"][0])

    for ndark in range(1, 4):
        print(f"\nTaking dark {ndark}/3 ...")

        cmd = tron.send_command("yao", "expose --dark 900")
        await cmd
        if cmd.status.did_fail:
            raise RuntimeError("Dark failed!")

        for reply in cmd.replies:
            if "filename" in reply.message and "b2" in reply.message["filename"][0]:
                print(reply.message["filename"][0])


async def off_lamps(tron: TronConnection):
    for lamp in ["HeAr", "Ne"]:
        await tron.send_command("lcolamps", f"off {lamp}")


async def send_command(tron: TronConnection, command: str, turn_off_lamps: bool = True):

    cmd = await tron.send_command(command.split()[0], " ".join(command.split()[1:]))
    if cmd.status.did_fail:
        if turn_off_lamps:
            await off_lamps(tron)
        raise RuntimeError()
    return cmd


async def boss_hartmann():

    EXP_TIME = 10
    N_STEPS = 10
    STEP_SIZE = 75

    tron = await TronConnection("LCO.jose_script", "sdss5-hub").start()

    print("Turning lamps on.")
    for lamp in ["HeAr", "Ne"]:
        await send_command(tron, f"lcolamps on {lamp}")

    print("Opening all doors")
    for door in ["left", "right"]:
        await send_command(tron, f"yao mech open {door}")

    for direction in [1, -1]:

        print()
        for motor in ["a", "b", "c"]:
            print(f"Moving collimator motor {motor} back to 1500.")
            await send_command(
                tron,
                f"yao mech move --absolute 1500 --motor {motor}",
            )

        for step in range(N_STEPS):
            absolute_value = 1500 + direction * STEP_SIZE * step

            print(f"\nMoving collimator to {absolute_value}")
            await send_command(tron, f"yao mech move {direction*STEP_SIZE}")

            print("Taking left hartmann exposure")
            await send_command(tron, "yao mech close left")

            cmd = await send_command(tron, f"yao expose --arc {EXP_TIME}")
            for reply in cmd.replies:
                if "filename" in reply.message and "b2" in reply.message["filename"][0]:
                    print(reply.message["filename"][0])

            print("Taking right hartmann exposure")
            await send_command(tron, "yao mech open left")
            await send_command(tron, "yao mech close right")

            cmd = await send_command(tron, f"yao expose --arc {EXP_TIME}")
            for reply in cmd.replies:
                if "filename" in reply.message and "b2" in reply.message["filename"][0]:
                    print(reply.message["filename"][0])

            await send_command(tron, "yao mech open right")

    await off_lamps(tron)


if __name__ == "__main__":
    # asyncio.run(boss_calibs())
    asyncio.run(boss_hartmann())
