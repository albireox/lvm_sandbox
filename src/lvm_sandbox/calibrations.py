#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-08
# @Filename: calibrations.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio

from clu.client import AMQPClient


LAMPS = ["Argon", "Xenon", "Neon", "HgNe"]
EXPTIMES = [3, 10, 30, 90, 270]


async def reset(client):
    await (await client.send_command("lvmscp.sp1", "reset"))
    await (await client.send_command("lvmscp.sp3", "reset"))


async def calibrations():
    client = await AMQPClient("calibs-client").start()

    for lamp in LAMPS:
        await (await client.send_command("lvmnps.calib", f"off {lamp}"))

    for lamp in LAMPS:
        print(f"Starting lamp {lamp}")

        wt = 120.0
        if "Hg" in lamp:
            wt = 300
        if "LDLS" in lamp:
            wt = 400

        await (await client.send_command("lvmnps.calib", f"on {lamp}"))
        await asyncio.sleep(wt)

        for et in EXPTIMES:
            print(f"Exposing {lamp} for {et} seconds")
            await reset(client)
            await (
                await client.send_command("trurl", f"spec expose --flavour arc {et}")
            )

        await (await client.send_command("lvmnps.calib", f"off {lamp}"))

    for ii in range(10):
        print(f"Taking bias {ii+1}")
        await reset(client)
        await (await client.send_command("trurl", "spec expose --flavour bias 0.0"))

    for ii in range(10):
        print(f"Taking dark {ii+1}")
        await reset(client)
        await (await client.send_command("trurl", "spec expose --flavour dark 600"))


if __name__ == "__main__":
    asyncio.run(calibrations())
