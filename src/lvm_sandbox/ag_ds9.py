#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-03-03
# @Filename: ag_ds9.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio

import pyds9
from clu.client import AMQPClient


async def ag_ds9(actor="lvm.sci.agcam", camera="east", exptime=0.001):
    """Exposes an AG camera continuously and displays the image in DS9."""

    client = await AMQPClient(name="ds9_client", host="10.8.38.21").start()

    ds9 = pyds9.DS9()
    ds9.set("frame delete all")
    ds9.set("frame new")

    while True:
        cmd = await (await client.send_command(actor, f"expose {exptime}"))

        for reply in cmd.replies:
            if camera in reply.message and "filename" in reply.message[camera]:
                filename = reply.message[camera]["filename"]
                nfs_filename = "/Users/gallegoj/nfs/lvm-data/" + filename[5:]

                ds9.set(f"file {nfs_filename}")


if __name__ == "__main__":
    asyncio.run(ag_ds9())
