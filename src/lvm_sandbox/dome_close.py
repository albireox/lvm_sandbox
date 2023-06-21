#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-21
# @Filename: dome_close.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import asyncio

from clu import AMQPClient


async def close_dome():
    """Closes the dome."""

    client = await AMQPClient().start()
    await client.send_command("lvmecp", "dome close")


if __name__ == "__main__":
    asyncio.run(close_dome())
