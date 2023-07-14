#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-07-14
# @Filename: test_gort.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio

from gort import Gort, Tile
from gort.transforms import radec_sexagesimal_to_decimal


async def test_gort():
    g = await Gort(verbosity="debug").init()
    g.log.start_file_logger(
        "/home/gallegoj/software/lvm_sandbox/outputs/gort_ngc6523_12std_2.log",
        mode="w",
        rotating=False,
    )

    await g.telescopes.goto_named_position("zenith")

    ra, dec = radec_sexagesimal_to_decimal("18 03 37.0", "-24 23 12", ra_is_hours=True)

    tile = Tile.from_coordinates(ra=ra, dec=dec)
    # tile.spec_coords = [tile.spec_coords[0]]

    exposure = await g.observe_tile(tile)
    print(exposure)


if __name__ == "__main__":
    asyncio.run(test_gort())
