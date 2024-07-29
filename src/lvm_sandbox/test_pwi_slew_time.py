#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-02-26
# @Filename: test_pwi_slew_time.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import pathlib
import time

import astropy.units as uu
import numpy
import polars
from astropy.coordinates import ICRS, AltAz, EarthLocation
from astropy.time import Time
from gort import Gort


def get_pointing():
    """Returns a random pointing that is above the horizon."""

    observatory = EarthLocation.of_site("Las Campanas Observatory")
    time = Time.now()

    alt = numpy.random.uniform(40, 85) * uu.deg
    az = numpy.random.uniform(10, 350) * uu.deg

    altaz = AltAz(alt=alt, az=az, obstime=time, location=observatory)
    icrs = altaz.transform_to(ICRS())

    return icrs


async def test_pwi_slew_time(
    telescope_name: str,
    slew_time_constant: float = 0.5,
    n_points: int = 10,
    home: bool = True,
    kmirror: bool = False,
    fibsel: bool = False,
    save_to: str | None = None,
):
    """Tests the PWI slew time."""

    gg = await Gort(verbosity="debug").init()

    telescope = gg.telescopes[telescope_name]
    await telescope.initialise()

    if home:
        await telescope.home()

    data: list[tuple] = []

    for nn in range(1, n_points + 1):
        status = await telescope.status()
        current_pointing = (status["ra_j2000_hours"] * 15, status["dec_j2000_degs"])
        current_icrs = ICRS(
            ra=current_pointing[0] * uu.deg,
            dec=current_pointing[1] * uu.deg,
        )

        new_coords = get_pointing()
        ra, dec = (new_coords.ra.deg, new_coords.dec.deg)

        tasks = [telescope.goto_coordinates(ra=ra, dec=dec, pa=0, kmirror=kmirror)]
        if fibsel and telescope.fibsel:
            await telescope.fibsel.move_to_position(16610)
            tasks.append(telescope.fibsel.move_to_position(215))
        else:
            fibsel = False

        time0 = time.time()
        await asyncio.gather(*tasks)
        time1 = time.time()

        separation = current_icrs.separation(new_coords).deg

        data.append(
            (
                nn,
                current_pointing[0],
                current_pointing[1],
                ra,
                dec,
                separation,
                time1 - time0,
                separation / (time1 - time0),
            )
        )

    df = polars.DataFrame(
        data,
        schema=["n", "ra0", "dec0", "ra1", "dec1", "separation", "delay", "rate"],
    )
    df = df.with_columns(
        telescope=polars.lit(telescope_name),
        slew_time_constant=slew_time_constant,
        fibsel=fibsel,
        kmirror=kmirror,
    )

    if save_to:
        df.write_parquet(pathlib.Path(save_to).expanduser())

    return df
