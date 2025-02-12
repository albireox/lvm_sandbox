#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-02-12
# @Filename: get_lco_weather_data.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime

import polars
from lvmopstools.weather import get_weather_data
from rich import print
from rich.progress import track


async def get_lco_weather_data(backdays: int = 360):
    """Gets weather data for the last ``backdays`` days."""

    current_date = datetime.datetime.now()
    start_date = current_date.date()

    dfs: list[polars.DataFrame] = []

    for _ in track(range(backdays)):
        start_date -= datetime.timedelta(days=1)

        # If we ask for 24 hours of data the API only returns a few hours so
        # ask just for one hour at a time.
        for hh in range(0, 24):
            time0 = start_date.strftime("%Y-%m-%d") + f"T{hh:02d}:00:00"
            time1 = start_date.strftime("%Y-%m-%d") + f"T{hh:02d}:59:59"

            try:
                data = await get_weather_data(time0, time1)
                dfs.append(data)
            except Exception:
                print(f"[yellow]Error or end of data reached at {start_date}.[/yellow]")
                return polars.concat(dfs)

    return polars.concat(dfs)
