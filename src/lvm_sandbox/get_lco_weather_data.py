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


async def get_lco_weather_data(backdays: int = 360):
    """Gets weather data for the last ``backdays`` days."""

    current_date = datetime.datetime.now()
    start_date = current_date.date()

    dfs: list[polars.DataFrame] = []

    for ii in range(backdays):
        time0 = start_date.strftime("%Y-%m-%dT00:00:00")
        time1 = start_date.strftime("%Y-%m-%dT23:59:59")

        try:
            data = await get_weather_data(time0, time1)
            dfs.append(data)
        except Exception:
            print(f"[yellow]End of data reached at {start_date}.[/yellow]")
            break

        start_date -= datetime.timedelta(days=1)

    return polars.concat(dfs)
