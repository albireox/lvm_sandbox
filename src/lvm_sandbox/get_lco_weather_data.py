#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-02-12
# @Filename: get_lco_weather_data.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime
import json
import os

import polars
from influxdb_client.client.influxdb_client import InfluxDBClient
from lvmopstools.weather import get_weather_data
from rich import print
from rich.progress import track


def get_rain_sensor(start_date: str = "2024-06-25"):
    """Gets the rain sensor data."""

    query = f"""
    from(bucket: "actors")
        |> range(start: {start_date}, stop: now())
        |> filter(fn: (r) => r["_measurement"] == "lvmecp")
        |> filter(fn: (r) => r["_field"] == "registers.rain_sensor_alarm")
        |> toInt()
        |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
        |> yield(name: "mean")
    """

    with InfluxDBClient(
        url="http://10.8.38.26:9999",
        token=os.environ["INFLUXDB_V2_TOKEN"],
        org="LVM",
        timeout=600000,
    ) as client:
        api = client.query_api()
        query_results = api.query(query)

    df = polars.DataFrame(
        json.loads(query_results.to_json()),
        schema={"_time": polars.String, "_value": polars.Float32},
    ).select(
        ts=polars.col._time.cast(polars.Datetime("ms", time_zone="UTC")),
        rain=polars.col._value > 0,
    )

    return df


def add_rain_sensor_data(data: polars.DataFrame):
    """Adds rain sensor data to the weather data."""

    print("[gray]Getting rain sensor data.[/gray]")

    data = data.select(polars.all().exclude("rain_intensity"))
    rain_data = get_rain_sensor(
        start_date=data["ts"].min().date().strftime("%Y-%m-%d")  # type: ignore
    ).sort("ts")

    data = data.sort("ts").join_asof(
        rain_data,
        on="ts",
        strategy="nearest",
        tolerance="5m",
    )

    return data.sort("ts")


async def get_lco_weather_data(backdays: int = 360):
    """Gets weather data for the last ``backdays`` days."""

    current_date = datetime.datetime.now()
    start_date = current_date.date()

    dfs: list[polars.DataFrame] = []

    for _ in track(range(backdays), transient=True):
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
                return add_rain_sensor_data(polars.concat(dfs))

    return add_rain_sensor_data(polars.concat(dfs))
