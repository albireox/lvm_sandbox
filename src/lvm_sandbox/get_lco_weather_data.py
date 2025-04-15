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
import pathlib

import httpx
import polars
import seaborn
from influxdb_client.client.influxdb_client import InfluxDBClient
from lvmopstools.weather import get_weather_data, is_weather_data_safe
from matplotlib import pyplot as plt
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


def get_weather_data_from_api(
    start_time: str,
    end_time: str,
    api_url: str = "http://localhost:8085",
):
    """Returns the weather data from the LVM API."""

    with httpx.Client(base_url=api_url, follow_redirects=True) as client:
        response = client.get(
            "/weather",
            params={
                "start_time": start_time,
                "end_time": end_time,
                "station": "DuPont",
            },
        )

        response.raise_for_status()

        data = (
            polars.DataFrame(response.json())
            .with_columns(
                ts=(polars.col.ts + "Z").str.to_datetime(
                    format="%Y-%m-%dT%H:%M:%SZ",
                    time_zone="UTC",
                    time_unit="ms",
                ),
            )
            .with_columns(polars.selectors.numeric().cast(polars.Float32))
        )

    data = data.with_columns(
        wind_average=polars.col("wind_speed_avg")
        .rolling_mean_by(
            by="ts",
            window_size="10m",
        )
        .cast(polars.Float32),
    )

    wind_safe: list[None | bool] = [None]

    for ii in range(1, len(data)):
        wind_safe.append(
            is_weather_data_safe(
                data[0:ii],
                "wind_speed_avg",
                threshold=35,
                reopen_value=30,
                now=data["ts"][ii],
            )
        )

    data = data.with_columns(
        polars.Series(
            "wind_safe",
            wind_safe,
        ).cast(polars.Boolean),
    )

    return data


def plot_weather_data(
    data: polars.DataFrame,
    outfile: str | pathlib.Path | None = None,
):
    """Plots the weather data."""

    OUTPATH = pathlib.Path(__file__).parent / "../../outputs/weather_plot"
    OUTPATH.mkdir(exist_ok=True, parents=True)

    seaborn.set_theme()

    if outfile is not None:
        plt.ioff()
    else:
        plt.ion()

    fig, axes = plt.subplots(3, 1, figsize=(18, 12))

    axes[0].plot(
        data["ts"],
        data["wind_speed_avg"],
        color="b",
        alpha=0.5,
        label="Wind speed avg.",
        zorder=10,
    )
    axes[0].plot(
        data["ts"],
        data["wind_speed_avg_5m"],
        color="r",
        label="Wind speed avg. (5m)",
        zorder=15,
    )
    axes[0].legend(loc="upper left")
    axes[0].set_ylabel("Wind speed [mph]")

    axes[1].plot(
        data["ts"],
        data["relative_humidity"],
        color="b",
        label="Relative humidity",
        zorder=10,
    )

    if "rain" in data.columns:
        rain = data.filter(polars.col.rain)
        if len(rain) > 5:
            ylim1 = axes[1].get_ylim()
            axes[1].fill_between(
                data["ts"],
                ylim1[0],
                ylim1[1],
                where=(data["rain"]),
                color="r",
                lw=0,
                alpha=0.3,
                zorder=5,
                label="Rain",
            )
            axes[1].set_ylim(ylim1)

    axes[1].legend(loc="upper left")
    axes[1].set_ylabel("Relative humidity [%]")

    axes[2].plot(
        data["ts"],
        data["temperature"] - data["dew_point"],
        color="b",
        label="Temperature - dew point",
    )

    # Plot line at 0 delta temperature but preserve y limits.
    ylim2 = axes[2].get_ylim()
    axes[2].axhline(0, color="r", linestyle="--")
    axes[2].set_ylim(ylim2)

    axes[2].legend(loc="upper left")
    axes[2].set_ylabel("Temperature - dew point [degC]")

    dates = data["ts"].dt.date().unique()
    date = dates[0]

    axes[0].set_title(f"Weather data — {date.strftime('%Y-%m-%d')}", fontsize=16)

    fig.tight_layout()

    if outfile is not None:
        outfile = OUTPATH / outfile
        fig.savefig(outfile)
    else:
        plt.show(block=True)

    plt.close("all")

    return fig, axes


def plot_weather_dates(data: polars.DataFrame):
    """Plots the weather data."""

    OUTPATH = pathlib.Path(__file__).parent / "../../outputs/weather_plot"
    OUTPATH.mkdir(exist_ok=True, parents=True)

    seaborn.set_theme()
    plt.ioff()

    data = data.sort(polars.col.ts)
    dates = data["ts"].dt.date().unique()

    for date in dates:
        data_ts = data.filter(polars.col.ts.dt.date() == date)
        plot_weather_data(data_ts, outfile=f"weather_{date.strftime('%Y-%m-%d')}.pdf")
