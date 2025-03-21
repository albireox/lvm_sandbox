#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-02-26
# @Filename: get_bench_data.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os

import pandas
import polars
from influxdb_client.client.influxdb_client import InfluxDBClient


def get_bench_data():
    """Returns bench temperatures and humidities."""

    query = """
    from(bucket: "actors")
        |> range(start: 2024-06-01, stop: now())
        |> filter(fn: (r) => r["_measurement"] =~ /lvm.(sci|spec|skyw|skye).telemetry/)
        |> filter(fn: (r) => r["_field"] == "sensor1.humidity" or
                             r["_field"] == "sensor2.humidity" or
                            r["_field"] == "sensor2.temperature" or
                            r["_field"] == "sensor1.temperature")
        |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
        |> yield(name: "mean")
    """

    token = os.environ.get("INFLUXDB_V2_TOKEN")
    assert token, "INFLUXDB_V2_TOKEN is not set."

    client = InfluxDBClient("http://10.8.38.26:9999", token=token, timeout=3600)
    api = client.query_api()

    pandas_df = api.query_data_frame(query, org="LVM")
    assert isinstance(pandas_df, pandas.DataFrame)

    data = polars.from_pandas(pandas_df)
    data = (
        data.select(["_time", "_measurement", "_field", "_value"])
        .rename(
            {
                "_time": "time",
                "_measurement": "telescope",
                "_field": "sensor",
                "_value": "value",
            }
        )
        .with_columns(
            telescope=polars.col.telescope.str.extract("lvm.(.+).telemetry"),
            sensor=polars.col.sensor.replace(
                {
                    "sensor1.temperature": "temperature_inside",
                    "sensor2.temperature": "temperature_outside",
                    "sensor1.humidity": "humidity_inside",
                    "sensor2.humidity": "humidity_outside",
                }
            ),
        )
    )

    return data
