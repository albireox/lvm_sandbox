#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-10-21
# @Filename: ln2_fill_analysis.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime
import json
import pathlib

import httpx
import polars
from rich.progress import track

from sdsstools.logger import get_logger


log = get_logger("lvm_sandbox.ln2_fill_analysis", use_rich_handler=True)
log.set_level(5)
log.sh.setLevel(5)

console = log.rich_console
assert console is not None


def get_db_data(reject_failed: bool = True):
    """Returns LN2 fills DB data."""

    uri = "postgresql://sdss@localhost:39432/lvmdb"
    query = "SELECT * FROM lvmopsdb.ln2_fill"

    data = polars.read_database_uri(query, uri=uri, engine="adbc")

    if reject_failed:
        data = data.filter(~polars.col.aborted, ~polars.col.failed)

    return data


def get_ln2_temp(path: pathlib.Path):
    """Gets the initial LN2 temperature for a fill."""

    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")

    data = polars.read_parquet(path).sort("time")
    return data[0].select(polars.selectors.matches("^temp_[rzb][1-3]_ln2$"))


def get_ambient_temperature(time: datetime.datetime):
    """Gets the ambient temperature at a given time."""

    start_time_iso = time.isoformat()
    end_time_iso = (time + datetime.timedelta(minutes=1)).isoformat()

    uri = f"http://localhost:8085/weather/report?start_time={start_time_iso}&end_time={end_time_iso}"
    with httpx.Client() as client:
        response = client.get(uri)
        if not response.is_success:
            raise ValueError(f"Failed to get weather data: {response.text}")

        data = response.json()

    return data[0]["temperature"]


def collect_fill_data():
    """Collects LN2 fill data."""

    data = get_db_data()

    temperatures: list[polars.DataFrame] = []

    for row in track(list(data.iter_rows(named=True)), console=console):
        data_path = json.loads(row["configuration"])["data_path"]

        ln2_temps = get_ln2_temp(pathlib.Path(pathlib.Path(data_path)))
        ambient_temp = get_ambient_temperature(row["start_time"])
        ln2_temps = ln2_temps.with_columns(
            ambient_temp=polars.lit(ambient_temp, polars.Float32())
        )

        temperatures.append(ln2_temps)

    column_order = temperatures[0].columns
    temperatures = [df.select(column_order) for df in temperatures]

    temp_df = polars.concat(temperatures, how="vertical")
    data = polars.concat([data, temp_df], how="horizontal")

    return data


def _all_therms_active(valve_times: str) -> bool:
    """Returns `True` if all the thermistors are active."""

    valve_data = json.loads(valve_times)
    for value in valve_data.values():
        print(value)
        if "thermistor_first_active" in value:
            if not value["thermistor_first_active"]:
                return False
        elif "timed_out" in value:
            if not value["timed_out"]:
                return False
        else:
            return False

    return True


def ln2_fill_analysis(data: polars.DataFrame | str | None = None):
    """Runs the analysis."""

    if isinstance(data, (str, pathlib.Path)):
        log.info(f"Reading data from {data}.")
        data = polars.read_parquet(data)
    elif data is None:
        log.info("Collecting LN2 fill data.")
        data = collect_fill_data()

    data = data.filter(
        ~polars.col.aborted,
        ~polars.col.failed,
        ~polars.col.fill_start.is_null(),
        ~polars.col.fill_complete.is_null(),
    )
    log.info(f"Total number of successful fills: {len(data)}")

    # Select only fills with all the thermistors active.
    data = data.with_columns(
        all_therms=polars.col.valve_times.map_elements(
            _all_therms_active,
            return_dtype=polars.Boolean(),
        )
    ).filter(polars.col.all_therms)

    return data
