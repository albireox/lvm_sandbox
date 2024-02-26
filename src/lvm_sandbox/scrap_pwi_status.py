#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-02-22
# @Filename: scrap_pwi_status.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import json
import multiprocessing
import pathlib
import re
from copy import deepcopy

import adbc_driver_postgresql.dbapi as dbapi
import polars as pl
from astropy.coordinates import SkyCoord


LINE_PATTERN = re.compile(
    r"(?P<timestamp>[0-9-]+ [0-9:,]+) - REPLY - (?P<status>{\".*is_slewing.*})"
)


def process_log_file(file_: pathlib.Path):
    """Processes a file and returns a list of status dictionaries."""

    data: list[dict] = []

    with open(file_, "r") as f:
        lines = f.readlines()

        for line in lines:
            if match := LINE_PATTERN.match(line):
                status = json.loads(match.group("status"))
                timestamp = match.group("timestamp")
                status["timestamp"] = timestamp.replace(",", ".")

                status_copy = deepcopy(status)
                for key in status:
                    if isinstance(status[key], dict):
                        for skey in status[key]:
                            status_copy[f"{key}_{skey}"] = status[key][skey]
                        del status_copy[key]

                data.append(status_copy)

    return data


def get_pwi_status_data(path: str | pathlib.Path):
    """Construct a dataframe of PWI status data from log files.."""

    path = pathlib.Path(path)
    files = path.glob("*.log*")

    with multiprocessing.Pool(4) as pool:
        results = pool.map(process_log_file, files)

    data = pl.concat([pl.DataFrame(result) for result in results], how="diagonal")

    data = data.with_columns(timestamp=pl.col("timestamp").str.to_datetime())
    data = data.select(pl.col.timestamp, pl.col("*").exclude("timestamp"))
    data = data.sort("timestamp")

    data = data.with_columns(
        ra_deg=pl.col.ra_j2000_hours * 15,
        dec_deg=pl.col.dec_j2000_degs,
    )

    return data


def create_slew_dataframe(data: pl.DataFrame):
    """Creates a dataframe with on row per slew."""

    # Create a grouping column for slews.
    data = data.with_columns(
        slew_no=pl.col.is_slewing.ne(pl.col.is_slewing.shift()).cum_sum()
    )

    # Keep only slews.
    data = data.filter(pl.col.is_slewing)

    # Reindex slew_no
    data = data.with_columns(
        slew_no=pl.col.slew_no.ne(pl.col.slew_no.shift()).cum_sum() + 1
    )
    data[0, "slew_no"] = 1  # There is a null in the first row.

    # Group by slew_no and get timestamps and pointing.
    data = (
        data.group_by("slew_no")
        .agg(
            t0=pl.col.timestamp.first(),
            t1=pl.col.timestamp.last(),
            ra=pl.col.ra_deg.last(),
            dec=pl.col.dec_deg.last(),
        )
        .sort("slew_no")
    )

    # Calculate the time on target between slews
    data = data.with_columns(time_on_target=pl.col.t0.shift(-1) - pl.col.t1)

    return data


def cross_match_with_standards(data: pl.DataFrame, connection_uri: str):
    """Cross-matches slew coordinates with standard stars."""

    with dbapi.connect(connection_uri) as conn:
        standards = pl.read_database("SELECT * FROM standard;", conn)

    data_skycoords = SkyCoord(data["ra"], data["dec"], unit="deg")
    standards_skycoords = SkyCoord(standards["ra"], standards["dec"], unit="deg")

    idx, d2d, _ = data_skycoords.match_to_catalog_sky(standards_skycoords)

    print(idx, d2d)
    breakpoint()
