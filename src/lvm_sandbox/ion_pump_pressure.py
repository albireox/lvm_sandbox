#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-08-10
# @Filename: ion_pump_pressure.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import polars
import seaborn
from matplotlib import pyplot as plt


DATA_DIR = pathlib.Path(__file__).parent / "../../data"
OUTPUTS_DIR = DATA_DIR / "../outputs"


def read_data(path: str | pathlib.Path | None = None) -> polars.DataFrame:
    """Reads ion pump pressure data from a CSV file."""

    if path is None:
        path = DATA_DIR / "ion_pressure.csv"

    df = polars.read_csv(path)

    df = (
        df.select("_time", "ccd", "_value")
        .rename({"_time": "time", "ccd": "camera", "_value": "pressure"})
        .with_columns(polars.col.time.str.to_datetime(time_zone="UTC", time_unit="ms"))
        .with_columns(date=polars.col.time.dt.date())
    )

    return df


def plot_min_pressure():
    """Plots the evolution of the minimum ion pump pressure over time."""

    df = read_data()

    min_pressure = (
        df.group_by("date", "camera")
        .agg(min_pressure=polars.col.pressure.sort().head(100).median())
        .sort("date", "camera")
    )

    seaborn.set_theme(style="whitegrid", font_scale=1.0, palette="deep")

    with plt.ioff():
        fig, ax = plt.subplots(figsize=(20, 8))

        seaborn.lineplot(
            data=min_pressure,
            x="date",
            y="min_pressure",
            hue="camera",
            ax=ax,
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Min. Ion Pump Pressure [torr]")

        fig.savefig(
            OUTPUTS_DIR / "min_ion_pump_pressure.png",
            dpi=300,
            bbox_inches="tight",
        )
