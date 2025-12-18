#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-08-31
# @Filename: analyse_overheads.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime
import pathlib

import polars
import seaborn
from matplotlib import pyplot as plt


DATA_PATH = pathlib.Path(__file__).parents[2] / "data"
DEFAULT_OVERHEADS_FILE = DATA_PATH / "overheads_60996.parquet"

OUTPUTS_PATH = pathlib.Path(__file__).parents[2] / "outputs"


def plot_acquisition_overheads(
    data: polars.DataFrame | str | pathlib.Path | None = None,
    output_path: str | pathlib.Path | None = None,
):
    """Plots acquisition overheads over time."""

    if isinstance(data, (str, pathlib.Path)):
        data = polars.read_parquet(data)
    elif data is None:
        data = polars.read_parquet(DEFAULT_OVERHEADS_FILE)

    data = data.filter(
        polars.col.stage.str.starts_with("slew:")
        | polars.col.stage.str.starts_with("acquisition:"),
    ).with_columns(
        polars.col.dither_position.fill_null(0),
        stage_group=polars.col.stage.str.extract(r"^(slew|acquisition):"),
    )

    acq_data = data.group_by(
        "observer_id",
        "dither_position",
        "tile_id",
    ).agg(
        group_id=polars.col.pk.first(),
        stage_group=polars.col.stage_group.first(),
        start_time=polars.col.start_time.min(),
        duration=polars.col.duration.sum(),
    )

    acq_data = acq_data.group_by("group_id", "stage_group").agg(
        tile_id=polars.col.tile_id.first(),
        dither_position=polars.col.dither_position.first(),
        start_time=polars.col.start_time.min(),
        duration=polars.col.duration.sum(),
    )

    acq_data = acq_data.select(
        [
            "group_id",
            "tile_id",
            "dither_position",
            "start_time",
            "stage_group",
            "duration",
        ]
    )

    total = acq_data.group_by("group_id").agg(
        polars.col(["tile_id", "dither_position", "start_time"]).first(),
        stage_group=polars.lit("total"),
        duration=polars.col.duration.sum(),
    )

    acq_data = (
        polars.concat([acq_data, total])
        .select(
            [
                "group_id",
                "tile_id",
                "dither_position",
                "stage_group",
                "start_time",
                "duration",
            ]
        )
        .sort(["group_id", "start_time"])
    )

    acq_data = acq_data.with_columns(start_time=polars.from_epoch("start_time"))

    acq_data_roll = (
        acq_data.sort("start_time")
        .group_by_dynamic(
            "start_time",
            every="1d",
            closed="both",
            group_by=["stage_group"],
        )
        .agg(duration=polars.col.duration.median())
    )

    acq_data_roll = acq_data_roll.filter(
        polars.col.start_time >= datetime.datetime(2024, 7, 1)
    )

    seaborn.set_theme(
        style="darkgrid",
        palette="deep",
        font_scale=1.2,
        color_codes=True,
    )

    seaborn.lineplot(
        data=acq_data.filter(polars.col.stage_group == "total").to_pandas(),
        x="start_time",
        y="duration",
        hue="stage_group",
        markers=True,
    )

    plt.show()

    return acq_data_roll
