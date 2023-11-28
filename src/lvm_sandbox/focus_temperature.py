#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-08-14
# @Filename: focus_temperature.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import pandas
import scipy.stats
import seaborn
from influxdb_client import InfluxDBClient
from matplotlib import pyplot as plt


seaborn.set_theme()


TOKEN = "KNuEgMtZSsR-Bavk9ERgPgr7SDjQtcXLRqVEW-HG9q3Ixm1o7XXZTMywIybnmdJCdW61whpQ15jY0whljXRvfw=="  # noqa: E501

OUTPATH = pathlib.Path(__file__).parents[2] / "outputs"


def get_telescope_data(telescope: str, outpath: pathlib.Path):
    """Queries telescope data."""

    client = InfluxDBClient("http://localhost:39999", org="LVM", token=TOKEN)
    api = client.query_api()

    query_focus = f"""
        from(bucket: "actors")
          |> range(start: -1mo)
          |> filter(fn: (r) => r["_measurement"] == "lvm.{telescope}.guider")
          |> filter(fn: (r) => r["_field"] == "frame.focus_position" or r["_field"] == "frame.fwhm")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    """  # noqa: E501

    data_focus = api.query_data_frame(query_focus)
    data_focus = data_focus.drop(
        columns=[
            "table",
            "result",
            "_start",
            "_stop",
            "source",
            "_measurement",
        ]
    )
    data_focus.dropna(inplace=True)

    query_telemetry = f"""
        from(bucket: "actors")
          |> range(start: -1mo)
          |> filter(fn: (r) => r["_measurement"] == "lvm.{telescope}.telemetry")
          |> filter(fn: (r) => r["_field"] == "sensor1.temperature" or r["_field"] == "sensor2.temperature")
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")

    """  # noqa: E501

    data_telemetry = api.query_data_frame(query_telemetry)
    data_telemetry = data_telemetry.drop(
        columns=[
            "table",
            "result",
            "_start",
            "_stop",
            "source",
            "_measurement",
        ]
    )
    data_telemetry.dropna(inplace=True)

    combined = pandas.merge_asof(data_focus, data_telemetry, "_time")
    combined.rename(columns={"_time": "time"}, inplace=True)
    combined.dropna(inplace=True)

    combined = combined.loc[combined["frame.fwhm"] < 3]

    combined.to_hdf(str(outpath), "data")

    return combined


def fit_data(telescope: str, data: pandas.DataFrame):
    """Fits focus-temperature."""

    avg = data.groupby("frame.focus_position", as_index=False).mean()

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    for sensor in [1, 2]:
        if f"sensor{sensor}.temperature" in data:
            slope, intercept, r, p, sterr = scipy.stats.linregress(
                x=data[f"sensor{sensor}.temperature"],
                y=data["frame.focus_position"],
            )

            print(f"{telescope}, sensor{sensor}")
            print(
                f"slope={slope:.3f}, intercept={intercept:.3f}, "
                f"r={r:.3f}, p={p}, sterr={sterr:.6f}"
            )

        if f"sensor{sensor}.temperature" in data:
            seaborn.scatterplot(
                data=data,
                x=f"sensor{sensor}.temperature",
                y="frame.focus_position",
                ax=axes[0],
            )

            seaborn.regplot(
                data=avg,
                x=f"sensor{sensor}.temperature",
                y="frame.focus_position",
                ax=axes[1],
            )

    fig.savefig(str(OUTPATH / f"focus_temperature_{telescope}.pdf"))

    print()


def focus_temperature():
    """Determines the relationship focus-temperature."""

    OUTPATH.mkdir(parents=True, exist_ok=True)

    for telescope in ["sci", "spec", "skyw", "skye"]:
        data_path = OUTPATH / f"focus_temperature_{telescope}.h5"
        if data_path.exists():
            data = pandas.read_hdf(str(data_path))
        else:
            data = get_telescope_data(telescope, data_path)

        assert isinstance(data, pandas.DataFrame)
        fit_data(telescope, data)


if __name__ == "__main__":
    focus_temperature()
