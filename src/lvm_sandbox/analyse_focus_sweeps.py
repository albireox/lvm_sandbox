#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-01-26
# @Filename: analyse_focus_sweeps.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import pandas
from astropy.io import fits
from matplotlib import pyplot as plt


pandas.options.mode.copy_on_write = True
pandas.options.future.infer_string = True  # type: ignore


def join_cameras(data: pandas.DataFrame):
    """Matches camera frames and returns focus sweep columns."""

    columns = ["date_obs", "mjd", "frameno", "focusdt", "fwhm", "telescope"]
    data_filter = data.loc[:, columns]

    return data_filter.groupby(["telescope", "mjd", "frameno"]).mean().reset_index()


def filter_focus_sweeps(data: pandas.DataFrame):
    """Filters an AG frames dataframe to return only focus sweep frames."""

    data.reset_index(drop=True, inplace=True)
    data.date_obs = pandas.to_datetime(data.date_obs)

    data_sws: list[pandas.DataFrame] = []

    telescopes = ["sci", "spec", "skye", "skyw"]
    for telescope in telescopes:
        data_tel = data.loc[data.telescope == telescope]
        data_tel = join_cameras(data_tel).sort_values(["mjd", "frameno"])
        data_tel.reset_index(drop=True, inplace=True)

        data_shift_1 = data_tel.shift(1).reset_index(drop=True)
        data_shift_m1 = data_tel.shift(-1).reset_index(drop=True)

        data_sw = data_tel.loc[
            (~data_shift_1.focusdt.isna())
            & (~data_shift_m1.focusdt.isna())
            & (~data_tel.fwhm.isna())
            & (data_tel.focusdt != data_shift_1.focusdt)
            & (data_tel.focusdt != data_shift_m1.focusdt)
        ]

        data_sw["group"] = pandas.Series([], dtype="int32[pyarrow]")

        igroup = 1
        prev_row: pandas.Series | None = None
        for idx, row in data_sw.iterrows():
            if prev_row is None:
                prev_row = row
            elif (row.frameno != prev_row.frameno + 1) or (row.mjd != prev_row.mjd):
                igroup += 1

            data_sw.at[idx, "group"] = igroup
            prev_row = row

        data_sw = data_sw.groupby("group").filter(lambda x: len(x) > 5)
        data_sws.append(data_sw)

    return pandas.concat(data_sws, axis=0, ignore_index=True)


def collect_bench_temps(data: pandas.DataFrame):
    """Collect bench temperatures from the AG headers."""

    data["benchi_temp"] = pandas.Series([], dtype="float32[pyarrow]")
    data["bencho_temp"] = pandas.Series([], dtype="float32[pyarrow]")

    for idx, row in data.iterrows():
        ag_file = pathlib.Path(
            f"/data/agcam/{row.mjd}/"
            f"lvm.{row.telescope}.agcam.east_{row.frameno:08d}.fits"
        )
        if not ag_file.exists():
            continue

        header = fits.getheader(ag_file, 1)
        if "BENTEMPI" in header:
            data.at[idx, "benchi_temp"] = header["BENTEMPI"]
        if "BENTEMPO" in header:
            data.at[idx, "bencho_temp"] = header["BENTEMPO"]

    return data


def analyse_focus_sweeps(data: pandas.DataFrame):
    """Fits focus sweeps and plots results."""

    data = data.where(
        ~(data.benchi_temp.isna() | (data.benchi_temp < -100))
        | ~(data.bencho_temp.isna() | (data.bencho_temp < -100))
    )
    data.loc[data.benchi_temp < -100, "benchi_temp"] = pandas.NA
    data.loc[data.bencho_temp < -100, "bencho_temp"] = pandas.NA

    data = data.groupby(["telescope", "group"]).filter(lambda x: len(x) > 5)

    best_foc = data.groupby(["telescope", "group"]).apply(
        lambda x: x.loc[x.fwhm.idxmin()],
        include_groups=False,
    )
    best_sci = best_foc.loc["sci", :]
    plt.plot(best_sci.benchi_temp, best_sci.focusdt, "o")


if __name__ == "__main__":
    DATA_FILE = pathlib.Path("~/downloads/agcam/agcam_frames.parquet")
    # data = pandas.read_parquet(DATA_FILE, dtype_backend="pyarrow")

    DATA_SW_FILE = DATA_FILE.parent / "agcam_focus_sweeps.parquet"
    # data_sw = filter_focus_sweeps(data)
    # data_sw.to_parquet(DATA_SW_FILE)

    # data_sw = collect_bench_temps(data_sw)
    # data_sw.to_parquet(DATA_SW_FILE)

    analyse_focus_sweeps(pandas.read_parquet(DATA_SW_FILE))
