#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-01-29
# @Filename: plot_skye_fwhm.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import pandas
import seaborn
from matplotlib import pyplot as plt


seaborn.set_theme(font_scale=1.2)


def plot_skye_fwhm():
    """Plots SkyE FWHM vs coordinates."""

    path = pathlib.Path("/Users/gallegoj/Downloads/agcam/")

    data = pandas.concat(
        [
            pandas.read_parquet(fn, dtype_backend="pyarrow")
            for fn in path.glob("6*_agcam_frames.parquet")
        ]
    )

    data["date_obs"] = pandas.to_datetime(data.date_obs)
    data = data.loc[data.telescope == "skye"]

    data_5 = data.loc[(data.fwhm > 6), ["az", "alt", "saz", "sel"]]
    data_5.dropna(inplace=True)

    seaborn.histplot(data_5, x="az", y="alt", bins=20, cbar=True)

    plt.figure()
    seaborn.histplot(data_5, x="saz", y="sel", bins=20, cbar=True)

    plt.show()
