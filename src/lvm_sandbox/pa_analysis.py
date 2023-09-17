#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-09-12
# @Filename: pa_analysis.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
from functools import partial

from typing import Callable, Sequence

import astropy.utils.iers
import numpy
import pandas
import seaborn
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from lvmguider.tools import polyfit_with_sigclip
from lvmguider.transformations import azel2sazsel
from matplotlib import pyplot as plt

from lvm_sandbox.tools.utils import concat_agcam_data


astropy.utils.iers.conf.auto_download = True
seaborn.set_theme(style="ticks", font_scale=1.2)  # type: ignore


AGCAM_PATH = pathlib.Path("/data/agcam")

OUTPATH = pathlib.Path(__file__).parents[2] / "outputs" / "pa_analysis"
OUTPATH.mkdir(parents=True, exist_ok=True)

SITE = EarthLocation.from_geodetic(
    lon=-70.70166667,
    lat=-29.00333333,
    height=2282.0,
)


def reduce_one(
    frame_data: pandas.DataFrame,
    data: pandas.DataFrame,
):
    """Calculates the summary for a set of guider frames for a single exposure."""

    if len(data) == 0:
        return pandas.Series()

    data = data.loc[data.guide_mode == "guide"]

    frame_data = frame_data.copy()
    frame_data = frame_data.loc[frame_data.spec_frameno == data.spec_frameno.iloc[0]]

    data.set_index(["spec_frameno", "frameno"], inplace=True)
    frame_data.set_index(["spec_frameno", "frameno"], inplace=True)

    data = data.join(frame_data, rsuffix="_frame", how="inner")

    data["time"] = list(map(lambda do: Time(do, format="isot").unix, data.date_obs))

    data.sort_values("time", inplace=True)
    data["elapsed"] = data["time"]
    data["elapsed"] -= data.iloc[0].elapsed

    ra_field = data.ra_field.iloc[0]
    dec_field = data.dec_field.iloc[0]
    pa_field = data.pa_field.iloc[0]
    pa_off = data.pa_field.iloc[0] - (data.pa.mean() - 180)

    coeffs = polyfit_with_sigclip(data.elapsed.to_numpy(), data.pa.to_numpy())
    deg_s = coeffs[0]
    arcmin_15min = deg_s * 60 * 15 * 60

    date_obs = Time(data.date_obs.iloc[0], format="isot")
    date_obs.location = SITE

    sc = SkyCoord(
        ra=ra_field,
        dec=dec_field,
        unit="deg",
        frame="icrs",
        location=SITE,
        obstime=date_obs,
    )

    lst = date_obs.sidereal_time("mean").deg
    ha = lst - ra_field

    alt = sc.altaz.alt.deg
    az = sc.altaz.az.deg

    saz, sel = azel2sazsel(az, alt)

    return pandas.Series(
        {
            "ra": ra_field,
            "dec": dec_field,
            "ha": ha,
            "alt": alt,
            "az": az,
            "sel": sel,
            "saz": saz,
            "pa": pa_field,
            "pa_off": pa_off,
            "pa_drift_15min": arcmin_15min,
        }
    )


def plot_axes(
    data: pandas.DataFrame,
    col_value: str,
    col0: str,
    col1: str,
    range0: Sequence[float],
    range1: Sequence[float],
    bin0: float,
    bin1: float,
    filename: str,
    flipy: bool = False,
    ax0_label: str | None = None,
    ax1_label: str | None = None,
    preplot: Callable | None = None,
    abs_value: bool = True,
    vrange: Sequence[float | None] = [-1.4, 1.4],
):
    """Plots data."""

    means = []
    for ax0 in numpy.arange(range0[0], range0[1], bin0):
        for ax1 in numpy.arange(range1[0], range1[1], bin1):
            slice = data.loc[
                data[col0].between(ax0, ax0 + 30) & data[col1].between(ax1, ax1 + 5),
                :,
            ]

            values = abs(slice[col_value]) if abs_value else slice[col_value]
            means.append((ax0, ax1, values.mean()))

    bins = pandas.DataFrame(means, columns=[col0, col1, "value"])
    bins = bins.pivot(columns=col0, index=col1, values="value")

    if preplot:
        bins = preplot(bins)

    cmap = "vlag" if vrange[0] is not None and vrange[0] < 0 else "crest"

    fig, ax = plt.subplots()
    seaborn.heatmap(
        bins,
        vmin=vrange[0],
        vmax=vrange[1],
        cmap=cmap,
        ax=ax,
        cbar_kws={"label": "PA change rate [arcmin / 15 min]"},
    )

    if ax0_label:
        ax.set_xlabel(ax0_label)

    if ax1_label:
        ax.set_ylabel(ax1_label)

    if flipy:
        ax.invert_yaxis()

    telescope = data.index[0][1]
    fig.savefig(str(OUTPATH / filename.format(tel=telescope)))

    return


def calculate_pa_drift():
    """Recalculates the PA drift in deg/s."""

    guider_data = concat_agcam_data("guiderdata")
    frame_data = concat_agcam_data("frames")

    frame_data = frame_data.loc[frame_data.camera == "east"]

    guider_data = guider_data.loc[guider_data.telescope != "spec"]
    guider_data = guider_data.loc[guider_data.mjd >= 60175]
    guider_data = guider_data.dropna(subset=["pa", "pa_field"])

    guider_data = guider_data.groupby("spec_frameno").filter(lambda gg: len(gg) > 50)
    guider_data = guider_data.groupby("telescope").filter(lambda gg: len(gg) > 0)

    reduce_one_p = partial(reduce_one, frame_data)
    exp_data = guider_data.groupby(["spec_frameno", "telescope"]).apply(reduce_one_p)

    exp_data.to_parquet(OUTPATH / "guider_data.parquet")


def plot_pa_drift(file: pathlib.Path):
    """Plots PA data."""

    data = pandas.read_parquet(file)
    data = data.loc[abs(data.pa_drift_15min) < 5]

    print(data.groupby("telescope").apply(lambda gg: abs(gg.pa_drift_15min).describe()))

    for tel, grp in data.groupby("telescope"):
        plot_axes(
            grp,
            col_value="pa_drift_15min",
            col0="ra",
            col1="dec",
            range0=[0, 360],
            range1=[-80, 0],
            bin0=15,
            bin1=5,
            filename=f"drift_{tel}_radec.pdf",
            ax0_label="Right Ascension [deg]",
            ax1_label="Declination [deg]",
            abs_value=False,
        )

        plot_axes(
            grp,
            col_value="pa_drift_15min",
            col0="ha",
            col1="dec",
            range0=[-45, 45],
            range1=[-80, 0],
            bin0=5,
            bin1=5,
            filename=f"drift_{tel}_hadec.pdf",
            ax0_label="Hour Angle [deg]",
            ax1_label="Declination [deg]",
            abs_value=False,
        )

        plot_axes(
            grp,
            col_value="pa_drift_15min",
            col0="az",
            col1="alt",
            range0=[-10, 85],
            range1=[20, 90],
            bin0=5,
            bin1=5,
            flipy=True,
            filename=f"drift_{tel}_azalt.pdf",
            ax0_label="Az [deg]",
            ax1_label="Alt [deg]",
            abs_value=False,
        )

        plot_axes(
            grp,
            col_value="pa_drift_15min",
            col0="saz",
            col1="sel",
            range0=[-90, 90],
            range1=[-20, 55],
            bin0=5,
            bin1=5,
            filename=f"drift_{tel}_sazsel.pdf",
            flipy=True,
            ax0_label="SAz [deg]",
            ax1_label="SEl [deg]",
            abs_value=False,
        )

    plt.close("all")


def plot_pa_off(file: pathlib.Path):
    """Plots PA offset data."""

    data = pandas.read_parquet(file)
    data = data.loc[abs(data.pa_off) < 5]

    print(data.groupby("telescope").apply(lambda gg: gg.pa_off.describe()))

    for tel, grp in data.groupby("telescope"):
        plot_axes(
            grp,
            col_value="pa_off",
            col0="ra",
            col1="dec",
            range0=[0, 360],
            range1=[-80, 0],
            bin0=15,
            bin1=5,
            filename=f"pa_off_{tel}_radec.pdf",
            ax0_label="Right Ascension [deg]",
            ax1_label="Declination [deg]",
            vrange=[None, None],
            abs_value=False,
        )

        plot_axes(
            grp,
            col_value="pa_off",
            col0="ha",
            col1="dec",
            range0=[-45, 45],
            range1=[-80, 0],
            bin0=5,
            bin1=5,
            filename=f"pa_off_{tel}_hadec.pdf",
            ax0_label="Hour Angle [deg]",
            ax1_label="Declination [deg]",
            vrange=[None, None],
            abs_value=False,
        )

        plot_axes(
            grp,
            col_value="pa_off",
            col0="az",
            col1="alt",
            range0=[-10, 85],
            range1=[20, 90],
            bin0=5,
            bin1=5,
            flipy=True,
            filename=f"pa_off_{tel}_azalt.pdf",
            ax0_label="Az [deg]",
            ax1_label="Alt [deg]",
            vrange=[None, None],
            abs_value=False,
        )

        plot_axes(
            grp,
            col_value="pa_off",
            col0="saz",
            col1="sel",
            range0=[-90, 90],
            range1=[-20, 55],
            bin0=5,
            bin1=5,
            filename=f"pa_off_{tel}_sazsel.pdf",
            flipy=True,
            ax0_label="SAz [deg]",
            ax1_label="SEl [deg]",
            vrange=[None, None],
            abs_value=False,
        )

    plt.close("all")


if __name__ == "__main__":
    # calculate_pa_drift()
    plot_pa_drift(OUTPATH / "guider_data.parquet")
    plot_pa_off(OUTPATH / "guider_data.parquet")
