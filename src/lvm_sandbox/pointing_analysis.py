#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-10-07
# @Filename: pointing_analysis.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy
import polars
import seaborn
from astropy.coordinates import angular_separation


INPUTS_DIR = pathlib.Path(__file__).parent / "../../inputs"
OUTPUT_DIR = pathlib.Path(__file__).parent / "../../outputs/pointing_analysis_60996"


def hypot(x: polars.Expr, y: polars.Expr) -> polars.Expr:
    """Calculates the hypotenuse of two polars expressions."""

    return (x**2 + y**2).sqrt()


def great_circle_distance(
    ra: polars.Expr,
    dec: polars.Expr,
    ra_off: polars.Expr,
    dec_off: polars.Expr,
) -> polars.Expr:
    """Calculates the great circle distance between two points on the sphere."""

    ra_off = (ra_off / 3600).radians()
    dec_off = (dec_off / 3600).radians()

    ra = ra.radians()
    ra1 = ra + ra_off

    dec = dec.radians()
    dec1 = dec + dec_off

    sep = angular_separation(ra, dec, ra1, dec1)

    return sep * 180 / numpy.pi


def process_exposure(df: polars.DataFrame) -> polars.DataFrame:
    """Processes a single exposure."""

    df = df.sort("frameno")

    guide_frames = df.filter(polars.col.guide_mode == "guide")
    guide_avg = guide_frames["sep_gc"].mean()
    guide_std = guide_frames["sep_gc"].std()

    acq_1 = df.filter(polars.col.guide_mode == "acquisition").sort("frameno").head(1)
    acq_1 = acq_1.with_columns(
        guide_avg=polars.lit(guide_avg),
        guide_std=polars.lit(guide_std),
    )

    acq_1 = acq_1.rename({"sep_gc": "acq_sep_gc", "sep_eucl": "acq_sep_eucl"})

    return acq_1


def process_guider_frame_data():
    """Pointing analysis."""

    GUIDER_FRAME_FILE = INPUTS_DIR / "guider_frame_60996.parquet"

    df = polars.read_parquet(GUIDER_FRAME_FILE)

    df = (
        df.with_columns(
            sep_eucl=hypot(
                polars.col.ra_off * polars.col.dec.radians().cos(),
                polars.col.dec_off,
            ),
            sep_gc=great_circle_distance(
                polars.col.ra,
                polars.col.dec,
                polars.col.ra_off,
                polars.col.dec_off,
            )
            * 3600.0,
        )
        .drop_nans(["sep_eucl", "sep_gc", "exposure_no"])
        .drop_nulls(["sep_eucl", "sep_gc", "exposure_no"])
    )

    # Require at least 50 guide frames per exposure.
    df = df.filter((polars.col.pk.len() > 50).over("exposure_no"))

    # Remove dithered exposures.
    df = df.filter(polars.col.dpos == 0)

    df_exp = df.group_by("exposure_no", "telescope").map_groups(process_exposure)

    df_exp.write_parquet(OUTPUT_DIR / "guider_frame_60996_acq.parquet", mkdir=True)

    return df, df_exp


def plot_initial_pointing():
    """Produces initial pointing analysis plots."""

    seaborn.set_theme("paper", style="darkgrid", font_scale=1.2, color_codes=True)
    plt.ioff()

    df = polars.read_parquet(OUTPUT_DIR / "guider_frame_60996_acq.parquet")

    fig, axes = plt.subplots(4, 1, figsize=(16, 18))

    for i_ax, telescope in enumerate(["sci", "spec", "skye", "skyw"]):
        df_telescope = df.filter(polars.col.telescope == telescope)

        seaborn.scatterplot(
            data=df_telescope,
            x="mjd",
            y="acq_sep_gc",
            c="0.7",
            s=10,
            ax=axes[i_ax],
        )

        seaborn.lineplot(
            data=df_telescope,
            x="mjd",
            y="acq_sep_gc",
            ax=axes[i_ax],
            label=telescope,
        )

        if i_ax != 3:
            axes[i_ax].set_xlabel(None)
        else:
            axes[i_ax].set_xlabel("MJD")

        axes[i_ax].set_ylabel("Initial acquisition pointing error [arcsec]")

        if telescope in ["spec", "skye"]:
            axes[i_ax].set_ylim(0, 150)

        seaborn.move_legend(axes[i_ax], "upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pointing_initial_acq.pdf")
    plt.close()


def plot_guiding_avg():
    """Produces guiding  analysis plots."""

    seaborn.set_theme("paper", style="darkgrid", font_scale=1.2, color_codes=True)
    plt.ioff()

    df = polars.read_parquet(OUTPUT_DIR / "guider_frame_60996_acq.parquet")

    fig, axes = plt.subplots(4, 1, figsize=(16, 18))

    for i_ax, telescope in enumerate(["sci", "spec", "skye", "skyw"]):
        df_telescope = df.filter(polars.col.telescope == telescope)

        seaborn.scatterplot(
            data=df_telescope,
            x="mjd",
            y="guide_avg",
            c="0.7",
            s=10,
            ax=axes[i_ax],
        )

        seaborn.lineplot(
            data=df_telescope,
            x="mjd",
            y="guide_avg",
            ax=axes[i_ax],
            label=telescope,
        )

        if i_ax != 3:
            axes[i_ax].set_xlabel(None)
        else:
            axes[i_ax].set_xlabel("MJD")

        axes[i_ax].set_ylabel("Guiding average [arcsec]")

        seaborn.move_legend(axes[i_ax], "upper left")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "guiding_avg.pdf")
    plt.close()
