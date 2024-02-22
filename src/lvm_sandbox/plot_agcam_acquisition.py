#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-02-21
# @Filename: plot_agcam_acquisition.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import matplotlib.dates as mdates
import numpy
import polars as pl
import seaborn
from matplotlib import pyplot as plt


seaborn.set_theme(style="ticks", context="paper", font_scale=1.2)


def plot_agcam_acquisition(guiderdata: str):
    """Plots acquisition AGcam data."""

    df = pl.read_parquet(guiderdata)
    df = df.with_columns(
        date_obs=pl.col.date_obs.str.to_datetime(),
        offset=(pl.col.ra_off**2 + pl.col.dec_off**2) ** 0.5,
    )

    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, hspace=0)

    axes = gs.subplots(sharex=True, sharey=False)
    assert isinstance(axes, numpy.ndarray)

    for ii, ax in enumerate(axes):
        ax.label_outer()

        ax_guide = ax.twinx()

        if ii == 0:
            telescope = "sci"
        elif ii == 1:
            telescope = "skye"
        elif ii == 2:
            telescope = "skyw"
        elif ii == 3:
            telescope = "spec"

        tel_data = df.filter(pl.col.telescope == telescope)

        if telescope != "spec":
            fields = tel_data.group_by("ra_field").agg(pl.col.date_obs.sort().first())
        else:
            fields = (
                df.filter(pl.col.telescope == "sci")
                .group_by("ra_field")
                .agg(pl.col.date_obs.sort().first())
            )

        acq = tel_data.filter(pl.col.guide_mode == "acquisition")
        guide = tel_data.filter(pl.col.guide_mode == "guide")

        guide_p = ax_guide.scatter(
            guide["date_obs"],
            guide["offset"],
            marker=".",
            s=1,
            zorder=10,
        )
        acq_p = ax.scatter(
            acq["date_obs"],
            acq["offset"],
            marker="x",
            s=10,
            color="r",
            zorder=20,
        )

        max_offset = acq["offset"].max()
        assert isinstance(max_offset, float)

        ax.vlines(
            fields["date_obs"],
            0,
            max_offset + 5,
            color="0.5" if telescope != "spec" else "m",
            linestyle="--",
            zorder=5,
        )

        if telescope == "sci":
            ax.legend(
                [acq_p, guide_p],
                ["Acquisition", "Guide"],
                bbox_to_anchor=(0.96, 0.99),
                framealpha=0.9,
            )

        ax.text(0.99, 0.97, telescope, transform=ax.transAxes, ha="right", va="top")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        ax.set_xlim(
            df["date_obs"].dt.offset_by("-5m").min(),
            df["date_obs"].dt.offset_by("5m").max(),
        )
        ax.set_ylim(0, max_offset + 5)

        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Acq. offset [arcsec]")
        ax_guide.set_ylabel("Guide offset [arcsec]")

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    fig.suptitle(f"MJD={df[0, 'mjd']}")

    plt.show()
