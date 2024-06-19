#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-06-15
# @Filename: compare_nominal_to_self_calibration.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import json
import os
import pathlib
import re
from functools import partial

from typing import Literal

import numpy
import polars
import seaborn
from astropy.io import fits
from astropy.stats import biweight_location, biweight_scale
from astropy.table import Table
from gtools.lvm.pipe.flux_calibration import (
    flux_calibration_self,
    get_mean_sensitivity_response,
)
from gtools.lvm.pipe.flux_calibration import (
    log as flux_calibration_log,
)
from gtools.lvm.pipe.tools import get_wavelength_array_from_header
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from sdssdb.peewee.sdss5db import database


PathType = os.PathLike | str | pathlib.Path

CWD = pathlib.Path(__file__).parent
OUTPUTS = CWD / "outputs"


def compare_nominal_to_self_calibration(
    hobject: PathType,
    profile: str | None = None,
    nstar_limit: int = 20,
    gmag_limit: float = 15.0,
    max_sep: float = 7.0,
    skip_if_no_fluxcal: bool = False,
    reject_multiple_per_fibre: bool | Literal["keep_brightest"] = "keep_brightest",
    output_path: PathType | None = None,
):
    """Compares the standards sensitivity function to one derived from field stars.

    Parameters
    ----------
    hobject
        The path to the ``hobject`` file. This must be final stage of the ``hobject``
        file, after flux calibration has been performed and the sensitivity functions
        have been added to the ``FLUXCAL`` extension, but before the flux calibration
        has been applied to the data (the ``lvmFFrame`` file).

    """

    hobject = pathlib.Path(hobject)

    output_path = pathlib.Path(output_path) if output_path else OUTPUTS
    output_path.mkdir(exist_ok=True, parents=True)

    hdul = fits.open(hobject)

    wave = get_wavelength_array_from_header(hdul[0].header)

    # Connect to operations database.
    if profile:
        database.set_profile(profile)

    assert database.connected, "Database is not connected."

    ### Calculate mean sensitivity from FLUXCAL extension. ###
    if has_fluxcal := ("FLUXCAL" in hdul):
        fluxcal = polars.DataFrame(Table(hdul["FLUXCAL"].data).to_pandas())
        fluxcal = fluxcal.select(polars.all().name.to_lowercase())

        mean_fluxcal = biweight_location(fluxcal, axis=1, ignore_nan=True)
        rms_fluxcal = biweight_scale(fluxcal, axis=1, ignore_nan=True)

        if numpy.isnan(mean_fluxcal).all():
            if skip_if_no_fluxcal:
                flux_calibration_log.warning("FLUXCAL contains only NaN values.")
                return

            has_fluxcal = False

    else:
        if skip_if_no_fluxcal:
            flux_calibration_log.warning("FLUXCAL extension not found.")
            return

        mean_fluxcal = numpy.full_like(wave, numpy.nan)
        rms_fluxcal = numpy.full_like(wave, numpy.nan)

    ### Get the sensitivity functions for the standards. ###
    # # NOTE: this is usually not necessary to run since the FLUXCAL extension
    # # already contains the sensitivity functions. Uncomment this section only
    # # if you want to check that the gtools code is doing the same as lvmdrp.

    # df_stds = flux_calibration(hobject, database, plot=False)
    # sens_mean_stds, _ = get_mean_sensitivity_response(df_stds, use_smooth=True)

    # plot_stem = str(OUTPUTS / pathlib.Path(hobject).stem)

    # # Plot the sensitivity function for each Gaia source.
    # _plot_sensitivity_functions(
    #     df_stds,
    #     wave,
    #     f"{plot_stem}_orig_sensitivity.pdf",
    #     title=f"{hobject.name} (standards)",
    # )

    # # Plot mean sensitivity function and residuals.
    # _plot_sensitivity_mean(
    #     df_stds,
    #     wave,
    #     f"{plot_stem}_orig_sensitivity_mean.pdf",
    #     title=f"{hobject.name} (standards)",
    # )

    ### Get the sensitivity functions for field stars. ###
    df_self = flux_calibration_self(
        hobject,
        database,
        plot=True,
        plot_dir=output_path,
        gmag_limit=gmag_limit,
        nstar_limit=nstar_limit,
        max_sep=max_sep,
        reject_multiple_per_fibre=reject_multiple_per_fibre,
    )
    sens_mean_self, sens_rms_self = get_mean_sensitivity_response(df_self)

    ### Plot the mean sensitivity functions and residuals. ###
    seaborn.set_theme(context="paper", style="ticks", font_scale=0.8)

    with plt.ioff():
        if has_fluxcal:
            fig, axes = plt.subplots(2, height_ratios=[4, 1], sharex=True)
            fig.subplots_adjust(hspace=0)

            ax_sens: Axes = axes[0]
        else:
            fig, ax_sens = plt.subplots()

        ax_sens.plot(
            wave,
            sens_mean_self,
            color="b",
            label="Self-calibration",
            zorder=100,
        )
        ax_sens.fill_between(
            wave,
            sens_mean_self - sens_rms_self,
            sens_mean_self + sens_rms_self,
            lw=0,
            color="b",
            alpha=0.3,
        )

        if has_fluxcal:
            ax_sens.plot(wave, mean_fluxcal, color="r", label="Nominal", zorder=99)
            ax_sens.fill_between(
                wave,
                mean_fluxcal - rms_fluxcal,
                mean_fluxcal + rms_fluxcal,
                lw=0,
                color="r",
                alpha=0.3,
            )

        ax_sens.legend()

        if has_fluxcal:
            ax_res: Axes = axes[1]

            ax_res.plot(wave, sens_mean_self / mean_fluxcal, color="k", lw=0.8)

            ax_res.axhline(1.1, 0, 1, color="0.3", linestyle="dashed", lw=0.5)
            ax_res.axhline(0.9, 0, 1, color="0.3", linestyle="dashed", lw=0.5)

            ax_res.set_xlabel("Wavelength [A]")
            ax_res.set_ylabel("Self / nominal")

            # Remove the first y-tick label to avoid overlap with the second axis.
            y_ticks = ax_sens.yaxis.get_major_ticks()
            y_ticks[0].label1.set_visible(False)

            ax_sens.set_ylim(
                min(mean_fluxcal.min(), sens_mean_self.min()) * 0.8,
                max(mean_fluxcal.max(), sens_mean_self.max()) * 1.1,
            )

        else:
            ax_sens.set_ylim(sens_mean_self.min() * 0.8, sens_mean_self.max() * 1.1)

        ax_sens.set_ylabel("Sensitivity response (XP / instrumental)")
        ax_sens.set_title(hobject.name)

        fig.savefig(output_path / f"{hobject.stem}_sensitivity_comparison.pdf")

    # Save files.
    hobject_lit = polars.lit(str(hobject.absolute()), dtype=polars.String)

    if has_fluxcal:
        fluxcal = fluxcal.with_columns(hobject=hobject_lit)
        fluxcal.write_parquet(output_path / f"{hobject.stem}_fluxcal.parquet")

    df_self = df_self.with_columns(hobject=hobject_lit)
    df_self.write_parquet(output_path / f"{hobject.stem}_self.parquet")


def create_file_list(
    root: PathType,
    pattern: str = "**/ancillary/lvm-hobject-*.fits",
    max_mjd=60384,
    reject_tile_ids=[11111],
    reject_no_astro: bool = True,
    outpath: str | None = None,
) -> dict[str, list[str]]:
    """Returns a list of hobject files matching the pattern in the root directory."""

    root = pathlib.Path(root)
    files = root.glob(pattern)

    tree: dict[str, list[str]] = {}

    for file in files:
        dir = str(file.absolute().parent)
        filename = file.name

        match = re.match(r"^.+?/[0-9]{4}XX/([0-9]+)/([0-9]+)", dir)
        if not match:
            raise ValueError(f"Could not extract tile and mjd from {dir}.")

        tile_id = int(match.group(1))
        mjd = int(match.group(2))

        if (max_mjd and mjd > max_mjd) or tile_id in reject_tile_ids:
            continue

        if reject_no_astro:
            hdul = fits.open(file)
            if "ra" not in hdul["SLITMAP"].data.dtype.names:
                continue

        if dir in tree:
            tree[dir].append(filename)
        else:
            tree[dir] = [filename]

    OUTPUTS.mkdir(exist_ok=True, parents=True)
    outpath = str(OUTPUTS / (outpath or "hobject_files.json"))

    with open(outpath, "w") as f:
        f.write(json.dumps(tree, indent=2))

    return tree


def process_file_list(
    tree_file: dict[str, list[str]] | PathType,
    output_path: PathType | None = None,
    **kwargs,
):
    """Processes the list of files."""

    if not isinstance(tree_file, dict):
        with open(tree_file, "r") as f:
            tree = json.load(f)
        log_file = pathlib.Path(tree_file).name
    else:
        tree = tree_file
        log_file = "log"

    flux_calibration_log.sh.setLevel(5)
    flux_calibration_log.start_file_logger(
        str(OUTPUTS / f"{log_file}.log"),
        mode="w",
        rotating=False,
    )

    ndir = len(tree)

    for nn, dir in enumerate(sorted(tree.keys())):
        flux_calibration_log.info(f"Processing {dir} ({nn+1} / {ndir}) ...")

        path = pathlib.Path(dir)
        chunks = list(path.parts[-5:-1])

        partial_call = partial(
            compare_nominal_to_self_calibration,
            skip_if_no_fluxcal=True,
            output_path=(output_path or OUTPUTS) / pathlib.Path(*chunks),
            **kwargs,
        )

        hobjects = [pathlib.Path(dir) / file for file in tree[dir]]
        for hobject in hobjects:
            hdul = fits.open(hobject)
            if "ra" not in hdul["SLITMAP"].data.dtype.names:
                continue

            try:
                partial_call(hobject)
            except Exception as ee:
                flux_calibration_log.error(f"Error processing {hobject}: {ee}")


if __name__ == "__man__":
    pass
