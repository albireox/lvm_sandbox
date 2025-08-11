#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-04-08
# @Filename: focus_temperature.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib

import polars
import scipy
import seaborn
from astropy.io import fits
from matplotlib import pyplot as plt
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)


AGCAM_ROOT = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/agcam/lco"


def _process_mjd(dir: pathlib.Path):
    """Processes an AGCAM MJD directory and collects focus sweep data."""

    data: list[dict] = []

    mjd = dir.parts[-1]

    for telescope in ["sci", "spec", "skye", "skyw"]:
        n_sweep = 0
        was_sweep = False

        files = sorted(dir.glob(f"lvm.{telescope}.agcam*.fits"))

        for file in files:
            image_no = int(file.stem.split("_")[-1])

            guider_path = dir / f"lvm.{telescope}.guider_{image_no:08d}.fits"
            if guider_path.exists():
                continue

            with fits.open(file) as hdul:
                if "RAW" not in hdul or "PROC" not in hdul:
                    continue

                raw_header = hdul["RAW"].header
                proc_header = hdul["PROC"].header

                if "ISFSWEEP" not in proc_header or not proc_header["ISFSWEEP"]:
                    was_sweep = False
                    continue

                if was_sweep is False:
                    n_sweep += 1
                    was_sweep = True

                data.append(
                    {
                        "mjd": int(mjd),
                        "image_no": image_no,
                        "telescope": raw_header["TELESCOP"],
                        "camname": raw_header["CAMNAME"],
                        "bentempi": raw_header["BENTEMPI"],
                        "bentempo": raw_header["BENTEMPO"],
                        "benhumi": raw_header["BENHUMI"],
                        "benhumo": raw_header["BENHUMO"],
                        "focusdt": raw_header["FOCUSDT"],
                        "airmass": raw_header["AIRMASS"],
                        "fwhm": proc_header["FWHM"],
                        "n_sweep": n_sweep,
                    }
                )

    return data


def get_dirs(dirs: list[pathlib.Path], min_mjd: int | None, max_mjd: int | None):
    """Filters directories based on MJD range."""

    if min_mjd is not None:
        dirs = [d for d in dirs if int(d.parts[-1]) >= min_mjd]

    if max_mjd is not None:
        dirs = [d for d in dirs if int(d.parts[-1]) <= max_mjd]

    return dirs


def collect_agcam_data():
    """Collects focus sweep data from AGCAM files."""

    agcam_path = pathlib.Path(AGCAM_ROOT)

    dirs = sorted(agcam_path.glob("6*"))
    dirs = get_dirs(dirs, min_mjd=60350, max_mjd=60900)

    data: list[dict] = []

    with Progress(
        TextColumn("[yellow]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        expand=True,
        refresh_per_second=1,
    ) as progress:
        tid = progress.add_task("Processing AGCAM data ...", total=len(dirs))
        with multiprocessing.Pool(processes=6) as pool:
            data_imap = pool.imap(_process_mjd, dirs)
            for dd in data_imap:
                progress.update(tid, advance=1)
                data += dd

    df = polars.DataFrame(
        data,
        schema={
            "mjd": polars.Int32,
            "image_no": polars.Int32,
            "telescope": polars.String,
            "camname": polars.String,
            "bentempi": polars.Float32,
            "bentempo": polars.Float32,
            "benhumi": polars.Float32,
            "benhumo": polars.Float32,
            "focusdt": polars.Float32,
            "airmass": polars.Float32,
            "fwhm": polars.Float32,
            "n_sweep": polars.Int32,
        },
    )

    return df.sort(["mjd", "telescope", "image_no"])


def fit_data(telescope: str, data: polars.DataFrame):
    """Fits focus-temperature."""

    avg = data.group_by("frame.focus_position", as_index=False).mean()

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


def process_focus_data(data: polars.DataFrame | str | pathlib.Path):
    """Processes focus data and determines the focus-temperature relation."""
