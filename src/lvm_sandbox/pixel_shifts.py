#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-08-26
# @Filename: pixel_shifts.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import functools
import multiprocessing
import pathlib
import re
import subprocess

from typing import Literal

import numpy
import polars
from astropy.io import fits
from rich import print as rprint
from sklearn.cluster import DBSCAN


DATA_ROOT = pathlib.Path("/data/spectro/lvm/pixel_shifts")


def rsync_shift_monitor_files():
    """Rsyncs the pixel shift files from Utah to a local directory."""

    dst = DATA_ROOT / "shift_monitor"
    dst.mkdir(parents=True, exist_ok=True)

    # Rsync the files from the source to the destination
    subprocess.run(
        [
            "rsync",
            "-av",
            "--progress",
            "utah-pipelines:/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/lvm/sandbox/shift_monitor/",
            str(dst),
        ]
    )


def parse_shift_monitor_files():
    """Parses the shift monitor files into a dataframe."""

    rows: list[tuple] = []

    for file in DATA_ROOT.glob("shift_monitor/shift_*.txt"):
        with open(file, "r") as f:
            data = f.read()

        match = re.findall(
            r"(?P<MJD>6[0-9]{4})[ ]+(?P<exp_no>[0-9]+)[ ]+"
            r"(?P<exp_type>[a-zA-Z]+)[ ]*(?P<spec>sp[1-3])?",
            data,
            re.MULTILINE,
        )
        rows.extend(match)

    df = polars.DataFrame(
        rows,
        schema={
            "MJD": polars.Int32,
            "exp_no": polars.Int32,
            "exp_type": polars.String,
            "spec": polars.String,
        },
        orient="row",
    ).sort(["MJD", "exp_no"])

    return df


def rsync_spectro_files(
    mjd: int,
    exp_no: int,
    spec: Literal["sp1", "sp2", "sp3"] | None = None,
):
    """Rsyncs the spectroscopic files from Utah to the local directory."""

    dst = DATA_ROOT / f"{mjd}"
    dst.mkdir(parents=True, exist_ok=True)

    if spec:
        query = f"sdR-s-*{spec[-1]}-*{exp_no}.fits.gz"
    else:
        query = f"sdR-s-*{exp_no}.fits.gz"

    # Rsync the files from the source to the destination
    subprocess.run(
        [
            "rsync",
            f"utah-pipelines:/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/lvm/lco/{mjd}/{query}",
            str(dst),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    subprocess.run(
        [
            "chmod",
            "-R",
            "755",
            str(dst),
        ]
    )


def moving_average(a, n=3):
    """Compute a rolling mean."""

    ret = numpy.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def _process_spectro_file(file: pathlib.Path):
    """Processes a single spectrograph file."""

    hdul = fits.open(file)

    mjd = hdul[0].header["SMJD"]
    exp_no = hdul[0].header["EXPOSURE"]
    exp_type = hdul[0].header["IMAGETYP"]
    exp_time = hdul[0].header["EXPTIME"]
    spec = hdul[0].header["SPEC"]

    ccd_temp = hdul[0].header["CCDTEMP1"]
    ln2_temp = hdul[0].header["CCDTEMP2"]

    return_dict = {
        "MJD": mjd,
        "exp_no": exp_no,
        "exp_type": exp_type,
        "exp_time": exp_time,
        "spec": spec,
    }

    if ccd_temp > -80 or ln2_temp > -150:
        return None

    # Get the first column of the data array.
    col = hdul[0].data[:, 0].astype("f4")

    # Start with a simple test. If the standard deviation of the first column for
    # each quadrant is small, that means that there are no shifted pixels.
    col0 = col[0:2040]
    col1 = col[2040:]

    if col0.std() < 5 and col1.std() < 5:
        return None

    if col0.std() > 5:
        i0 = 0
        col = col0
    else:
        i0 = 2040
        col = col1

    # Now identify the location of the shifted pixel. We first run a rolling mean to
    # smooth the data. This means the position we'll determine is only approximate,
    # but that's good enough.
    col = moving_average(col, n=10)

    # Now use a DBSCAN algorithm to identify the shifted pixels.
    labels = DBSCAN(eps=0.5, min_samples=10).fit_predict(col.reshape(-1, 1))

    # If all labels are -1 or all are 0, then no clusters were found.
    if numpy.all(labels == -1):
        return None

    labels = labels[labels != -1]
    if numpy.all(labels == 0):
        return None

    # Find indices in which the label changes
    ch_idx = numpy.where(labels[:-1] != labels[1:])[0]

    # There should only be one change, corresponding to the pixel shift.
    if len(ch_idx) > 1:
        rprint(f"[yellow]Multiple pixel shifts found in {file}[/yellow]")
        return None

    return_dict["y_shift"] = i0 + ch_idx[0] + 9  # +9 because of the rolling mean

    return return_dict


def _process_spectro_mjd(
    dir_path: pathlib.Path,
    camera: Literal["r", "b", "z"] = "z",
    skip_mjds: list[int] = [],
) -> tuple[int, list[dict] | None]:
    """Processes all spectrograph files in a given MJD directory."""

    mjd = int(dir_path.name)
    if mjd in skip_mjds or mjd < 60200:
        return (mjd, None)

    data_files = sorted(dir_path.glob(f"sdR-s-{camera}*-*.fits.gz"))
    if len(data_files) == 0:
        return (mjd, None)

    results = []
    for file in data_files:
        try:
            data = _process_spectro_file(file)
        except Exception as e:
            rprint(f"[red]Error processing {file}: {e}[/red]")
            continue

        if data:
            results.append(data)

    return (mjd, results)


def generate_pixel_shift_list(
    camera: Literal["r", "b", "z"] = "z",
    n_cpus: int = 10,
    skip_processed: bool = True,
):
    """Processes all LVM raw images to identify files with pixel shifts.

    Parameters
    ----------
    camera
        The camera to use. Only the file corresponding to this camera for each
        spectrograph will be used.
    n_cpus
        The number of CPUs to use for processing.
    skip_processed
        If :obj:`True`, skip MJDs that have already been processed.

    Returns
    -------
    pixel_shift
        A Polars data frame with the list of pixel shifts. The data is also
        written to ``data/pixel_shifts.parquet``.

    """

    DATA_DIR = pathlib.Path(__file__).parents[2] / "data"
    DF_PATH = DATA_DIR / "pixel_shifts.parquet"

    SPECTRO_DIR = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/lvm/lco/"

    SCHEMA = {
        "MJD": polars.Int64,
        "exp_no": polars.Int32,
        "spec": polars.String,
        "exp_type": polars.String,
        "exp_time": polars.Float32,
        "y_shift": polars.Int32,
    }

    dirs = list(sorted(pathlib.Path(SPECTRO_DIR).glob("6*")))

    if DF_PATH.exists() and not skip_processed:
        df = polars.read_parquet(DF_PATH)
    else:
        df = polars.DataFrame(None, schema=SCHEMA)

    mjds = df["MJD"].unique().to_list()

    _partial_process_mjd = functools.partial(
        _process_spectro_mjd,
        camera=camera,
        skip_mjds=mjds,
    )

    with multiprocessing.Pool(processes=n_cpus) as pool:
        for mjd, data in pool.imap_unordered(_partial_process_mjd, dirs):
            if data and len(data) > 0:
                rprint(f"Adding results for MJD [blue]{mjd}[/blue]")

                df_mjd = polars.DataFrame(data, schema=SCHEMA)
                df = polars.concat([df, df_mjd], how="vertical")
                df.write_parquet(DF_PATH)

        return df
