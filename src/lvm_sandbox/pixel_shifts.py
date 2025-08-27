#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-08-26
# @Filename: pixel_shifts.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re
import subprocess

from typing import Literal

import polars


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
