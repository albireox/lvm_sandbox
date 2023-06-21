#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-18
# @Filename: plot_tracking.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib
from functools import partial

from lvmguider.transformations import solve_from_files


def get_wcs(mjd: int, telescope: str, seqno: int):
    """Returns the pointing WCS."""

    mjd_path = pathlib.Path(f"/data/agcam/{mjd}")
    files = list(mjd_path.glob(f"*{telescope}.agcam.*_{seqno:08d}.fits"))
    if len(files) < 2:
        return None

    return solve_from_files([str(fn) for fn in files], telescope=telescope)


def process_agcam(mjd: int, seq0: int, seq1: int, telescope="sci"):
    """Processes AG frames."""

    with multiprocessing.Pool(4) as pool:
        wcs = pool.map(partial(get_wcs, mjd, telescope), range(seq0, seq1 + 1))

    return wcs
