#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-09-12
# @Filename: utils.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
from os import PathLike

from typing import Literal

import pandas
from astropy.io import fits

from lvmguider.tools import get_model


__all__ = ["concat_agcam_data"]


AnyPath = str | PathLike


def concat_agcam_data(
    type: Literal["frames", "guiderdata", "coadds"],
    path: AnyPath = "/data/agcam",
):
    """Concatenates all AGcam data."""

    path = pathlib.Path(path)

    if type == "frames" or type == "guiderdata":
        files = path.glob(f"**/coadds/lvm.guider.mjd_*_{type}.parquet")
        data_mjd: list[pandas.DataFrame] = []
        for file in files:
            df = pandas.read_parquet(file)
            df["mjd"] = int(file.parts[-3])
            data_mjd.append(df)
        return pandas.concat(data_mjd)

    global_dm = get_model("GLOBAL_COADD")
    files = path.glob("**/coadds/lvm.*.coadd_*.fits")

    data: list[dict] = []
    for file in files:
        header = fits.getheader(file, "GLOBAL")
        header_dict = dict(header)

        data.append({kk: vv for kk, vv in header_dict.items() if kk in global_dm})

    df = pandas.DataFrame(data)
    df = df.rename(columns={col: col.lower() for col in df.columns})

    return df
