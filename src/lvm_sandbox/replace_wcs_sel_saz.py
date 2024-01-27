#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-11-28
# @Filename: replace_wcs_sel_saz.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import sys

import pandas
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points

from lvmguider.transformations import radec2azel


def replace_wcs_sel_saz(path: pathlib.Path):
    """Replaces the WCS of an AGcam file with the SAz/SEl equivalent."""

    hdul = fits.open(path)

    wcs_proc = WCS(hdul["PROC"].header)
    for keyword in wcs_proc.to_header():
        del hdul["PROC"].header[keyword]

    sources_file = path.with_suffix(".parquet")
    sources = pandas.read_parquet(sources_file)
    sources = sources.loc[~sources.match_sep.isna()]

    raw_header = hdul["RAW"].header
    lst = raw_header["LMST"]

    saz: list[float] = []
    sel: list[float] = []

    for _, row in sources.iterrows():
        saz_, sel_ = radec2azel(row.ra_epoch, row.dec_epoch, lst * 15)
        saz.append(saz_)
        sel.append(sel_)

    xy = sources.loc[:, ["x", "y"]].to_numpy()
    world = SkyCoord(saz, sel, unit="deg", frame="icrs")

    wcs_azel = fit_wcs_from_points(xy.T, world)
    assert isinstance(wcs_azel, WCS)

    hdul["RAW"].header.update(wcs_azel.to_header())

    hdul.writeto(f"/Users/gallegoj/Downloads/{path.name}", overwrite=True)


if __name__ == "__main__":
    file_ = pathlib.Path(sys.argv[1])
    replace_wcs_sel_saz(file_)
