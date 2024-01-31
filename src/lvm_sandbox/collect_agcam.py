#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-01-29
# @Filename: collect_agcam.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib

from typing import Any

import numpy
import pandas
from astropy.io.fits import getheader
from astropy.utils.iers import conf

from lvmguider.dataclasses import CameraSolution


conf.auto_download = True


def azel2sazsel(azD, elD):
    """Returns the siderostat coordinates (saz, Sel) for the supplied Az-El.

    From Tom Herbst and Florian Briegel.

    Parameters
    ----------
    az,el
        Azimuth and Elevation (Altitude) in decimal degrees

    Returns
    -------
    sazD,selD
        Siderostat angles in degrees

    """

    r90 = numpy.radians(90.0)  # 90 deg in radians
    az, el = numpy.radians(azD), numpy.radians(elD)  # Convert to radians
    SEl = numpy.arccos(numpy.cos(el) * numpy.cos(az)) - r90  # SEl in radians
    rat = numpy.sin(el) / numpy.cos(SEl)  # Ratio
    if azD < 180.0:
        SAz = r90 - numpy.arcsin(rat)  # saz in radians
    else:
        SAz = numpy.arcsin(rat) - r90

    return numpy.degrees(SAz), numpy.degrees(SEl)  # Return values in degrees


def _collect_mjd(path: pathlib.Path):
    """Collect AG camera date from an MJD."""

    if (path / "reprocessed").exists():
        path = path / "reprocessed"
        mjd = int(path.parts[-2])
    else:
        mjd = int(path.parts[-1])

    agcam_files = path.glob("lvm.*.agcam.*.fits")
    data: list[dict[str, Any]] = []

    agcam_data_path = pathlib.Path.cwd() / f"{mjd}_agcam_frames.parquet"
    if agcam_data_path.exists():
        return

    print(f"Running {path!s} ...")

    sources = []
    sources_path = pathlib.Path.cwd() / f"{mjd}_sources.parquet"

    for agf in agcam_files:
        frameno = int(agf.parts[-1].split(".")[3].split("_")[-1])

        try:
            solution = CameraSolution.open(agf)
            header = getheader(agf, 1)

            date_obs = header["DATE-OBS"]
            if solution.solved:
                ra, dec = solution.pointing
            else:
                ra = header["RA"]
                dec = header["DEC"]

            alt = header["ALT"]
            az = header["AZ"]

            if not numpy.isnan(ra) and not numpy.isnan(dec):
                saz, sel = azel2sazsel(alt, az)
            else:
                saz = sel = numpy.nan

            data.append(
                dict(
                    mjd=mjd,
                    frameno=frameno,
                    telescope=solution.telescope,
                    camera=solution.camera,
                    date_obs=date_obs,
                    exptime=header["EXPTIME"],
                    kmirror_drot=header["KMIRDROT"],
                    focusdt=header["FOCUSDT"],
                    fwhm=solution.fwhm,
                    ra=ra,
                    dec=dec,
                    alt=alt,
                    az=az,
                    saz=saz,
                    sel=sel,
                    pa=solution.pa,
                    zero_point=solution.zero_point,
                    solved=solution.solved,
                    wcs_mode=solution.wcs_mode,
                )
            )
        except Exception:
            continue

        if not sources_path.exists():
            sources_frame = solution.sources
            if sources_frame is not None:
                sources_frame["mjd"] = mjd
                sources_frame["telescope"] = solution.telescope
                sources_frame["camera"] = solution.camera
                sources_frame["frameno"] = frameno

                sources.append(sources_frame)

    if len(sources) > 0 and not sources_path.exists():
        sources = pandas.concat(
            [source.dropna(axis=1, how="all") for source in sources],
            ignore_index=True,
        )
        sources = sources.convert_dtypes(dtype_backend="pyarrow")
        sources.to_parquet(sources_path)

    if not agcam_data_path.exists():
        data_df = pandas.DataFrame(
            data,
            columns=[
                "mjd",
                "frameno",
                "telescope",
                "camera",
                "date_obs",
                "exptime",
                "kmirror_drot",
                "focusdt",
                "fwhm",
                "ra",
                "dec",
                "alt",
                "az",
                "saz",
                "sel",
                "pa",
                "zero_point",
                "solved",
                "wcs_mode",
            ],
        )
        data_df = data_df.convert_dtypes(dtype_backend="pyarrow")
        data_df.to_parquet(agcam_data_path)

        return data_df

    return None


def collect_agcam():
    """Collect AG camera data."""

    path = pathlib.Path("/data/agcam")
    agcam_dirs = sorted(path.glob("6*"))

    # for agcam_dir in agcam_dirs[::-1]:
    #     _collect_mjd(agcam_dir)

    with multiprocessing.Pool(processes=4) as pool:
        pool.map(_collect_mjd, agcam_dirs)
