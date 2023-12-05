#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-09-12
# @Filename: pa_analysis.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import astropy.utils.iers
import numpy
import pandas
import seaborn


astropy.utils.iers.conf.auto_download = True
seaborn.set_theme(style="ticks", font_scale=1.2)  # type: ignore


AGCAM_PATH = pathlib.Path("/data/agcam")

OUTPATH = pathlib.Path(__file__).parents[2] / "outputs" / "fwhm_analysis"
OUTPATH.mkdir(parents=True, exist_ok=True)


def compile_fwhm_frame_data():
    """Compiles FWHM data."""

    # We recalculate all the FWHM per frame for each MJD to use 25% percentile.

    frame_data: list[tuple] = []
    for mjd_path in AGCAM_PATH.glob("602*"):
        mjd = int(mjd_path.parts[-1])
        for tel in ["sci", "spec", "skye", "skyw"]:
            source_files = mjd_path.glob(f"lvm.{tel}.agcam.*.parquet")
            for file in source_files:
                data = pandas.read_parquet(file)
                if len(data) == 0:
                    continue

                frameno = int(file.name.split("_")[-1].split(".")[0])
                camera = data.iloc[0].camera
                fwhm = numpy.percentile(data.loc[data.valid == 1].fwhm, 25)

                frame_data.append((mjd, frameno, tel, camera, fwhm))

    frames = pandas.DataFrame(
        frame_data,
        columns=[
            "mjd",
            "frameno",
            "telescope",
            "camera",
            "fwhm",
        ],
    )
    frames.to_parquet(OUTPATH / "frames.parquet")


def compile_fwhm_coadd_data():
    """Compiles FWHM data from co-added frames."""

    frames = pandas.read_parquet(OUTPATH / "frames.parquet")

    coadd_data: list[tuple] = []
    for mjd_path in AGCAM_PATH.glob("602*"):
        mjd = int(mjd_path.parts[-1])
        for tel in ["sci", "spec", "skye", "skyw"]:
            frames_mjd_tel = frames.loc[(frames.telescope == tel) & (frames.mjd == mjd)]
            coadd_files = (mjd_path / "coadds").glob(f"lvm.{tel}.coadd*.fits")
            for file in coadd_files:
                seqno = int(file.name.split("_")[-1].split(".")[0][1:])

                sources_fn = file.name.replace(".fits", "_sources.parquet")
                guiderdata_fn = file.name.replace(".fits", "_guiderdata.parquet")

                sources = pandas.read_parquet(mjd_path / "coadds" / sources_fn)
                guiderdata = pandas.read_parquet(mjd_path / "coadds" / guiderdata_fn)

                guide_frames = guiderdata.loc[guiderdata.guide_mode == "guide"].frameno

                frames_guide = frames_mjd_tel.loc[
                    (frames_mjd_tel.frameno >= guide_frames.min())
                    & (frames_mjd_tel.frameno <= guide_frames.max())
                ]
                frames_fwhm = numpy.median(frames_guide.fwhm)

                if len(sources) == 0:
                    coadd_fwhm = numpy.nan
                else:
                    valid = sources.loc[sources.valid == 1]
                    coadd_fwhm = numpy.percentile(valid.fwhm, 25)

                coadd_data.append((mjd, tel, seqno, coadd_fwhm, frames_fwhm))

    coadd_data_df = pandas.DataFrame(
        coadd_data,
        columns=["mjd", "telescope", "seqno", "coadd_fwhm", "frames_fwhm"],
    )
    coadd_data_df.to_parquet(OUTPATH / "coadds.parquet")


if __name__ == "__main__":
    # compile_fwhm_frame_data()
    compile_fwhm_coadd_data()
