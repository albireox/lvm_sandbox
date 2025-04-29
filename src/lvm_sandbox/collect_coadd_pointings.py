#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-04-08
# @Filename: collect_coadd_pointings.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib
import re
from functools import lru_cache

import numpy
import polars
from astropy.io import fits
from astropy.wcs import WCS
from rich.progress import track


AGCAM_ROOT = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/agcam/lco"
SPECTRO_ROOT = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/lvm/spectro/data/lco/"

DITHER_OFFSETS = {
    0: [0.00, 0.00],
    1: [-10.68, 18.50],
    2: [10.68, 18.50],
    3: [0.00, -12.33],
    4: [10.68, -6.17],
    5: [-10.68, -6.17],
    6: [10.68, 6.17],
    7: [-10.68, 6.17],
    8: [0.00, 12.33],
}


def get_rot_matrix(angle: float):
    """Returns a rotation matrix for the given angle in degrees."""

    return numpy.array(
        [
            [numpy.cos(-angle), -numpy.sin(-angle)],
            [numpy.sin(-angle), numpy.cos(-angle)],
        ]
    )


@lru_cache()
def get_coadd_images():
    """Gets the co-add images from the AGCAM data directory."""

    # Get the list of co-add images
    coadd_images: list[pathlib.Path] = []

    # Loop through the AGCAM data directory and get the co-add images
    dirs = os.listdir(AGCAM_ROOT)
    for dir in sorted(dirs):
        if not dir.startswith("6"):
            continue

        coadds_path = pathlib.Path(AGCAM_ROOT) / dir / "coadds"
        if not coadds_path.exists():
            continue

        sci_files = sorted(coadds_path.glob("lvm.sci.coadd_s*.fits"))
        for sci_file in sci_files:
            coadd_images.append(sci_file.absolute())

    return coadd_images


def collect_coadd_pointings():
    """Collects pointing data from the co-add images for the sci telescope."""

    print("Collecting co-add images ...")
    coadd_images = get_coadd_images()

    output_file = "coadd_pointings.parquet"
    if os.path.exists(output_file):
        df = polars.read_parquet(output_file)
    else:
        df = None

    new_data: list[dict] = []
    for coadd_image in track(coadd_images):
        mjd = int(coadd_image.parts[-3])

        exposure_match = re.match(r"^lvm\.sci\.coadd_s(\d+)\.fits$", coadd_image.name)
        if exposure_match is None:
            continue

        exposure_no = int(exposure_match.group(1))
        if df is not None and (df["exposure_no"] == exposure_no).any():
            continue

        spec_image_path = (
            pathlib.Path(SPECTRO_ROOT)
            / str(mjd)
            / f"sdR-s-b1-{exposure_no:08d}.fits.gz"
        )

        if not spec_image_path.exists():
            continue

        header = fits.getheader(coadd_image, 1)

        if not header["SOLVED"]:
            continue

        spectro_header = fits.getheader(spec_image_path, 0)

        tile_id = spectro_header["TILE_ID"]
        if tile_id is None or tile_id <= 0:
            continue

        if "SMJD" not in spectro_header:
            continue

        try:
            dpos = int(spectro_header["DPOS"])
        except ValueError:
            continue

        wcs = WCS(header)
        wcs_pointing = wcs.pixel_to_world(2500, 1000)
        wcs_ra = float(wcs_pointing.ra.deg)
        wcs_dec = float(wcs_pointing.dec.deg)

        field_ra = header["RAFIELD"]
        field_dec = header["DECFIELD"]
        field_pa = header["PAFIELD"]

        dpos_offset = DITHER_OFFSETS[dpos]
        rot_matrix = get_rot_matrix(field_pa or 0.0)

        dpos_offset_pa = numpy.dot(rot_matrix, dpos_offset) / 3600
        ra_offset = field_ra - dpos_offset_pa[0] / numpy.cos(numpy.radians(field_dec))
        dec_offset = field_dec - dpos_offset_pa[1]

        new_data.append(
            {
                "mjd": int(spectro_header["SMJD"]),
                "exposure_no": exposure_no,
                "tile_id": tile_id,
                "dpos": dpos,
                "dpos_ra": dpos_offset[0],
                "dpos_dec": dpos_offset[1],
                "dpos_rot_ra": dpos_offset_pa[0],
                "dpos_rot_dec": dpos_offset_pa[1],
                "field_ra": field_ra,
                "field_dec": field_dec,
                "field_pa": field_pa,
                "poscira": spectro_header.get("POSCIRA", None),
                "poscidec": spectro_header.get("POSCIDE", None),
                "poscipa": spectro_header.get("POSCIPA", None),
                "wcs_ra": wcs_ra,
                "wcs_dec": wcs_dec,
                "ra_offset": ra_offset,
                "dec_offset": dec_offset,
            }
        )

    new_data_df = polars.from_records(
        new_data,
        schema={
            "mjd": polars.Int32,
            "exposure_no": polars.Int32,
            "tile_id": polars.Int32,
            "dpos": polars.Int32,
            "dpos_ra": polars.Float64,
            "dpos_dec": polars.Float64,
            "dpos_rot_ra": polars.Float64,
            "dpos_rot_dec": polars.Float64,
            "field_ra": polars.Float64,
            "field_dec": polars.Float64,
            "field_pa": polars.Float64,
            "poscira": polars.Float64,
            "poscidec": polars.Float64,
            "poscipa": polars.Float64,
            "wcs_ra": polars.Float64,
            "wcs_dec": polars.Float64,
            "ra_offset": polars.Float64,
            "dec_offset": polars.Float64,
        },
    )

    if df is not None:
        df = polars.concat([df, new_data_df])
    else:
        df = new_data_df

    df = df.sort("exposure_no")
    df.write_parquet(output_file)

    return df
