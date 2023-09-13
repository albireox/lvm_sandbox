#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-09-13
# @Filename: dither_pattern.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Sequence

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from astropy.io import fits
from astropy.wcs import WCS
from lvmguider.coadd import create_global_coadd
from lvmguider.tools import get_frame_range


seaborn.set_theme(style="ticks", font="serif", font_scale=1.5)


AGCAM_DATA = "/data/agcam/60200"
DITHER_TO_FRAMES: dict[int, Sequence[int]] = {
    0: [160, 164],
    1: [168, 173],
    2: [176, 181],
    3: [184, 189],
    4: [191, 196],
    5: [198, 202],
    6: [205, 209],
    7: [215, 220],
    8: [222, 226],
}

DITHER_OFFSETS: dict[int, tuple[float, float]] = {
    0: (0.00, 0.00),
    1: (-10.68, 18.50),
    2: (10.68, 18.50),
    3: (-0.00, -12.33),
    4: (10.68, -6.17),
    5: (-10.68, -6.17),
    6: (10.68, 6.17),
    7: (-10.68, 6.17),
    8: (0.00, 12.33),
}


OUTPATH = pathlib.Path(__file__).parents[2] / "outputs" / "dither_pattern"
OUTPATH.mkdir(parents=True, exist_ok=True)

COADDS_PATH = coadd_outpath = str(OUTPATH / "coadds" / "coadd_dpos_{dpos}.fits")


def create_coadds():
    """Creates co-adds for each dither sequence."""

    for dpos, (frame1, frame2) in DITHER_TO_FRAMES.items():
        files = get_frame_range(AGCAM_DATA, "sci", frame1, frame2)

        create_global_coadd(
            files,
            "sci",
            outpath=COADDS_PATH.format(dpos=dpos),
            generate_qa=False,
            database_profile="mako",
        )


def plot_dither_position():
    """Prints a table of the relative dither positions and plots them."""

    df = pandas.DataFrame(
        {
            "dpos": pandas.Series(dtype="Int16"),
            "ra": pandas.Series(dtype="f8"),
            "dec": pandas.Series(dtype="f8"),
            "ra_off": pandas.Series(dtype="f8"),
            "dec_off": pandas.Series(dtype="f8"),
            "ra_off_expected": pandas.Series(dtype="f8"),
            "dec_off_expected": pandas.Series(dtype="f8"),
            "separation": pandas.Series(dtype="f8"),
        }
    )

    for dpos in range(9):
        coadd_path = pathlib.Path(COADDS_PATH.format(dpos=dpos))

        header = fits.getheader(coadd_path, 1)
        header["NAXIS"] = None

        wcs = WCS(header, naxis=2)
        ra, dec = wcs.pixel_to_world_values(2500, 1000)

        if dpos == 0:
            ra_off = 0.0
            dec_off = 0.0
        else:
            ra_0, dec_0 = df.loc[0][["ra", "dec"]]

            dec_off = (dec_0 - dec) * 3600

            cos_dec = numpy.cos(numpy.radians(dec))
            ra_off = (ra_0 - ra) * 3600 * cos_dec

        expected = numpy.array(DITHER_OFFSETS[dpos])
        separation = numpy.hypot(*(expected - numpy.array([ra_off, dec_off])))

        df.loc[dpos, :] = [dpos, ra, dec, ra_off, dec_off, *expected, separation]

    df = df.set_index("dpos")
    print(df)

    # Plot expected and measured positions.
    fig, ax = plt.subplots()

    ax.scatter(
        df.ra_off_expected,
        df.dec_off_expected,
        s=30,
        marker="x",  # type: ignore
        color="r",
        label="Expected",
    )

    ax.scatter(
        df.ra_off,
        df.dec_off,
        s=30,
        marker="o",  # type: ignore
        color="b",
        label="Measured",
    )

    for dpos, row in df.iterrows():
        ax.text(
            row.ra_off_expected + 0.5,
            row.dec_off_expected + 0.5,
            str(dpos),
            fontsize=13,
        )

    ax.legend()
    ax.set_title("MJD 60200 - AG sci 160-226")
    ax.set_xlabel("RA offset [arcsec]")
    ax.set_xlabel("Dec offset [arcsec]")

    fig.savefig(str(OUTPATH / "dither_pattern.pdf"))
    fig.savefig(str(OUTPATH / "dither_pattern.png"), dpi=300)


def plot_cutouts():
    """Plots cutouts of sources in the co-co-added image."""

    coadd_data = []
    for dpos in range(9):
        coadd_path = pathlib.Path(COADDS_PATH.format(dpos=dpos))
        coadd_data.append(fits.getdata(coadd_path, 2))

    coadd_data = numpy.sum(coadd_data, axis=0)
    fits.HDUList([fits.PrimaryHDU(data=coadd_data)]).writeto(
        OUTPATH / "coadd.fits",
        overwrite=True,
    )

    sources_path = COADDS_PATH.format(dpos=0).replace(".fits", "_sources.parquet")
    sources = pandas.read_parquet(sources_path)
    sources.sort_values("flux", ascending=False, inplace=True)
    sources = sources.loc[sources.camera == "east"]

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    nsource = 0
    for ii in range(4):
        for jj in range(4):
            ax = axes[ii][jj]

            while True:
                source = sources.iloc[nsource]
                y0 = int(source.y - 50)
                y1 = int(source.y + 50)
                x0 = int(source.x - 50)
                x1 = int(source.x + 50)

                nsource += 1
                if x0 < 0 or y0 < 0 or x1 > 1600 or y1 > 1100:
                    continue

                break

            cutout = coadd_data[
                int(source.y - 50) : int(source.y + 50 + 1),
                int(source.x - 50) : int(source.x + 50 + 1),
            ]
            ax.imshow(
                cutout,
                origin="lower",
                vmin=numpy.median(cutout) - 1 * cutout.std(),
                vmax=numpy.median(cutout) + 1 * cutout.std(),
            )

    for ax in axes.flatten():
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("equal")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0, hspace=0)

    fig.savefig(str(OUTPATH / "dither_pattern_cutouts.pdf"))
    fig.savefig(str(OUTPATH / "dither_pattern_cutouts.png"), dpi=300)


if __name__ == "__main__":
    # create_coadds()
    plot_dither_position()
    plot_cutouts()
