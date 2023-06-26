#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-26
# @Filename: reprocess_ag_astrometry.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import os
import pathlib
import shutil
import subprocess
import warnings

import pandas
import seaborn
from astropy.io import fits
from astropy.wcs import WCS
from lvmguider.transformations import rot_shift_locs, solve_from_files
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle


seaborn.set_theme(style="ticks")


def extract_sources_and_wcs(file: str) -> tuple[pandas.DataFrame, WCS] | None:
    """Returns the sources and WCS from an image"""

    try:
        hdus = fits.open(str(file))
        wcs = WCS(hdus[1].header)
        sources = pandas.DataFrame(hdus[2].data)
    except Exception:
        return None

    return (sources, wcs)


def reprocess_wcs(files: list[pathlib.Path]):
    """Reprocesses the astrometric solution of a frame."""

    # Determine the path of the proc- image for this pair of images.
    dirname = os.path.dirname(str(files[0]))
    basename = os.path.basename(files[0])
    proc_image = os.path.join(dirname, "proc-" + basename)
    proc_image = proc_image.replace(".east", "").replace(".west", "")

    tel = basename.split(".")[1]

    # Get sources and previous WCS from images.
    data = extract_sources_and_wcs(proc_image)
    if data is None:
        return None

    sources, wcs_pre = data

    # Calculate new astrometric solution from files. This calls rot_shift_locs().
    try:
        wcs_new = solve_from_files([str(file_) for file_ in files], tel)
    except RuntimeError:
        return None

    if wcs_new is None:
        return None

    # Recalculate x/y for each camera to be in the master frame.
    sources_e = sources.loc[sources.camera == "east"]
    if len(sources_e) > 0:
        locs_e, _ = rot_shift_locs(tel + "-e", sources_e.loc[:, ["x", "y"]].values)
        sources_e.loc[:, ["x", "y"]] = locs_e

    sources_w = sources.loc[sources.camera == "west"]
    if len(sources_w) > 0:
        locs_w, _ = rot_shift_locs(tel + "-w", sources_w.loc[:, ["x", "y"]].values)
        sources_w.loc[:, ["x", "y"]] = locs_w

    return proc_image, wcs_pre, wcs_new, pandas.concat([sources_e, sources_w])


def plot_pre_new(
    proc_image: str,
    wcs_pre: WCS,
    wcs_new: WCS,
    sources: pandas.DataFrame,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Plot sources with the previous vs the new WCS."""

    # Determine the path for the plot by replacing the proc- file's extension with .png
    OUTDIR = pathlib.Path(".").absolute().parents[1] / "outputs" / "agcam"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plot_file = os.path.basename(proc_image).replace(".fits", ".png")

    # Calculate the astrometric position of each source with the old and new solutions.
    astro_pre = wcs_pre.pixel_to_world(sources.loc[:, "x"], sources.loc[:, "y"])
    astro_new = wcs_new.pixel_to_world(sources.loc[:, "x"], sources.loc[:, "y"])

    # Calculate average separation between the two solutions.
    sep = astro_pre.separation(astro_new).arcsec.mean()

    fig, ax = plt.subplots()

    ax.scatter(
        astro_pre.ra.deg,
        astro_pre.dec.deg,
        marker=MarkerStyle("o"),
        color="r",
        label="On sky",
    )
    ax.scatter(
        astro_new.ra.deg,
        astro_new.dec.deg,
        marker=MarkerStyle("x"),
        color="b",
        label="Reprocess",
    )

    ax.set_title(f"{os.path.basename(proc_image)} - Sep: {sep:.1f} arcsec")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")

    # If provided, use xlim, ylim.
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)

    ax.legend()

    fig.savefig(str(OUTDIR / plot_file))

    plt.close("all")

    return str(OUTDIR / plot_file), ax.get_xlim(), ax.get_ylim()


def create_movie(plotfiles: list[str]):
    """Creates a movie."""

    # Uses the path of the first image, replacing the extension to .mp4.
    outpath = plotfiles[0].replace(".png", ".mp4")

    tmp_dir = pathlib.Path("/tmp/ag_movie")
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir))
    tmp_dir.mkdir()

    # Copy PNGs to a temporary file, renaming them in sequence so we can use ffmpeg's
    # -start_number and -vframes.
    for ii, file_ in enumerate(plotfiles):
        shutil.copyfile(file_, os.path.join(str(tmp_dir), f"frame_{ii+1}.png"))

    nframes = len(plotfiles)
    subprocess.run(
        "ffmpeg -r 1 -f image2 -s 1920x1080 -start_number 1 "
        f"-i frame_%d.png -vframes {nframes} "
        "-vcodec libx264 -crf 25  -pix_fmt yuv420p movie.mp4",
        cwd=str(tmp_dir),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Copy movie back to outputs dir.
    shutil.copyfile(os.path.join(str(tmp_dir), "movie.mp4"), outpath)


def reprocess_files(mjd: int, telescope: str, frame0: int, frame1: int):
    """Reprocesses a series of files."""

    path = pathlib.Path(f"/data/agcam/{mjd}")
    plotfiles = []
    xlim = ylim = None
    for frame in range(frame0, frame1 + 1):
        files = list(path.glob(f"lvm.{telescope}.agcam.*_{frame:08d}.fits"))

        if len(files) == 0:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = reprocess_wcs(files)

        if result is None:
            continue

        proc_image, wcs_pre, wcs_new, sources = result
        plotfile, xlim, ylim = plot_pre_new(
            proc_image,
            wcs_pre,
            wcs_new,
            sources,
            xlim=xlim,
            ylim=ylim,
        )
        plotfiles.append(plotfile)

    create_movie(plotfiles)


if __name__ == "__main__":
    MJD = 60116

    frame0 = 114
    frame1 = 139
    telescope = "spec"
    reprocess_files(MJD, telescope, frame0, frame1)

    frame0 = 15
    frame1 = 34
    telescope = "sci"
    reprocess_files(MJD, telescope, frame0, frame1)

    frame0 = 35
    frame1 = 107
    telescope = "sci"
    reprocess_files(MJD, telescope, frame0, frame1)

    frame0 = 35
    frame1 = 107
    telescope = "sci"
    reprocess_files(MJD, telescope, frame0, frame1)

    frame0 = 154
    frame1 = 177
    telescope = "sci"
    reprocess_files(MJD, telescope, frame0, frame1)
