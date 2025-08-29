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

    data = hdul[0].data
    saturated_frac = numpy.sum(data >= 65500) / data.size

    return_dict = {
        "MJD": mjd,
        "exp_no": exp_no,
        "exp_type": exp_type,
        "exp_time": exp_time,
        "spec": spec,
        "saturated_frac": float(saturated_frac),
        "perc_98": float(numpy.percentile(data, 98)),
        "n_shifts": None,
        "y_shifts": None,
    }

    if ccd_temp > -80 or ln2_temp > -150:
        return None

    # Get the first column of the data array.
    col = data[:, 0].astype("f4")

    # Start with a simple test. If the standard deviation of the first column for
    # each quadrant is small, that means that there are no shifted pixels.
    col0 = col[0:2040]
    col1 = col[2040:]

    if col0.std() < 5 and col1.std() < 5:
        return None

    # Now identify the location of the shifted pixel. We first run a rolling mean to
    # smooth the data. This means the position we'll determine is only approximate,
    # but that's good enough.
    col = moving_average(col, n=10)

    # Now use a DBSCAN algorithm to identify the shifted pixels.
    labels = DBSCAN(eps=0.5, min_samples=10).fit_predict(col.reshape(-1, 1))

    # If all labels are -1 or all are 0, then no clusters were found.
    if numpy.all(labels == -1):
        return return_dict

    if numpy.any(labels == -1):
        last = 0
        for ii in range(len(labels)):
            if labels[ii] == -1:
                labels[ii] = last
            else:
                last = labels[ii]

    # Find indices in which the label changes
    ch_idx = numpy.where(labels[:-1] != labels[1:])[0]

    y_shifts: list[int] = []
    for i0 in ch_idx:
        # Skip the transition between quadrants
        if abs(i0 - 2040 + 9) < 20:
            continue
        y_shifts.append(int(i0) + 9)  # +9 because of the rolling mean

    return_dict["n_shifts"] = len(y_shifts)
    return_dict["y_shifts"] = y_shifts

    if len(y_shifts) == 0:
        return None

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
        "MJD": polars.Int32,
        "exp_no": polars.Int32,
        "spec": polars.String,
        "exp_type": polars.String,
        "exp_time": polars.Float32,
        "saturated_frac": polars.Float32,
        "perc_98": polars.Float32,
        "n_shifts": polars.Int16,
        "y_shifts": polars.List(polars.Int32),
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


def _process_spectro_file_ff(file: pathlib.Path):
    """Processes a single spectrograph file (FF method)."""

    hdul = fits.open(file)

    mjd = hdul[0].header["SMJD"]
    exp_no = hdul[0].header["EXPOSURE"]
    exp_type = hdul[0].header["IMAGETYP"]
    exp_time = hdul[0].header["EXPTIME"]
    spec = hdul[0].header["SPEC"]

    ccd_temp = hdul[0].header["CCDTEMP1"]
    ln2_temp = hdul[0].header["CCDTEMP2"]

    data = hdul[0].data

    return_dict = {
        "MJD": mjd,
        "exp_no": exp_no,
        "exp_type": exp_type,
        "exp_time": exp_time,
        "spec": spec,
        "n_shifts": None,
    }

    if ccd_temp > -80 or ln2_temp > -150:
        return None

    # Get the last row (top row in zX), convert it to a bytearray and reverse it.
    # If there are pixel shifts there will be at least 2 pixels with value 65535
    # (i.e., four bytes with values 0xFFFFFFFF). We count how many of those there are
    # and that is the number of shifts (each shift is two pixels).
    last_row = data[-1, :].tobytes()[::-1]
    n_shifts = 0

    idx = 0
    while True:
        test = last_row[idx : idx + 4]
        if test == b"\xff\xff\xff\xff":
            n_shifts += 1
            idx += 4
        else:
            break

    if n_shifts == 0:
        return None

    return_dict["n_shifts"] = n_shifts
    return return_dict


def _process_spectro_mjd_ff(
    dir_path: pathlib.Path,
    skip_mjds: list[int] = [],
) -> tuple[int, list[dict] | None]:
    """Processes all spectrograph files in a given MJD directory (FF method)."""

    mjd = int(dir_path.name)
    if mjd in skip_mjds or mjd < 60200:
        return (mjd, None)

    data_files = sorted(dir_path.glob("sdR-s-z*-*.fits.gz"))
    if len(data_files) == 0:
        return (mjd, None)

    results = []
    for file in data_files:
        try:
            data = _process_spectro_file_ff(file)
        except Exception as e:
            rprint(f"[red]Error processing {file}: {e}[/red]")
            continue

        if data:
            results.append(data)

    return (mjd, results)


def generate_pixel_shift_list_ff(n_cpus: int = 10, skip_processed: bool = True):
    """Processes all LVM raw images to identify pixel shifts using the FF method.

    Parameters
    ----------
    n_cpus
        The number of CPUs to use for processing.
    skip_processed
        If :obj:`True`, skip MJDs that have already been processed.

    Returns
    -------
    pixel_shift
        A Polars data frame with the list of pixel shifts. The data is also
        written to ``data/pixel_shifts_ff.parquet``.

    """

    DATA_DIR = pathlib.Path(__file__).parents[2] / "data"
    DF_PATH = DATA_DIR / "pixel_shifts_ff.parquet"

    SPECTRO_DIR = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/lvm/lco/"

    SCHEMA = {
        "MJD": polars.Int32,
        "exp_no": polars.Int32,
        "spec": polars.String,
        "exp_type": polars.String,
        "exp_time": polars.Float32,
        "n_shifts": polars.Int16,
    }

    dirs = list(sorted(pathlib.Path(SPECTRO_DIR).glob("6*")))

    if DF_PATH.exists() and not skip_processed:
        df = polars.read_parquet(DF_PATH)
    else:
        df = polars.DataFrame(None, schema=SCHEMA)

    mjds = df["MJD"].unique().to_list()

    _partial_process_mjd = functools.partial(_process_spectro_mjd_ff, skip_mjds=mjds)

    with multiprocessing.Pool(processes=n_cpus) as pool:
        for mjd, data in pool.imap_unordered(_partial_process_mjd, dirs):
            if data and len(data) > 0:
                rprint(f"Adding results for MJD [blue]{mjd}[/blue]")

                df_mjd = polars.DataFrame(data, schema=SCHEMA)
                df = polars.concat([df, df_mjd], how="vertical")
                df.write_parquet(DF_PATH)

        return df


def fits_to_buffer(
    exp_no: int,
    spec: Literal["sp1", "sp2", "sp3"],
    root_path: str | pathlib.Path,
    prefix: bytearray | None = None,
    replace_position: tuple[int, int] | None = None,
):
    """Reconstructs the image buffer for a given exposure number."""

    PIXELS = (4080, 4120)

    if isinstance(root_path, str):
        root_path = pathlib.Path(root_path)

    buffer_array = numpy.zeros((PIXELS[0], PIXELS[1] * 3), dtype=numpy.int16)

    for ii, camera in enumerate(["r", "b", "z"]):
        cam_spec = f"{camera}{spec[-1]}"

        file_path = root_path / f"sdR-s-{cam_spec}-{exp_no:08d}.fits.gz"
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        data = fits.getdata(file_path).astype("<u2")
        buffer_array[:, ii * PIXELS[1] : (ii + 1) * PIXELS[1]] = data

    buffer_ba = bytearray(buffer_array.tobytes())
    if (rem := len(buffer_ba) % 1024) != 0:
        buffer_ba += b"\0" * (1024 - rem)

    assert len(buffer_ba) % 1024 == 0

    if prefix is not None:
        if replace_position:
            i0 = replace_position[1] * 4120 * 2 + replace_position[0] * 2
            buffer_ba[i0 : i0 + 4] = prefix

        buffer_chunks: list[bytearray] = []
        ii = 0
        while True:
            start = ii * 1024
            end = start + 1024
            chunk = buffer_ba[start:end]
            if len(chunk) == 0:
                break
            buffer_chunks.append(bytearray(prefix) + chunk)
            ii += 1

        buffer_ba = b"".join(buffer_chunks)

    return buffer_ba


def reconstruct_from_buffer(
    buffer_file: str | pathlib.Path,
    exp_no: int,
    spec: Literal["sp1", "sp2", "sp3"],
    root_path: str | pathlib.Path | None = None,
    compare_files: bool = False,
    has_prefix: bool = True,
    remove_prefix_mode: Literal["replace", "slice"] = "replace",
    fits_file_prefix: str = "reconstructed-",
):
    """Reconstruct FITS images from a buffer byte array file."""

    buffer_file = pathlib.Path(buffer_file)
    if not buffer_file.exists():
        raise FileNotFoundError(f"Buffer file not found: {buffer_file}")

    with open(buffer_file, "rb") as f:
        buffer_data = f.read()

    if has_prefix:
        prefix = buffer_data[:4]
        if remove_prefix_mode == "replace":
            buffer_data = buffer_data.replace(prefix, b"")
        elif remove_prefix_mode == "slice":
            raise NotImplementedError("Slicing prefix removal is not implemented.")

    N_PIXELS = 4080 * 4120 * 3
    buffer_data = buffer_data[: N_PIXELS * 2]

    data = numpy.frombuffer(buffer_data, dtype="<u2").reshape((4080, 4120 * 3))

    for ii, camera in enumerate(["r", "b", "z"]):
        cam_spec = f"{camera}{spec[-1]}"

        camera_data = data[:, ii * 4120 : (ii + 1) * 4120]
        assert camera_data.shape == (4080, 4120)

        if compare_files:
            if root_path is None:
                raise ValueError("Root path must be specified for comparison.")

            root_path = pathlib.Path(root_path)
            orig_path = root_path / f"sdR-s-{cam_spec}-{exp_no:08d}.fits.gz"
            if not orig_path.exists():
                raise FileNotFoundError(f"Original file not found: {orig_path}")

            orig_data = fits.getdata(orig_path).astype("<u2")
            assert orig_data.shape == (4080, 4120)

            if not numpy.array_equal(camera_data, orig_data):
                rprint(f"[red]Data mismatch found for {exp_no}-{cam_spec}.[/]")

        buffer_root = buffer_file.parent
        camera_file = buffer_root / f"{fits_file_prefix}{cam_spec}-{exp_no:08d}.fits"

        hdul = fits.PrimaryHDU(data=camera_data)
        hdul.writeto(camera_file, overwrite=True)


def collate_cameras(
    exposure: str | pathlib.Path,
    output: str | pathlib.Path | None = None,
):
    """Collates the three camera FITS files into a single FITS file.

    Parameters
    ----------
    exposure
        The exposure path (any of the three cameras, the other two are derived
        from the file name).
    output
        The output file path.

    """

    exposure = pathlib.Path(exposure)
    if not exposure.exists():
        raise FileNotFoundError(f"Exposure file not found: {exposure}")

    root_path = exposure.parent

    sample_name = exposure.name
    sample_match = re.match(r"sdR-s-([rbz][123])-(\d{8})\.fits.gz", sample_name)
    if not sample_match:
        raise ValueError(f"Invalid exposure file name: {sample_name}")

    sample_camera = sample_match.group(1)
    sample_exposure = sample_match.group(2)
    spec = sample_camera[-1]

    collated = numpy.zeros((4080, 4120 * 3), dtype=numpy.uint16)
    for ii, camera in enumerate(["r", "b", "z"]):
        cam_name = sample_name.replace(sample_camera, f"{camera}{spec}")

        cam_data = fits.getdata(root_path / cam_name).astype("<u2")
        cam_header = fits.getheader(root_path / cam_name)

        collated[:, ii * 4120 : (ii + 1) * 4120] = cam_data

    hdul = fits.PrimaryHDU(data=collated, header=cam_header)

    if not output:
        output = root_path / f"sdR-s-sp{spec}-collated-{sample_exposure}.fits.gz"

    hdul.writeto(output, overwrite=True)


def fix_collated_image(image: str | pathlib.Path, shifts: list[tuple[int, int]]):
    """Fixes the collated image by applying a shift."""

    image = pathlib.Path(image)
    if not image.exists():
        raise FileNotFoundError(f"Image file not found: {image}")

    data, header = fits.getdata(image, header=True)

    data = data.flatten()
    new_data = numpy.zeros(len(data), dtype=data.dtype)
    for _, (xx, yy) in enumerate(shifts):
        idx = yy * 4120 * 3 + xx
        new_data[0:idx] = data[0:idx]
        new_data[idx + 2 :] = data[idx:-2]

    new_image = image.with_name(image.name.replace(".fits.gz", "_fixed.fits.gz"))
    hdul = fits.PrimaryHDU(data=new_data.reshape((4080, 4120 * 3)), header=header)
    hdul.writeto(new_image, overwrite=True)
