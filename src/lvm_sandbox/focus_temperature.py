#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-04-08
# @Filename: focus_temperature.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib

import numpy
import numpy.typing as npt
import polars
import scipy
import seaborn
from astropy.io import fits
from matplotlib import pyplot as plt
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from scipy.interpolate import UnivariateSpline


AGCAM_ROOT = "/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/agcam/lco"
OUTPATH = pathlib.Path(__file__).parent / "../../outputs/focus_sweeps_2025"


def _process_mjd(dir: pathlib.Path):
    """Processes an AGCAM MJD directory and collects focus sweep data."""

    data: list[dict] = []

    mjd = dir.parts[-1]

    for telescope in ["sci", "spec", "skye", "skyw"]:
        n_sweep = 0
        was_sweep = False

        files = sorted(dir.glob(f"lvm.{telescope}.agcam.east_*.fits"))

        for file in files:
            image_no = int(file.stem.split("_")[-1])

            guider_path = dir / f"lvm.{telescope}.guider_{image_no:08d}.fits"
            if guider_path.exists():
                continue

            with fits.open(file) as hdul:
                if "RAW" not in hdul or "PROC" not in hdul:
                    continue

                raw_header = hdul["RAW"].header
                proc_header = hdul["PROC"].header

                if "ISFSWEEP" not in proc_header or not proc_header["ISFSWEEP"]:
                    was_sweep = False
                    continue

                if was_sweep is False:
                    n_sweep += 1
                    was_sweep = True

                data.append(
                    {
                        "mjd": int(mjd),
                        "image_no": image_no,
                        "telescope": raw_header["TELESCOP"],
                        "camname": raw_header["CAMNAME"],
                        "bentempi": raw_header["BENTEMPI"],
                        "bentempo": raw_header["BENTEMPO"],
                        "benhumi": raw_header["BENHUMI"],
                        "benhumo": raw_header["BENHUMO"],
                        "focusdt": raw_header["FOCUSDT"],
                        "airmass": raw_header["AIRMASS"],
                        "fwhm": proc_header["FWHM"],
                        "n_sweep": n_sweep,
                    }
                )

                # Check if the "west" version of the file exists and if so,
                # add that files as well.
                ag_west = pathlib.Path(str(file).replace(".east_", ".west_"))
                if ag_west.exists():
                    with fits.open(ag_west) as hdul_west:
                        raw_header_west = hdul_west["RAW"].header
                        proc_header_west = hdul_west["PROC"].header

                        data.append(
                            {
                                "mjd": int(mjd),
                                "image_no": image_no,
                                "telescope": raw_header_west["TELESCOP"],
                                "camname": raw_header_west["CAMNAME"],
                                "bentempi": raw_header_west["BENTEMPI"],
                                "bentempo": raw_header_west["BENTEMPO"],
                                "benhumi": raw_header_west["BENHUMI"],
                                "benhumo": raw_header_west["BENHUMO"],
                                "focusdt": raw_header_west["FOCUSDT"],
                                "airmass": raw_header_west["AIRMASS"],
                                "fwhm": proc_header_west["FWHM"],
                                "n_sweep": n_sweep,
                            }
                        )

    return data


def get_dirs(dirs: list[pathlib.Path], min_mjd: int | None, max_mjd: int | None):
    """Filters directories based on MJD range."""

    if min_mjd is not None:
        dirs = [d for d in dirs if int(d.parts[-1]) >= min_mjd]

    if max_mjd is not None:
        dirs = [d for d in dirs if int(d.parts[-1]) <= max_mjd]

    return dirs


def collect_agcam_data():
    """Collects focus sweep data from AGCAM files."""

    agcam_path = pathlib.Path(AGCAM_ROOT)

    dirs = sorted(agcam_path.glob("6*"))
    dirs = get_dirs(dirs, min_mjd=60350, max_mjd=60900)

    data: list[dict] = []

    with Progress(
        TextColumn("[yellow]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        expand=True,
        refresh_per_second=1,
    ) as progress:
        tid = progress.add_task("Processing AGCAM data ...", total=len(dirs))
        with multiprocessing.Pool(processes=6) as pool:
            data_imap = pool.imap(_process_mjd, dirs)
            for dd in data_imap:
                progress.update(tid, advance=1)
                data += dd

    df = polars.DataFrame(
        data,
        schema={
            "mjd": polars.Int32,
            "image_no": polars.Int32,
            "telescope": polars.String,
            "camname": polars.String,
            "bentempi": polars.Float32,
            "bentempo": polars.Float32,
            "benhumi": polars.Float32,
            "benhumo": polars.Float32,
            "focusdt": polars.Float32,
            "airmass": polars.Float32,
            "fwhm": polars.Float32,
            "n_sweep": polars.Int32,
        },
    )

    return df.sort(["mjd", "telescope", "image_no"])


def fit_data(telescope: str, data: polars.DataFrame):
    """Fits focus-temperature."""

    avg = data.group_by("frame.focus_position", as_index=False).mean()

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    for sensor in [1, 2]:
        if f"sensor{sensor}.temperature" in data:
            slope, intercept, r, p, sterr = scipy.stats.linregress(
                x=data[f"sensor{sensor}.temperature"],
                y=data["frame.focus_position"],
            )

            print(f"{telescope}, sensor{sensor}")
            print(
                f"slope={slope:.3f}, intercept={intercept:.3f}, "
                f"r={r:.3f}, p={p}, sterr={sterr:.6f}"
            )

        if f"sensor{sensor}.temperature" in data:
            seaborn.scatterplot(
                data=data,
                x=f"sensor{sensor}.temperature",
                y="frame.focus_position",
                ax=axes[0],
            )

            seaborn.regplot(
                data=avg.to_pandas(),
                x=f"sensor{sensor}.temperature",
                y="frame.focus_position",
                ax=axes[1],
            )

    fig.savefig(str(OUTPATH / f"focus_temperature_{telescope}.pdf"))

    print()


def fit_spline(x: npt.ArrayLike, y: npt.ArrayLike, w: npt.ArrayLike | None = None):
    """Fits a spline to data."""

    spl = UnivariateSpline(x, y, w=w)

    corr_matrix = numpy.corrcoef(y, spl(x))  # type: ignore
    R2 = corr_matrix[0, 1] ** 2

    return spl, R2


def _do_sweep(data: polars.DataFrame):
    """Performs a focus sweep on the data."""

    data = data.select(
        polars.col(
            "mjd",
            "n_sweep",
            "bentempi",
            "bentempo",
            "focusdt",
            "fwhm",
        )
    )

    schema = {
        "mjd": polars.Int32,
        "n_sweep": polars.Int32,
        "bentempi": polars.Float32,
        "bentempo": polars.Float32,
        "focusdt": polars.Float32,
        "fwhm": polars.Float32,
        "valid": polars.Boolean,
    }
    return_df = polars.DataFrame(
        {
            "mjd": data["mjd"][0],
            "n_sweep": data["n_sweep"][0],
            "bentempi": None,
            "bentempo": None,
            "focusdt": None,
            "fwhm": None,
            "valid": False,
        },
        schema=schema,
    )

    data = data.group_by("focusdt").mean()

    if data.height < 5:
        return return_df

    data = data.sort("focusdt")

    spl, r2 = fit_spline(
        x=data["focusdt"].to_numpy(),
        y=data["fwhm"].to_numpy(),
        w=numpy.ones(data.height),
    )

    if r2 < 0.8:
        return return_df

    x_refine = numpy.arange(data["focusdt"].min(), data["focusdt"].max(), 0.01)
    y_refine = spl(x_refine)

    arg_min = numpy.argmin(y_refine)
    xmin = x_refine[arg_min]
    ymin = y_refine[arg_min]

    return_df = polars.DataFrame(
        {
            "mjd": int(data["mjd"][0]),
            "n_sweep": int(data["n_sweep"][0]),
            "bentempi": data["bentempi"].mean(),
            "bentempo": data["bentempo"].mean(),
            "focusdt": xmin,
            "fwhm": ymin,
            "valid": True,
        },
        schema=schema,
    )

    if numpy.isnan(ymin) or ymin < 0:  # type:ignore
        return_df[0, "valid"] = False

    return return_df


def process_focus_data(data: polars.DataFrame | str | pathlib.Path):
    """Processes focus data and determines the focus-temperature relation."""

    if isinstance(data, (str, pathlib.Path)):
        data = polars.read_parquet(data)

    tels_data = []
    for telescope in ["sci", "spec", "skye", "skyw"]:
        tel_data = data.filter(polars.col.telescope == telescope)

        tel_focus_data = tel_data.group_by("mjd", "n_sweep").map_groups(_do_sweep)
        tel_focus_data = tel_focus_data.with_columns(telescope=polars.lit(telescope))

        tels_data.append(tel_focus_data)

    focus_data = polars.concat(tels_data)

    return focus_data.select(
        "telescope",
        "mjd",
        "n_sweep",
        "bentempi",
        "bentempo",
        "focusdt",
        "fwhm",
        "valid",
    ).sort("telescope", "mjd", "n_sweep")


def plot_focus_data(data: polars.DataFrame | str | pathlib.Path):
    """Plots focus-temperature data."""

    seaborn.set_theme(
        style="whitegrid",
        font_scale=1.2,
        palette="deep",
        color_codes=True,
    )
    plt.ioff()

    if isinstance(data, (str, pathlib.Path)):
        data = polars.read_parquet(data)

    for telescope in ["sci", "spec", "skye", "skyw"]:
        tel_data = data.filter(
            polars.col.telescope == telescope,
            polars.col.valid,
            polars.col.mjd <= 60980,
        )

        tel_data_new = data.filter(
            polars.col.telescope == telescope,
            polars.col.valid,
            polars.col.mjd >= 60986,
        )

        has_bentempo = (
            tel_data["bentempo"].is_not_null().any()
            and (tel_data["bentempo"] > -999).any()
        )

        fig, axes = plt.subplots(2 if has_bentempo else 1, 1, figsize=(18, 10))
        ax = axes[0] if has_bentempo else axes

        seaborn.scatterplot(
            data=tel_data.to_pandas(),
            x="bentempi",
            y="focusdt",
            ax=ax,
            zorder=10,
        )

        seaborn.scatterplot(
            data=tel_data_new.to_pandas(),
            x="bentempi",
            y="focusdt",
            ax=ax,
            color="r",
            zorder=20,
        )

        coeffs = numpy.polyfit(tel_data["bentempi"], tel_data["focusdt"], deg=1)
        fit = numpy.poly1d(coeffs)

        coeffs_new = numpy.polyfit(
            tel_data_new["bentempi"], tel_data_new["focusdt"], deg=1
        )
        fit_new = numpy.poly1d(coeffs_new)

        xx = numpy.arange(-5, 25, 0.05)
        yy = fit(xx)
        yy_new = fit_new(xx)

        fit_plot = ax.plot(xx, yy, c="k")
        fit_plot_new = ax.plot(xx, yy_new, c="r", linestyle="--")

        ax.legend(
            fit_plot + fit_plot_new,
            [
                f"y={coeffs[0]:.3f}x + {coeffs[1]:.3f}",
                f"y={coeffs_new[0]:.3f}x + {coeffs_new[1]:.3f}",
            ],
        )

        if has_bentempo:
            seaborn.scatterplot(
                data=tel_data.to_pandas(),
                x="bentempo",
                y="focusdt",
                ax=axes[1],
            )
            seaborn.scatterplot(
                data=tel_data_new.to_pandas(),
                x="bentempo",
                y="focusdt",
                ax=axes[1],
                zorder=20,
                color="r",
            )

        OUTPATH.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(OUTPATH / f"focus_{telescope}.pdf"))

        plt.close(fig)
