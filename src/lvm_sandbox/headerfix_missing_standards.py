#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-07-29
# @Filename: headerfix_missing_standards.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

from typing import Any

import polars
import yaml
from astropy.io import fits
from astropy.time import Time, TimeDelta


FIBRES = [
    "P1-2",
    "P1-1",
    "P1-12",
    "P1-11",
    "P1-10",
    "P1-9",
    "P1-8",
    "P1-7",
    "P1-6",
    "P1-5",
    "P1-4",
    "P1-3",
]


SCHEMA: list = [
    {
        "description": "the raw frame file root with * as wildcard",
        "dtype": "str",
        "name": "fileroot",
    },
    {
        "description": "the name of the header keyword to fix",
        "dtype": "str",
        "name": "keyword",
    },
    {
        "description": "the value of the header keyword to update",
        "dtype": "str",
        "name": "value",
    },
]


def get_files(mjd: int):
    """Returns the list of tiles that need fixing."""

    path = pathlib.Path(f"/data/spectro/{mjd}")

    # Use b1 for reference. This issue affects to all the cameras similarly.
    files_b1 = sorted(path.glob("sdR-*-b1-*.fits.gz"))

    to_fix: list[pathlib.Path] = []
    for file in files_b1:
        header = fits.getheader(file)
        if header["IMAGETYP"] != "object":
            continue
        if header["STD2FIB"] is not None:
            continue
        if header["EXPTIME"] < 600:
            continue
        if not header["TILE_ID"]:
            continue

        exp_no = header["EXPOSURE"]
        to_fix += path.glob(f"sdR-*{exp_no}.fits.gz")

    return to_fix


def get_db_data(file_: pathlib.Path):
    """Retrieves information about the exposure from the ``overhead`` table."""

    header = fits.getheader(file_)

    tile_id = header["TILE_ID"]
    dither_pos = header["DPOS"]

    int0 = Time(header["INTSTART"], format="isot")
    int1 = Time(header["INTEND"], format="isot")

    query = f"""
        SELECT * FROM overhead
        WHERE tile_id = {tile_id} AND dither_position = {dither_pos}
        AND start_time >= {int0.unix} AND end_time <= {int1.unix};"""

    data_prelim = polars.read_database_uri(
        query,
        uri="postgresql://sdss@lvm-webapp.lco.cl/lvmdb",
        engine="adbc",
    )

    # Get the observer.
    observer_id = data_prelim["observer_id"][0]

    # Run another query, this time esuring we get all the data.
    int0 -= TimeDelta(180, format="sec")

    query2 = f"""
        SELECT * FROM overhead
        WHERE tile_id = {tile_id} AND dither_position = {dither_pos}
        AND observer_id = {observer_id}
        AND start_time >= {int0.unix} AND end_time <= {int1.unix};"""

    data = polars.read_database_uri(
        query2,
        uri="postgresql://sdss@lvm-webapp.lco.cl/lvmdb",
        engine="adbc",
    )

    header_data: dict[int, dict[str, Any]] = {}

    t0 = data.filter(polars.col.stage == "acquisition:acquire")["end_time"][0]
    acquired = True
    for fibre_no in range(1, 13):
        fibre = FIBRES[fibre_no - 1]

        try:
            if fibre_no != 12:
                t1 = data.filter(
                    polars.col.stage == f"standards:standard-{fibre_no+1}-slew"
                )["start_time"][0]
            else:
                t1 = float(int1.unix)  # type: ignore

            if acquired:
                header_data[fibre_no] = {
                    "fibre": fibre,
                    "t0": Time(t0, format="unix").isot,
                    "t1": Time(t1, format="unix").isot,
                    "acquired": acquired,
                    "exptime": round(t1 - t0, 2),
                }
            else:
                header_data[fibre_no] = {
                    "fibre": fibre,
                    "t0": None,
                    "t1": None,
                    "acquired": acquired,
                    "exptime": None,
                }

        except IndexError:
            # Not all 12 standards were observed.
            header_data[fibre_no] = {
                "fibre": fibre,
                "t0": Time(t0, format="unix").isot,
                "t1": Time(int1.unix, format="unix").isot,
                "acquired": acquired,
                "exptime": round(float(int1.unix) - t0, 2),  # type: ignore
            }

            break

        if fibre_no != 12:
            new_acquire = data.filter(
                polars.col.stage == f"standards:standard-{fibre_no+1}-acquire"
            )
            t0 = new_acquire["end_time"][0]
            acquire_duration = new_acquire["duration"][0]
            if acquire_duration > 60:
                acquired = False
            else:
                acquired = True

    return header_data


def headerfix_missing_standards(mjd: int):
    """Generates ``headerfix`` files for missing standard entries.

    This problem occurred during the nights of MJD 60510 to 60513 as a result
    of a change in the standard iteration code which caused standards to be
    correctly observed but the metadata in the headers to be missing.

    """

    to_fix = get_files(mjd)
    to_fix_b1 = [file_ for file_ in to_fix if "-b1-" in file_.name]

    headerfix = {"schema": SCHEMA, "fixes": []}

    for file_ in to_fix_b1:
        data = get_db_data(file_)
        pattern = file_.name.replace("b1", "*").replace("-s-", "-*-")

        for std in range(1, 13):
            if std not in data:
                std_headerfix = {
                    f"STD{std}T0": None,
                    f"STD{std}T1": None,
                    f"STD{std}ACQ": None,
                    f"STD{std}EXP": None,
                    f"STD{std}FIB": FIBRES[std - 1],
                }
            else:
                std_headerfix = {
                    f"STD{std}T0": data[std]["t0"],
                    f"STD{std}T1": data[std]["t1"],
                    f"STD{std}ACQ": data[std]["acquired"],
                    f"STD{std}EXP": data[std]["exptime"],
                    f"STD{std}FIB": data[std]["fibre"],
                }

            for key, value in std_headerfix.items():
                headerfix["fixes"].append(
                    {
                        "fileroot": pattern,
                        "keyword": key,
                        "value": value,
                    }
                )

    yaml.dump(headerfix, open(f"lvmHdrFix-{mjd}.yaml", "w"))
