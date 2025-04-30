#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-04-29
# @Filename: fix_posci.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import os
import pathlib

import polars
from yaml import safe_dump, safe_load


MAX_MJD: int = 60791

SCHEMA = [
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


def get_hdrfix_file(mjd: int) -> pathlib.Path:
    """Get the hdrfix file for the given MJD."""

    lvmcore_dir = os.environ.get("LVMCORE_DIR", None)
    if lvmcore_dir is None:
        raise ValueError("LVMCORE_DIR environment variable not set.")

    return (
        pathlib.Path(lvmcore_dir)
        / "hdrfix"
        / f"{mjd:05d}"
        / f"lvmHdrFix-{mjd:05d}.yaml"
    )


def hdrfix_posci():
    """Generate lvmHdrFix files to fix the POSCI header."""

    coadd_data = polars.read_parquet(
        pathlib.Path(__file__).parent
        / "../../outputs/pointing_data/coadd_pointings.parquet"
    )

    for row in coadd_data.iter_rows(named=True):
        mjd = row["mjd"]

        if mjd >= MAX_MJD:
            continue

        if row["dpos"] == 0:
            continue

        hdrfix_path = get_hdrfix_file(mjd)

        if not hdrfix_path.exists():
            hdrfix_path.parent.mkdir(parents=True, exist_ok=True)
            with open(hdrfix_path, "w") as file:
                initial_data = {"schema": SCHEMA, "fixes": []}
                file.write(safe_dump(initial_data, indent=2, sort_keys=False))

        hdrfix = safe_load(hdrfix_path.read_text())
        fileroot = f"sdR-*-*-{row['exposure_no']:08d}"

        for key in ["ra", "dec"]:
            if key == "ra":
                hdr = "POSCIRA"
                value = row["posci_ra"]
            else:
                hdr = "POSCIDE"
                value = row["posci_dec"]

            processed = False
            for idx, fix in enumerate(hdrfix["fixes"]):
                if fix["fileroot"] == fileroot and fix["keyword"] == hdr:
                    hdrfix["fixes"][idx]["value"] = value
                    processed = True
                    break

            if not processed:
                new_fix = {"fileroot": fileroot, "keyword": hdr, "value": value}
                hdrfix["fixes"].append(new_fix)

        with open(hdrfix_path, "w") as file:
            file.write(safe_dump(hdrfix, indent=2, sort_keys=False))

    return
