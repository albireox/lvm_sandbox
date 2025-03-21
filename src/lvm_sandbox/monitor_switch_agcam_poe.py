#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2025-03-21
# @Filename: monitor_switch_agcam_poe.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import datetime
import pathlib
import sys
import time

from lvmopstools.devices.switch import get_poe_port_info


def monitor_switch_agcam_poe(dirname: str | pathlib.Path):
    """Monitors the PoE status of the AG camera ports in the main switches."""

    dirname = pathlib.Path(dirname)
    dirname.mkdir(exist_ok=True, parents=True)

    while True:
        try:
            data = get_poe_port_info()
        except Exception as ee:
            print(f"Failed to get PoE port info: {ee}")
            time.sleep(60)
            continue

        data_str = "\n".join(data.values())

        date = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
        filename = dirname / f"poe_agcam_ports_{date}.dat"

        with open(filename, "w") as f:
            f.write(data_str)

        time.sleep(60)


if __name__ == "__main__":
    monitor_switch_agcam_poe(sys.argv[1])
