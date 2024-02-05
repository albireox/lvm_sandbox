#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2024-01-31
# @Filename: plot_bias_electronic_bands.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib
import re

import nptyping as npt
import numpy
import seaborn
from astropy.io import fits
from matplotlib import pyplot as plt


seaborn.set_theme(context="paper", style="ticks", font_scale=1.2)

ARRAY_2D = npt.NDArray[npt.Shape["*,*"], npt.Float32]


def plot_bias_electronic_bands(subtract_overscan: bool = True):
    """Collect data and plot collapsed bias images."""

    PATH = "/data/spectro/lvm/60341/"

    for camera in ["r1", "r2", "r3", "b1", "b2", "b3", "z1", "z2", "z3"]:
        files = sorted(pathlib.Path(PATH).glob(f"sdR-s-{camera}-*.fits.gz"))

        data: list[ARRAY_2D] = []
        framenos: list[int] = []

        for file in files:
            filename = file.name
            if not (match := re.match(rf"sdR-s-{camera}-(\d+).fits.gz", filename)):
                continue

            file_data: ARRAY_2D = fits.getdata(file).astype(numpy.float32)
            file_data += numpy.random.uniform(0, 1e-3, file_data.shape)

            if subtract_overscan:
                bias_left = numpy.median(file_data[:, 2046:2060], axis=1)
                bias_right = numpy.median(file_data[:, 2060:2076], axis=1)

                file_data[:, :2040] -= bias_left[:, numpy.newaxis]
                file_data[:, 2040:] -= bias_right[:, numpy.newaxis]

            masked_data = numpy.ma.masked_where(file_data > 1500, file_data)
            data.append(numpy.ma.mean(masked_data, axis=1))

            framenos.append(int(match.group(1)))

        xx = numpy.arange(data[-1].size)

        edge_mask = numpy.ones(xx.size, dtype=bool)
        edge_mask[:20] = False
        edge_mask[-20:] = False

        with plt.ioff():
            fig, ax = plt.subplots()
            for ii, dd in enumerate(data):
                ax.plot(xx[edge_mask], dd[edge_mask], label=str(framenos[ii]))

            ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")
            fig.tight_layout()

            filename = f"bias_60341_{camera}"
            if subtract_overscan:
                filename += "_no_overscan"

            fig.savefig(f"/Users/gallegoj/Downloads/{filename}.pdf")
            plt.close(fig)
