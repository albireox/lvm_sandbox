#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-06-21
# @Filename: fibermap.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import pathlib

import pandas
from matplotlib import pyplot as plt

from sdsstools import read_yaml_file


FIBERMAP = pathlib.Path(__file__).parents[2] / "data" / "lvm_fiducial_fibermap.yaml"


def read_fibermap():
    """Reads and formats the fibermap."""

    fibermap_y = read_yaml_file(FIBERMAP)

    schema = fibermap_y["schema"]
    cols = [it["name"] for it in schema]

    fibers = pandas.DataFrame(fibermap_y["fibers"], columns=cols)

    return fibers


def plot_fibermap():
    """Plots the fibermap."""

    fibers = read_fibermap()
    plt.scatter(fibers.xpmm, fibers.ypmxx)
    plt.xlabel("xpmm")
    plt.ylabel("ypmxx")
    plt.show()


if __name__ == "__main__":
    # read_fibermap()
    plot_fibermap()
