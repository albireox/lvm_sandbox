#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-07-09
# @Filename: pointing_model.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import asyncio
import pathlib

import numpy
import pandas
from astropy.coordinates import (
    EarthLocation,
    SkyCoord,
    uniform_spherical_random_surface,
)
from astropy.time import Time
from gort import Gort
from numpy.random import choice


def get_random_sample(
    n_points: int,
    ra_range: tuple[float, float] | None = None,
    dec_range: tuple[float, float] | None = None,
):
    """Provides a random sample of RA/Dec points on the surface of a sphere.

    Parameters
    ----------
    n_points
        Number of points to return. This number is ensured even if ``ra_range``
        or ``dec_range`` are provided.
    ra_range
        The range of RA to which to limit the sample, in degrees.
    dec_range
        The range of Dec to which to limit the sample, in degrees.

    Returns
    -------
    coordinates
        A 2D array with the RA/Dec coordinates of the points on the sphere.

    """

    points = numpy.zeros((0, 2), dtype=numpy.float64)

    lco = EarthLocation.of_site("Las Campanas Observatory")
    now = Time.now()

    while True:
        sph_points = uniform_spherical_random_surface(n_points)

        radec = SkyCoord(ra=sph_points.lon.deg, dec=sph_points.lat.deg, unit="deg")
        radec.location = lco
        radec.obstime = now

        altaz = radec.transform_to("altaz")
        sph_points = sph_points[altaz.alt.deg > 30]

        radec = numpy.array(
            [sph_points.lon.deg, sph_points.lat.deg],
            dtype=numpy.float64,
        ).T

        if ra_range is not None:
            radec = radec[(radec[:, 0] > ra_range[0]) & (radec[:, 0] < ra_range[1])]
        if dec_range is not None:
            radec = radec[(radec[:, 1] > dec_range[0]) & (radec[:, 1] < dec_range[1])]

        points = numpy.vstack((points, radec))

        if points.shape[0] >= n_points:
            idx = numpy.arange(points.shape[0])
            sel = choice(idx, n_points)
            return points[sel]


async def get_offset(
    gort: Gort,
    telescope: str,
    ra: float,
    dec: float,
    exposure_time: float = 5,
):
    """Determines the offset between a pointing an the measured coordinates.

    Parameters
    ----------
    gort
        The instance of `.Gort` used to command the telescope.
    telescope
        The telescope: ``'sci'``, ``'spec'``, etc.
    ra,dec
        The RA/Dec coordinates of the field to observe.
    exposure_time
        The exposure time.

    Returns
    -------
    offset
        A dictionary with the commanded and measured RA/Dec and the offset,
        or None if the measurement failed.

    """

    await gort.telescopes[telescope].goto_coordinates(ra=ra, dec=dec)

    replies = await gort.guiders[telescope].actor.commands.guide(
        reply_callback=gort.guiders[telescope].print_reply,
        ra=ra,
        dec=dec,
        exposure_time=exposure_time,
        one=True,
        apply_corrections=False,
    )

    replies = replies.flatten()
    if "measured_pointing" not in replies:
        return None

    return_dict = {"commanded_ra": ra, "commanded_dec": dec, "telescope": telescope}
    return_dict["seqno"] = replies["frame"]["seqno"]
    measured_poining = replies["measured_pointing"]
    return_dict["measured_ra"] = measured_poining["ra"]
    return_dict["measured_dec"] = measured_poining["dec"]
    return_dict["offset_ra"] = measured_poining["radec_offset"][0]
    return_dict["offset_dec"] = measured_poining["radec_offset"][1]
    return_dict["offset_ax0"] = measured_poining["motax_offset"][0]
    return_dict["offset_ax1"] = measured_poining["motax_offset"][1]
    return_dict["separation"] = measured_poining["separation"]

    return return_dict


async def pointing_model(
    output_file: str | pathlib.Path,
    n_points: int,
    ra_range: tuple[float, float],
    dec_range: tuple[float, float],
    telescopes: list[str] = ["sci", "spec", "skye", "skyw"],
):
    """Iterates over a series of points on the sky measuring offsets.

    Parameters
    ----------
    output_file
        The HD5 file where to save the resulting table.
    n_points
        Number of points on the sky to measure.
    ra_range
        The range of RA to which to limit the sample, in degrees.
    dec_range
        The range of Dec to which to limit the sample, in degrees.
    telescopes
        The list of telescopes to expose.

    """

    gort = await Gort(verbosity="debug").init()

    points = get_random_sample(n_points, ra_range=ra_range, dec_range=dec_range)

    outputs_dir = pathlib.Path(__file__).parents[2] / "outputs"
    output_file = pathlib.Path(output_file)
    if not output_file.is_absolute():
        output_file = outputs_dir / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    store = pandas.HDFStore(str(output_file), "a")
    data = store["data"] if "data" in store else None

    await gort.telescopes.goto_named_position("zenith")

    for npoint, (ra, dec) in enumerate(points):
        print()
        gort.log.info(f"({npoint+1}/{n_points}): Going to {ra:.6f}, {dec:.6f}.")

        results = await asyncio.gather(
            *[get_offset(gort, tel, ra, dec) for tel in telescopes],
            return_exceptions=True,
        )

        valid = []
        for ii, tel in enumerate(telescopes):
            result = results[ii]
            if result is None:
                gort.log.warning(
                    f"Failed determining offset for telescope {tel} "
                    f"at ({ra:.6f}, {dec:.6f})"
                )
            elif isinstance(result, BaseException):
                gort.log.warning(f"Telescope {tel} failed with error: {str(result)}")
            else:
                valid.append(result)

        if len(valid) == 0:
            continue

        this_offsets = pandas.DataFrame.from_records(valid)

        print()
        print(this_offsets)

        if data is not None:
            data = pandas.concat([data, this_offsets])
        else:
            data = this_offsets

        data = data.reset_index(drop=True)
        store.put("data", data)

    store.close()


if __name__ == "__main__":
    NPOINTS = 95
    RA_RANGE = (30, 270)
    DEC_RANGE = (-75, 0)
    TELESCOPES = ["sci", "spec", "skye", "skyw"]

    asyncio.run(
        pointing_model(
            "pointing_100pt.h5",
            NPOINTS,
            RA_RANGE,
            DEC_RANGE,
            telescopes=TELESCOPES,
        )
    )
