#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-04-06
# @Filename: dar.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

from ctypes import c_double

import numpy

from coordio import sofa


g = 9.80665  # m / s^2
R0 = 8.314462618  # J / (mol K)
T0 = 288.16  # K
M = 0.02896968  # kg / mol
p0 = 1013.25  # millibar (hPa)


def calculate_athmospheric_refraction(
    zz: float,
    wave: float,
    altitude: float = 2400.0,
    temperature: float = 10.0,
    rh: float = 0.5,
):
    """Calculates the atmospheric refraction in the zenith direction.

    Parameters
    ----------
    zz
        Zenith distance of the target in degrees.
    wave
        The wavelength of observation in Angstrom.
    altitude
        The altitude of the observation in metres.
    temperature
        The temperature in degrees Celsius.
    rh
        The relative humidity as a fraction.

    Returns
    -------
    dz
        The atmospheric refraction in the zenith direction in arcsec.

    """

    pressure = p0 * numpy.exp(-(altitude * g * M) / (T0 * R0))

    aa = c_double()
    bb = c_double()

    sofa.iauRefco(pressure, temperature, rh, wave / 10000, aa, bb)

    zz_rad = numpy.deg2rad(zz)
    dz_rad = aa * numpy.tan(zz_rad) + bb * (numpy.tan(zz_rad)) ** 3

    return numpy.rad2deg(dz_rad) * 3600.0


def calculate_dar(zz: float, wave1: float, wave2: float, **kwargs):
    """Calculates the DAR between two wavelengths.

    Parameters
    ----------
    zz
        Zenith distance of the target in degrees.
    wave1
        The reference wavelength in Angstrom.
    wave2
        The comparison wavelength in Angstrom.
    kwargs
        Other parameters to pass to ``calculate_athmospheric_refraction``.

    Returns
    -------
    dar
        The DAR between the two wavelengths in arcsec.

    """

    dz1 = calculate_athmospheric_refraction(zz, wave1, **kwargs)
    dz2 = calculate_athmospheric_refraction(zz, wave2, **kwargs)

    return dz1 - dz2
