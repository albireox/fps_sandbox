#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-05-17
# @Filename: check_refraction.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import ctypes
import os
import pathlib
import warnings

import numpy
import pandas
from astropy import units as uu
from astropy.coordinates import AltAz, Distance, EarthLocation, HADec, SkyCoord
from astropy.time import Time
from astropy.utils.exceptions import ErfaWarning
from jaeger.target.tools import read_confSummary

from coordio import ICRS, Field, FocalPlane, Observed, Site, Wok, sofa
from coordio.defaults import INST_TO_WAVE


RESULTS = pathlib.Path(__file__).parents[1] / "results"
SDSSCORE_DIR = pathlib.Path(os.environ["SDSSCORE_DIR"])


def wok_to_observed(
    xwok: float,
    ywok: float,
    zwok: float,
    racen: float,
    deccen: float,
    obs_epoch: float,
    obs_angle: float,
    scale: float,
    wavelength: float,
):
    site = Site("APO")
    site.set_time(obs_epoch)
    assert site.time

    icrs_bore = ICRS([[racen, deccen]])
    ics_bore = icrs_bore.to_epoch(site.time.jd)
    obs_bore = Observed(ics_bore, site=site, wavelength=INST_TO_WAVE["GFA"])

    wok = Wok([[xwok, ywok, zwok]], site=site, obsAngle=obs_angle)
    focal = FocalPlane(wok, wavelength=wavelength, site=site, fpScale=scale)
    field = Field(focal, field_center=obs_bore)
    obs = Observed(field, site=site, wavelength=wavelength)

    return obs


def check_configuration_refraction(configuration_id: int):
    """Check ra/dec observed coordinates using SOFA and astropy."""

    assert sofa is not None

    header, data = read_confSummary(configuration_id)
    data = data.loc[(data.assigned == 1) & (data.on_target == 1), :]

    longitude = 254.179722
    latitude = 32.766666667
    altitude = 2788.0

    g = 9.80665  # m / s^2
    R0 = 8.314462618  # J / (mol K)
    T0 = 288.16  # K
    M = 0.02896968  # kg / mol
    p0 = 1013.25  # millibar (hPa)

    pressure = p0 * numpy.exp(-(altitude * g * M) / (T0 * R0))
    # pressure = 942.129

    astropy_apo = EarthLocation.from_geodetic(
        longitude - 360.0,
        latitude,
        height=altitude,
        ellipsoid="WGS84",
    )

    obs_epoch = Time(
        header["epoch"] + 2500 / 86400,
        format="jd",
        scale="utc",
        location=astropy_apo,
    )

    for fibre_type in data.index.get_level_values(1).unique():
        df = data.loc[pandas.IndexSlice[:, fibre_type], :]

        wavelength = 16600 if fibre_type == "APOGEE" else 5400

        cos_dec = numpy.cos(numpy.deg2rad(df.deccat))

        pmra = numpy.deg2rad(df.pmra / 1000.0 / 3600.0 / cos_dec).copy()  # radians / yr
        pmra.loc[df.pmra <= -999.0] = 1e-9

        pmdec = numpy.deg2rad(df.pmdec / 1000.0 / 3600.0).copy()  # radians / yr
        pmdec.loc[df.pmdec <= -999.0] = 1e-9

        parallax = df.parallax.copy() / 1000.0  # arcsec
        parallax.loc[df.parallax <= -999.0] = 1e-8

        altaz_refraction = AltAz(
            obstime=obs_epoch.utc,
            location=astropy_apo,
            pressure=pressure * uu.mbar,
            obswl=wavelength * uu.angstrom,  # type:ignore
            temperature=10 * uu.deg_C,
            relative_humidity=0.5,
        )

        observed = []
        for ii in range(len(df)):
            # SOFA

            ra_j2000_rad = ctypes.c_double()
            dec_j2000_rad = ctypes.c_double()
            pmra_j2000 = ctypes.c_double()
            pmdec_j2000 = ctypes.c_double()
            parallax_j2000 = ctypes.c_double()
            rvel_j2000 = ctypes.c_double()

            cat_epoch = Time(df.coord_epoch.values[ii], format="jyear", scale="tdb")

            sofa.iauPmsafe(
                numpy.deg2rad(df.racat.values[ii]),
                numpy.deg2rad(df.deccat.values[ii]),
                pmra.values[ii],
                pmdec.values[ii],
                parallax.values[ii],
                0.0,  # radial velocity,
                cat_epoch.jd1,
                cat_epoch.jd2,
                2451545.0,
                0.0,
                ra_j2000_rad,
                dec_j2000_rad,
                pmra_j2000,
                pmdec_j2000,
                parallax_j2000,
                rvel_j2000,
            )

            az_obs_rad = ctypes.c_double()
            zen_obs_rad = ctypes.c_double()
            ha_obs_rad = ctypes.c_double()
            dec_obs_rad = ctypes.c_double()
            ra_obs_rad = ctypes.c_double()
            eo_obs_rad = ctypes.c_double()

            sofa.iauAtco13(
                ra_j2000_rad.value,
                dec_j2000_rad.value,
                pmra_j2000.value,
                pmdec_j2000.value,
                parallax_j2000.value,
                0.0,  # radial velocity,
                obs_epoch.utc.jd1,
                obs_epoch.utc.jd2,
                obs_epoch.delta_ut1_utc,
                numpy.radians(longitude - 360),
                numpy.radians(latitude),
                altitude,
                0.0,
                0.0,
                pressure,
                10,  # temperature
                0.5,  # relative humidity
                wavelength / 10000.0,
                az_obs_rad,
                zen_obs_rad,
                ha_obs_rad,
                dec_obs_rad,
                ra_obs_rad,
                eo_obs_rad,
            )

            alt_obs_deg = 90 - numpy.rad2deg(zen_obs_rad.value)

            # Recalculate RA/Dec observed from confSummary

            original_obs = wok_to_observed(
                df.xwok.values[ii],
                df.ywok.values[ii],
                df.zwok.values[ii],
                header["raCen"],
                header["decCen"],
                obs_epoch.tai.jd,
                header["pa"],
                header["focal_scale"],
                wavelength,
            )

            observed.append(
                (
                    numpy.rad2deg(ra_obs_rad.value),
                    numpy.rad2deg(dec_obs_rad.value),
                    numpy.rad2deg(ha_obs_rad.value),
                    numpy.rad2deg(az_obs_rad.value),
                    alt_obs_deg,
                    original_obs.ra[0],
                    original_obs.dec[0],
                    original_obs.ha[0],
                    original_obs[0][1],
                    original_obs[0][0],
                )
            )

        if len(observed) > 0:
            data.loc[
                df.index,
                [
                    "raobs_sofa",
                    "decobs_sofa",
                    "haobs_sofa",
                    "azobs_sofa",
                    "altobs_sofa",
                    "raobs",
                    "decobs",
                    "haobs",
                    "azobs",
                    "altobs",
                ],
            ] = observed

            # Astropy

            pmra_mas = df.pmra.values
            pmra_mas[pmra_mas <= -999.0] = 1e-9
            pmdec_mas = df.pmdec.values
            pmdec_mas[pmdec_mas <= -999.0] = 1e-9
            parallax_mas = df.parallax.values
            parallax_mas[parallax_mas <= -999.0] = 1e-8

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ErfaWarning)

                cat_epoch = Time(df.coord_epoch.values, format="jyear", scale="tdb")

                astropy_icrs = SkyCoord(
                    ra=df.racat.values * uu.degree,
                    dec=df.deccat.values * uu.degree,
                    frame="icrs",
                    pm_ra_cosdec=pmra_mas * uu.mas / uu.yr,
                    pm_dec=pmdec_mas * uu.mas / uu.yr,
                    distance=Distance(parallax=parallax_mas * uu.mas),
                    obstime=cat_epoch,
                )
                astropy_epoch = astropy_icrs.apply_space_motion(new_obstime=obs_epoch)
                astropy_altaz = astropy_epoch.transform_to(altaz_refraction)
                # astropy_altaz = AltAz(
                #     astropy_altaz.data,
                #     location=astropy_apo,
                #     obstime=obs_epoch,
                # )
                # print(altaz_refraction)
                # astropy_altaz = astropy_altaz.transform_to(altaz_refraction)
                # print(astropy_altaz)

                # astropy_hadec = astropy_altaz.transform_to(HADec)
                astropy_hadec = astropy_altaz.transform_to(HADec)
                # breakpoint()

                astropy_ra = obs_epoch.sidereal_time("mean") - astropy_hadec.ha

                data.loc[df.index, "raobs_astropy"] = astropy_ra.value * 15.0
                data.loc[df.index, "decobs_astropy"] = astropy_hadec.dec.value
                data.loc[df.index, "haobs_astropy"] = astropy_hadec.ha.value * 15.0
                data.loc[df.index, "azobs_astropy"] = astropy_altaz.az.value
                data.loc[df.index, "altobs_astropy"] = astropy_altaz.alt.value

            # lst = obs_epoch.sidereal_time("apparent").value * 15.0
            # ha = lst - df.ra.values

            # sin_HA = numpy.sin(numpy.radians(ha))
            # cos_HA = numpy.cos(numpy.radians(ha))
            # sin_lat = numpy.sin(numpy.radians(latitude))
            # cos_lat = numpy.cos(numpy.radians(latitude))
            # sin_dec = numpy.sin(numpy.radians(df.dec.values))
            # cos_dec = numpy.cos(numpy.radians(df.dec.values))
            # tan_dec = numpy.tan(numpy.radians(df.dec.values))

            # A_rad = numpy.arctan(sin_HA / (cos_HA * sin_lat - tan_dec * cos_lat))
            # h_rad = numpy.arcsin(sin_lat * sin_dec + cos_dec * cos_lat * cos_HA)

            # data.loc[df.index, "azobs_manual"] = numpy.rad2deg(A_rad)
            # data.loc[df.index, "altobs_manual"] = numpy.rad2deg(h_rad)

    breakpoint()
    return data


if __name__ == "__main__":
    check_configuration_refraction(5315)
