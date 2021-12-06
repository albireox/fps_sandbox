#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-11-07
# @Filename: analyse_coords.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import asyncio
from time import time

import numpy
import pandas
import seaborn
from astropy.io import fits
from matplotlib import pyplot as plt

from coordio import PositionerMetrology, Site, Tangent, Wok
from coordio.conv import (
    positionerToTangent,
    tangentToPositioner,
    tangentToWok,
    wokToTangent,
)
from coordio.defaults import (
    INST_TO_WAVE,
    POSITIONER_HEIGHT,
    getHoleOrient,
    positionerTable,
)


FILE = "/Users/albireo/Downloads/59521/proc-fimg-fvcn-0057.fits"
FILE_h5 = FILE.removesuffix(".fits") + ".h5"

seaborn.set_theme()


def create_dataframe():

    hdus = fits.open(FILE)

    measured = pandas.DataFrame(hdus["MEASURED"].data.newbyteorder().byteswap())
    posangles = pandas.DataFrame(hdus["POSANGLES"].data.newbyteorder().byteswap())
    posangles.rename(columns={"positionerID": "robotID"}, inplace=True)

    merged = pandas.merge(measured, posangles)
    merged.to_hdf(FILE_h5, "data")


def to_positioner():

    data: pandas.DataFrame = pandas.read_hdf(FILE_h5).set_index("holeID")
    positionerTable.set_index("positionerID", inplace=True)

    site = Site("APO")
    site.set_time()

    gfaw = INST_TO_WAVE["GFA"]

    positioner_calculated = []
    positioner_low_calculated = []
    print(time())
    for holeID, row in data.iterrows():
        wok_expected = row.loc[["xWokMetExpect", "yWokMetExpect"]].to_numpy()
        # positioner_data = positionerTable.loc[row.robotID]

        wok = Wok(
            [[wok_expected[0], wok_expected[1], POSITIONER_HEIGHT]],
            site=site,
        )
        tangent = Tangent(wok, holeID=holeID, site=site, wavelength=gfaw)
        positioner = PositionerMetrology(tangent, holeID=holeID, site=site)

        # hole_orient = getHoleOrient(site, holeID)
        # tangent_low = wokToTangent(
        #     wok_expected[0],
        #     wok_expected[1],
        #     POSITIONER_HEIGHT,
        #     *hole_orient,
        #     dx=positioner_data.dx,
        #     dy=positioner_data.dy,
        # )
        # positioner_low = tangentToPositioner(
        #     tangent_low[0],
        #     tangent_low[1],
        #     positioner_data.metX,
        #     positioner_data.metY,
        #     la=positioner_data.alphaArmLen,
        #     alphaOffDeg=positioner_data.alphaOffset,
        #     betaOffDeg=positioner_data.betaOffset,
        # )

        # positioner_calculated.append((positioner[0][0], positioner[0][1]))
        # positioner_low_calculated.append((positioner_low[0][0], positioner_low[1][0]))

    # data[["alpha_calculated", "beta_calculated"]] = positioner_calculated
    # data[["alpha_low_calculated", "beta_low_calculated"]] = positioner_low_calculated
    print(time())
    # data = data.loc[data.cmdBeta < 180.0]
    # data.dropna(inplace=True)

    # pandas.set_option("display.max_rows", None)

    # print(data.loc[:, ["cmdAlpha", "alpha_calculated", "alpha_low_calculated"]])
    # print(
    #     numpy.mean(abs(data.alpha_calculated - data.alpha_low_calculated)),
    #     numpy.mean(abs(data.cmdAlpha - data.alpha_low_calculated)),
    #     numpy.mean(abs(data.cmdBeta - data.beta_low_calculated)),
    # )
    # import ipdb; ipdb.set_trace()
    # plt.scatter(data.cmdAlpha, abs(data.cmdAlpha - data.alpha_low_calculated))
    # plt.scatter(data.cmdAlpha, abs(data.cmdAlpha - data.alpha_calculated))
    # plt.scatter(data.cmdAlpha, data.alpha_calculated)
    # plt.scatter(data.cmdAlpha, data.alpha_low_calculated)
    # plt.legend()
    # plt.show()


def cycle_from_positioner():

    data: pandas.DataFrame = pandas.read_hdf(FILE_h5).set_index("holeID")

    site = Site("APO")
    site.set_time()

    wok_calculated = []
    positioner_calculated = []
    for holeID, row in data.iterrows():
        alpha_report, beta_report = row[["alphaReport", "betaReport"]]

        hole_orient = getHoleOrient(site, holeID)

        tangent_p = positionerToTangent(
            [alpha_report, beta_report],
            [14.314, 0.0],
            7.4,
            0.0,
            0.0,
        )
        tangent_p.append(0)
        wok = tangentToWok(tangent_p, *hole_orient, POSITIONER_HEIGHT, 1, 0, 0, 0)
        wok_calculated.append(wok[0:2])

        tangent_w = wokToTangent(wok, *hole_orient, POSITIONER_HEIGHT, 1, 0, 0, 0)
        positioner = tangentToPositioner(tangent_w[0:2], [14.314, 0], 7.4, 0, 0, 0)
        positioner_calculated.append(positioner[0:2])

    data[["xWokCalculated", "yWokCalculated"]] = wok_calculated
    data[["alphaCalculated", "betaCalculated"]] = positioner_calculated

    pandas.set_option("display.max_rows", None)

    # print(data[["alphaCalculated", "alphaReport", "betaCalculated", "betaReport"]])
    print(data.alphaCalculated - data.alphaReport)


async def test_design(design_id):
    from peewee_async import Manager

    from jaeger.design import Design, targetdb

    def test(database):
        obj = Manager(targetdb.database)
        print(database)

    # obj = Manager(targetdb.database)
    print(targetdb.database)

    await asyncio.get_running_loop().run_in_executor(None, test, targetdb.database)


if __name__ == "__main__":
    # create_dataframe()
    to_positioner()
    # cycle_from_positioner()

    # asyncio.run(test_design(100))
