#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-05
# @Filename: run_astrometrynet.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from glob import glob

import pandas
import sep
from astropy.io import fits
from matplotlib import pyplot as plt


plt.ioff()

DIRECTORIES = []


def process_image(image):

    print(f"Processing {image}")

    data = fits.getdata(image)
    back = sep.Background(data.astype("int32"))

    fig, ax = plt.subplots()
    ax.imshow(back, origin="lower")
    ax.set_title("Background: " + image)
    fig.savefig(image + ".png", dpi=300)

    regions = pandas.DataFrame(sep.extract(data - back.back(), 3, err=back.globalrms))


if __name__ == "__main__":

    for directory in DIRECTORIES:
        images = sorted(glob("*.fits"))
        for image in images:
            try:
                process_image(image)
            except Exception as err:
                print(f"Failed processing {image}: {err}")
