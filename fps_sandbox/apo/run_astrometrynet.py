#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-05
# @Filename: run_astrometrynet.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import os
from glob import glob

import pandas
import sep
from astropy.io import fits
from matplotlib import pyplot as plt


plt.ioff()


DIRECTORIES = ["/data/gcam/59551/", "/data/gcam/59552/"]


def process_image(image, sigma=10, min_npix=50):

    print(f"Processing {image}")

    data = fits.getdata(image)
    back = sep.Background(data.astype("int32"))

    fig, ax = plt.subplots()
    ax.imshow(back, origin="lower")
    ax.set_title("Background: " + image)
    ax.set_gid(False)
    fig.savefig(image + "background.png", dpi=300)

    regions = pandas.DataFrame(
        sep.extract(
            data - back.back(),
            sigma,
            err=back.globalrms,
        )
    )
    regions.loc[regions.npix > min_npix, "valid"] = 1
    regions.to_hdf(image + ".hdf", "data")

    fig, ax = plt.subplots()
    ax.set_title(image + r" $(\sigma={})$".format(sigma))
    ax.set_gid(False)

    data_back = data - back.back()
    ax.imshow(
        data_back,
        origin="lower",
        cmap="gray",
        vmin=data_back.mean() - back.globalrms,
        vmax=data_back.mean() + back.globalrms,
    )
    fig.savefig(image + "original.png", dpi=300)
    ax.scatter(
        regions.loc[regions.valid == 1].x,
        regions.loc[regions.valid == 1].y,
        marker="x",
        s=3,
        c="r",
    )
    fig.savefig(image + "centroids.png", dpi=300)

    plt.close("all")

    print(f"Number of sources: {len(regions)}")
    print()


if __name__ == "__main__":

    for directory in DIRECTORIES:
        images = sorted(glob(os.path.join(directory, "*.fits")))
        for image in images:
            try:
                process_image(image)
            except Exception as err:
                print(f"Failed processing {image}: {err}")
