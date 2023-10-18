#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2021-12-05
# @Filename: run_astrometrynet.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

import os
import pathlib
from glob import glob

import pandas
import sep
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from cherno.astrometry import AstrometryNet
from matplotlib import pyplot as plt


plt.ioff()


DIRECTORIES = ["/data/gcam/59551/", "/data/gcam/59552/"]


def process_image(image, outdir, sigma=10, min_npix=50):
    path = pathlib.Path(image)
    mjd = path.parts[-2]
    basename = path.parts[-1]

    outdir = os.path.join(outdir, mjd)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    print(f"Processing {image}")

    data = fits.getdata(image)
    back = sep.Background(data.astype("int32"))

    fig, ax = plt.subplots()
    ax.imshow(back, origin="lower")
    ax.set_title("Background: " + mjd + "/" + basename)
    ax.set_gid(False)
    fig.savefig(os.path.join(outdir, basename + ".background.png"), dpi=300)

    regions = pandas.DataFrame(
        sep.extract(
            data - back.back(),
            sigma,
            err=back.globalrms,
        )
    )
    regions.loc[regions.npix > min_npix, "valid"] = 1
    regions.to_hdf(os.path.join(outdir, basename + ".hdf"), "data")

    fig, ax = plt.subplots()
    ax.set_title(mjd + "/" + basename)
    ax.set_gid(False)

    data_back = data - back.back()
    ax.imshow(
        data_back,
        origin="lower",
        cmap="gray",
        vmin=data_back.mean() - back.globalrms,
        vmax=data_back.mean() + back.globalrms,
    )
    ax.set_title(mjd + "/" + basename + r" $(\sigma={})$".format(sigma))
    fig.savefig(os.path.join(outdir, basename + ".original.png"), dpi=300)

    ax.scatter(
        regions.loc[regions.valid == 1].x,
        regions.loc[regions.valid == 1].y,
        marker="x",
        s=3,
        c="r",
    )
    fig.savefig(os.path.join(outdir, basename + ".centroids.png"), dpi=300)

    plt.close("all")


def run_astrometry(image, outdir):
    path = pathlib.Path(image)
    mjd = path.parts[-2]
    basename = path.parts[-1]

    outdir = os.path.join(outdir, mjd)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    header = fits.getheader(image, 1)

    pixel_scale = 0.216

    backend_config = os.path.join(
        os.path.dirname(__file__),
        "../data/astrometrynet.cfg",
    )
    astrometry_net = AstrometryNet()
    astrometry_net.configure(
        backend_config=backend_config,
        width=2048,
        height=2048,
        no_plots=True,
        scale_low=pixel_scale * 0.9,
        scale_high=pixel_scale * 1.1,
        scale_units="arcsecperpix",
        radius=2.0,
        dir=outdir,
    )

    centroids = pandas.read_hdf(os.path.join(outdir, basename + ".hdf"))

    gfa_xyls = Table.from_pandas(centroids.loc[:, ["x", "y"]])
    gfa_xyls_file = os.path.join(outdir, basename + ".xyls")
    gfa_xyls.write(gfa_xyls_file, format="fits", overwrite=True)

    wcs_output = os.path.join(outdir, basename + ".wcs")
    if os.path.exists(wcs_output):
        os.remove(wcs_output)

    proc = astrometry_net.run(
        [gfa_xyls_file],
        stdout=os.path.join(outdir, basename + ".stdout"),
        stderr=os.path.join(outdir, basename + ".stderr"),
        ra=header["RA"],
        dec=header["DEC"],
    )

    proc_hdu = fits.open(image).copy()

    if not os.path.exists(wcs_output):
        proc_hdu[1].header["SOLVED"] = False
        proc_hdu[1].header["SOLVTIME"] = proc.elapsed
    else:
        proc_hdu[1].header["SOLVED"] = True
        proc_hdu[1].header["SOLVTIME"] = proc.elapsed
        wcs = WCS(open(wcs_output).read())
        proc_hdu[1].header.update(wcs.to_header())

    proc_hdu.writeto(os.path.join(outdir, "proc-" + basename), overwrite=True)


if __name__ == "__main__":
    for directory in DIRECTORIES:
        images = sorted(glob(os.path.join(directory, "*.fits")))
        for image in images:
            image = os.path.join("/data/gcam/59552", image)
            try:
                process_image(image, "/data/astrometrynet/full/")
                run_astrometry(image, "/data/astrometrynet/full/")
            except Exception as err:
                print(f"Failed processing {image}: {err}")
