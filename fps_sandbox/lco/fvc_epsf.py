#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-05-10
# @Filename: fvc_epsf.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib

import numpy
import pandas
import tqdm
from astropy.io import fits
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from photutils.psf import EPSFBuilder, EPSFModel, EPSFStar, EPSFStars

from sdsstools import get_logger


PATH = "/data/fcam/lco/60074"

log = get_logger("fvc_epsf")
log.sh.setLevel(5)


def extract_cutouts(file: pathlib.Path, box_size=50, plot=True):
    """Extract cutouts from an image."""

    log.info(f"processing image {file.name}")

    hdus = fits.open(str(file))
    cutouts = fits.HDUList([fits.PrimaryHDU()])

    data = hdus["RAW"].data
    meas = pandas.DataFrame(hdus["POSITIONERTABLEMEAS"].data)

    for _, row in meas.iterrows():
        pid = row.positionerID
        x = int(row.x)
        y = int(row.y)

        in_box = meas.loc[
            (meas.x > x - box_size)
            & (meas.x <= x + box_size)
            & (meas.y > y - box_size)
            & (meas.y <= y + box_size)
        ]

        if len(in_box) > 1:
            # log.warning(f"skipping positionerID={pid}.")
            continue

        b2 = box_size // 2
        cutout_data = data[y - b2 : y + b2, x - b2 : x + b2]

        cutout_hdu = fits.ImageHDU(data=cutout_data, header=hdus["RAW"].header)

        cutout_hdu.header["POSID"] = pid
        cutout_hdu.header["CENX"] = row.x
        cutout_hdu.header["CENY"] = row.y

        cutouts.append(cutout_hdu)

    cutouts_path = file.parent / f"cutouts-{file.name}"
    cutouts.writeto(cutouts_path, overwrite=True)

    # if plot:
    #     plot_cutouts(cutouts_path)

    return cutouts_path


def plot_cutouts(file: pathlib.Path):
    """Plots cutouts as PDFs."""

    cutout_huds = fits.open(file)
    sorted_hdus = list(sorted(cutout_huds[1:], key=lambda h: h.header["POSID"]))

    plot_file = pathlib.Path(str(file).replace("fits", "pdf"))

    with plt.ioff():
        with PdfPages(str(plot_file)) as pdf:
            for ii in range(0, len(sorted_hdus), 20):
                plot_hdus = sorted_hdus[ii : ii + 20]

                figure, ax = plt.subplots(5, 4, figsize=(8.5, 11))

                ihdu = 0
                for row in ax:
                    for col in row:
                        if len(plot_hdus) >= ihdu + 1:
                            data = plot_hdus[ihdu].data
                            pid = plot_hdus[ihdu].header["POSID"]
                            norm = simple_norm(data, "log", percent=99.0)
                            col.imshow(data, norm=norm, origin="lower", cmap="viridis")

                            col.text(
                                0.03,
                                0.97,
                                str(pid),
                                va="top",
                                weight="bold",
                                transform=col.transAxes,
                            )

                            x = plot_hdus[ihdu].header["CENX"]
                            y = plot_hdus[ihdu].header["CENY"]
                            col.text(
                                0.97,
                                0.97,
                                f"{int(x), int(y)}",
                                va="top",
                                ha="right",
                                transform=col.transAxes,
                            )

                        col.axis("off")

                        ihdu += 1

                figure.subplots_adjust(
                    wspace=0.05,
                    top=0.97,
                    bottom=0.03,
                    left=0.05,
                    right=0.95,
                )

                pdf.savefig(figure)
                plt.close(figure)

    log.debug(f"Saved cutouts plot {plot_file.name}")


def _create_one_epsf(args: tuple[int, list[numpy.ndarray]]):
    """Calculates the EPSF for one positioner."""

    posid, cutouts = args

    epsf_stars = EPSFStars(
        [
            EPSFStar(data, cutout_center=(data.shape[1] // 2, data.shape[0] // 2))
            for data in cutouts
        ]
    )
    epsf_builder = EPSFBuilder(
        smoothing_kernel="quadratic",
        maxiters=20,
        progress_bar=False,
    )
    epsf, fitted_stars = epsf_builder(epsf_stars)
    print(epsf)

    return (posid, (epsf, fitted_stars))


def create_epsf(images: list[pathlib.Path]):
    """Create EPSF for each positioner."""

    positioner_to_cutouts = {}
    for image in images:
        hdus = fits.open(image)
        for hdu in hdus[1:]:
            posid = hdu.header["POSID"]
            if posid not in positioner_to_cutouts:
                positioner_to_cutouts[posid] = []
            positioner_to_cutouts[posid].append(hdu.data)

    log.info("Fitting EPSF to FVC data.")

    epsfs = {}
    epsf_fitted_stars = {}

    with multiprocessing.Pool(processes=24) as p:
        with tqdm.tqdm(total=len(positioner_to_cutouts)) as pbar:
            items = positioner_to_cutouts.items()
            for posid, res in p.imap_unordered(_create_one_epsf, items):
                (epsf, fitted_stars) = res
                epsfs[posid] = epsf
                epsf_fitted_stars[posid] = fitted_stars

                pbar.update()

    epfs_hdus = fits.HDUList([fits.PrimaryHDU()])
    for posid, epsf in epsfs.items():
        epfs_hdu = fits.ImageHDU(data=epsf.data, name=f"P{posid:04d}")
        epfs_hdu.header["POSID"] = posid
        epfs_hdus.append(epfs_hdu)

    parent = images[0].parent
    epfs_hdus.writeto(str(parent / "fvc_epsf.fits"), overwrite=True)


def plot_epsf():
    """Plots the EPSF for each positioner."""

    parent = pathlib.Path(PATH)
    epfs_hdus = fits.open(str(parent / "fvc_epsf.fits"))
    sorted_hdus = list(sorted(epfs_hdus[1:], key=lambda h: h.header["POSID"]))

    plot_file = parent / "epsf.pdf"

    with plt.ioff():
        with PdfPages(str(plot_file)) as pdf:
            for ii in range(0, len(sorted_hdus), 20):
                plot_hdus = sorted_hdus[ii : ii + 20]

                figure, ax = plt.subplots(5, 4, figsize=(8.5, 11))

                ihdu = 0
                for row in ax:
                    for col in row:
                        if len(plot_hdus) >= ihdu + 1:
                            data = plot_hdus[ihdu].data
                            pid = plot_hdus[ihdu].header["POSID"]
                            norm = simple_norm(data, "log", percent=99.0)
                            col.imshow(data, norm=norm, origin="lower", cmap="viridis")

                            col.text(
                                0.03,
                                0.97,
                                str(pid),
                                va="top",
                                weight="bold",
                                transform=col.transAxes,
                            )

                        col.axis("off")

                        ihdu += 1

                figure.subplots_adjust(
                    wspace=0.05,
                    top=0.97,
                    bottom=0.03,
                    left=0.05,
                    right=0.95,
                )

                pdf.savefig(figure)
                plt.close(figure)

    log.debug(f"Saved EPSF plot {plot_file.name}")


# def fit_fvc_cutouts(cutout_image: pathlib.Path, plot=True):
#     """Fits FVC data using EPSF models and plots the results."""

#     cutouts = fits.open(cutout_image)
#     cutout_hdus = list(sorted(cutouts[1:], key=lambda h: h.header["POSID"]))

#     epsf = fits.open(str(cutout_image.parent / "fvc_epsf.fits"))

#     for hdu in cutout_hdus[1:]:
#         posid = hdu.header["POSID"]

#         x = hdu.header["CENX"]
#         y = hdu.header["CENY"]

#         epsf_data = epsf[f"P{posid:04d}"].data

#         b2 = hdu.data.shape[0] // 2

#         epsf_model = EPSFModel(
#             epsf_data,
#             oversampling=4,
#             normalize=False,
#             origin=(0, 0),
#             x_0=50,
#             y_0=50,
#         )

#         lev = LevMarLSQFitter()
#         xgrid, ygrid = numpy.meshgrid(
#             numpy.arange(0, hdu.data.shape[1]),
#             numpy.arange(0, hdu.data.shape[0]),
#         )

#         fit_model = lev(
#             epsf_model,
#             ygrid,
#             xgrid,
#             hdu.data,
#             maxiter=1000,
#         )


def calculate_epsf():
    """Calculate EPSF from FVC images."""

    # files = sorted(pathlib.Path(PATH).glob("proc-fimg-*.fits"))

    # cutout_images: list[pathlib.Path] = []

    # with multiprocessing.Pool(24) as pool:
    #     cutout_images = pool.map(extract_cutouts, files)

    # create_epsf(cutout_images)
    plot_epsf()


def fit_fvc_data():
    """Fits FVC data using the calculated EPSF for each positioner."""

    cutouts = list(sorted(pathlib.Path(PATH).glob("cutout*.fits")))[0:1]

    with multiprocessing.Pool(24) as pool:
        pool.map(fit_fvc_cutouts, cutouts)


if __name__ == "__main__":
    calculate_epsf()
    # fit_fvc_data()
