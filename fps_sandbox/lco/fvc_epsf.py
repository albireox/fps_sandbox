#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2023-05-10
# @Filename: fvc_epsf.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import pandas
import tqdm
from astropy.io import fits
from astropy.visualization import simple_norm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from photutils.psf import EPSFBuilder, EPSFModel, EPSFStar, EPSFStars

from sdsstools import get_logger


PATH = "/data/fcam/lco/60074"

log = get_logger("fvc_epsf")
log.sh.setLevel(5)


def extract_cutouts(file: pathlib.Path | str, box_size=50):
    """Extract cutouts from an image."""

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
            log.warning(f"skipping positionerID={pid}.")
            continue

        b2 = box_size // 2
        cutout_data = data[y - b2 : y + b2, x - b2 : x + b2]

        cutout_hdu = fits.ImageHDU(data=cutout_data, header=hdus["RAW"].header)

        cutout_hdu.header["POSID"] = pid
        cutout_hdu.header["CENX"] = row.x
        cutout_hdu.header["CENY"] = row.y

        cutouts.append(cutout_hdu)

    return cutouts


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


def create_epsf(images: list[pathlib.Path]):
    """Create EPSF for each positioner."""

    positioner_to_cutouts = {}
    for image in images:
        hdus = fits.open(image)
        for hdu in hdus[1:]:
            posid = hdu.header["POSID"]
            if posid not in positioner_to_cutouts:
                positioner_to_cutouts[posid] = []
            positioner_to_cutouts[posid].append(hdu)

    epsfs: dict[int, EPSFModel] = {}
    epsf_fitted_stars: dict[int, EPSFStars] = {}

    log.info("Fitting EPSF to FVC data.")

    for posid in tqdm.tqdm(list(positioner_to_cutouts)):
        cutouts = positioner_to_cutouts[posid]
        epsf_stars = EPSFStars(
            [
                EPSFStar(
                    hdu.data,
                    cutout_center=(hdu.data.shape[1] // 2, hdu.data.shape[0] // 2),
                )
                for hdu in cutouts
            ]
        )
        epsf_builder = EPSFBuilder(oversampling=4, maxiters=5, progress_bar=False)
        epsf, fitted_stars = epsf_builder(epsf_stars)
        epsfs[posid] = epsf
        epsf_fitted_stars[posid] = fitted_stars

    epfs_hdus = fits.HDUList([fits.PrimaryHDU()])
    for posid, epsf in epsfs.items():
        epfs_hdu = fits.ImageHDU(data=epsf.data)
        epfs_hdu.header["POSID"] = posid
        epfs_hdus.append(epfs_hdu)

    parent = images[0].parent
    epfs_hdus.writeto(str(parent / "fvc_epsf.fits"), overwrite=True)


def calculate_epsf():
    """Calculate EPSF from FVC images."""

    files = sorted(pathlib.Path(PATH).glob("proc-fimg-*.fits"))

    cutout_images: list[pathlib.Path] = []
    for file in files:
        log.info(f"processing image {file.name}")
        cutouts = extract_cutouts(file)

        cutouts_path = file.parent / f"cutouts-{file.name}"
        cutouts.writeto(cutouts_path, overwrite=True)
        cutout_images.append(cutouts_path)

        plot_cutouts(cutouts_path)

    create_epsf(cutout_images)


if __name__ == "__main__":
    calculate_epsf()
