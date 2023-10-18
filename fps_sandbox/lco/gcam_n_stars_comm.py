#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-09-12
# @Filename: gcam_n_stars_comm.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import pathlib

import numpy
import pandas
from peewee import fn
from tqdm import tqdm

from coordio.guide import gfa_to_radec
from sdssdb.peewee.sdss5db import catalogdb as cdb
from sdssdb.peewee.sdss5db import targetdb as tdb


INPUTS = pathlib.Path(__file__).parents[1] / "inputs/lco"
RESULTS = pathlib.Path(__file__).parents[1] / "results/lco"

# To run in mako
tdb.database.connect("sdss5db", user="sdss", port=5433)

# To run from operations
# tdb.database.set_profile("tunnel_operations")

assert tdb.database.connected


def get_comm_designs():
    designs = (
        tdb.Design.select(
            tdb.Design.design_id,
            tdb.Field.racen,
            tdb.Field.deccen,
            tdb.Field.position_angle,
            tdb.Observatory.label,
        )
        .join(tdb.DesignToField)
        .join(tdb.Field)
        .join(tdb.Observatory)
        .where(tdb.Design.design_mode % "%_eng%")
        .distinct(tdb.Design.design_id)
        .order_by(tdb.Design.design_id)
    ).tuples()

    return list(designs)


def gcam_n_stars_comm(designs_file: pathlib.Path | str | None = None):
    """Gets the number of stars in each GFA for a list of commissioning designs."""

    if designs_file is None:
        comm_designs = pandas.DataFrame(
            get_comm_designs(),
            columns=["design_id", "ra", "dec", "pa", "obs"],
        )
        comm_designs.sort_values("design_id", inplace=True)

        INPUTS.mkdir(parents=True, exist_ok=True)
        outfile = INPUTS / "comm_designs.csv"
        outfile.unlink(missing_ok=True)

        comm_designs.to_csv(outfile, index=False)
    else:
        comm_designs = pandas.read_csv(designs_file)

    comm_designs.set_index("design_id", inplace=True)

    n_stars = []
    solved = []
    for design_id, (ra, dec, pa, obs) in tqdm(list(comm_designs.iterrows())):
        # if design_id not in [112878, 111564, 111452, 111472, 111510]:
        #     continue
        gfas_solved = 0
        for gfa in [1, 2, 3, 4, 5, 6]:
            gfa_ra, gfa_dec = gfa_to_radec(obs.upper(), 1024, 1024, gfa, ra, dec, pa)

            # gfa_ra0 = gfa_ra - 0.04153 / numpy.cos(numpy.radians(gfa_dec))
            # gfa_ra1 = gfa_ra + 0.04153 / numpy.cos(numpy.radians(gfa_dec))
            # gfa_dec0 = gfa_dec - 0.04153
            # gfa_dec1 = gfa_dec + 0.04153

            # query = list(
            #     (
            #         cdb.Gaia_DR2.select(cdb.Gaia_DR2.phot_g_mean_mag)
            #         .where(cdb.Gaia_DR2.ra > gfa_ra0)
            #         .where(cdb.Gaia_DR2.ra < gfa_ra1)
            #         .where(cdb.Gaia_DR2.dec > gfa_dec0)
            #         .where(cdb.Gaia_DR2.dec < gfa_dec1)
            #         .where(cdb.Gaia_DR2.phot_g_mean_mag <= 19)
            #     ).tuples()
            # )

            query = list(
                (
                    cdb.Gaia_DR2.select(cdb.Gaia_DR2.phot_g_mean_mag)
                    .where(
                        fn.q3c_radial_query(
                            cdb.Gaia_DR2.ra,
                            cdb.Gaia_DR2.dec,
                            gfa_ra,
                            gfa_dec,
                            0.0415,
                        )
                    )
                    .where(cdb.Gaia_DR2.phot_g_mean_mag <= 19)
                ).tuples()
            )

            mags = numpy.array(list(zip(*query))[0])

            n_19 = (mags <= 19).sum()
            if n_19 >= 10:
                gfas_solved |= 1 << gfa - 1

            n_stars.append(
                (
                    design_id,
                    gfa,
                    (mags <= 19).sum(),
                    (mags <= 18).sum(),
                )
            )

        solved.append((design_id, ra, dec, pa, gfas_solved))

    n_stars_df = pandas.DataFrame(
        n_stars,
        columns=["design_id", "gfa", "n_mag19", "n_mag18"],
    )

    n_stars_path = RESULTS / "gcam_n_stars_comm" / "n_stars.csv"
    n_stars_path.parent.mkdir(exist_ok=True, parents=True)
    n_stars_path.unlink(True)
    n_stars_df.to_csv(n_stars_path, index=False)

    solved_df = pandas.DataFrame(
        solved,
        columns=["design_id", "ra", "dec", "pa", "gfas"],
    )
    solved_df["n_solved"] = solved_df.groupby("design_id")["gfas"].transform(
        lambda g: sum([1 if (int(g) & (1 << n)) > 0 else 0 for n in range(6)])
    )
    solved_path = RESULTS / "gcam_n_stars_comm" / "solved.csv"
    solved_path.parent.mkdir(exist_ok=True, parents=True)
    solved_path.unlink(True)
    solved_df.to_csv(solved_path, index=False)


if __name__ == "__main__":
    gcam_n_stars_comm()
