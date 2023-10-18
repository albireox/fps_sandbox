#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @Author: José Sánchez-Gallego (gallegoj@uw.edu)
# @Date: 2022-12-05
# @Filename: boss_stats.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)

from __future__ import annotations

import multiprocessing
import pathlib

import numpy
import pandas
import seaborn
from astropy import table
from astropy.stats.sigma_clipping import SigmaClip
from gtools.boss import quick_bias
from gtools.boss.tools import list_exposures
from matplotlib import pyplot as plt


seaborn.set_theme()


BASE_PATH = "/data/spectro/lco/"

LCO_MJDS_BIAS = [59917, 59918]
LCO_MJDS_DARK = [59917, 59918]

MIN_EXPNO = 4342
MIN_DARK = 4431

SHAPE = (4128, 4114)

RESULTS = pathlib.Path(__file__).parent / "../../results/lco/boss/"
RESULTS.mkdir(parents=True, exist_ok=True)


def get_biases():
    tables = [list_exposures(f"{BASE_PATH}/{mjd}") for mjd in LCO_MJDS_BIAS]

    exposures = table.vstack(tables)
    exposures = exposures[exposures["Exposure"] >= MIN_EXPNO]
    exposures.sort("Exposure")

    return exposures[exposures["Flavour"] == "bias"]


def get_perc98(data: numpy.ndarray):
    sigclip = SigmaClip(2.5)
    return numpy.percentile(sigclip(data, masked=False), 98)


def _multiprocess_bias(exp):
    jMid = SHAPE[0] // 2
    iMid = SHAPE[1] // 2

    bias_exp = quick_bias(f'{BASE_PATH}/{exp["MJD"]}/{exp["File"]}')

    bias_global = get_perc98(bias_exp[100:-100, 100:-100])
    bias_q1 = get_perc98(bias_exp[10:jMid, 10:iMid])
    bias_q2 = get_perc98(bias_exp[10:jMid, iMid:-10])
    bias_q3 = get_perc98(bias_exp[jMid:-10, iMid:-10])
    bias_q4 = get_perc98(bias_exp[jMid:-10, 10:iMid])

    return [
        (exp["Exposure"], "global", bias_global),
        (exp["Exposure"], "q1", bias_q1),
        (exp["Exposure"], "q2", bias_q2),
        (exp["Exposure"], "q3", bias_q3),
        (exp["Exposure"], "q4", bias_q4),
    ]


def plot_bias_readnoise():
    biases = get_biases()

    for camera in ["r2", "b2"]:
        cam_biases = biases[biases["Camera"] == camera]

        with multiprocessing.Pool(8) as pool:
            results = pool.map(_multiprocess_bias, cam_biases)

        cam_results = []
        for res in results:
            for section in res:
                cam_results.append(section)

        df = pandas.DataFrame(cam_results, columns=["expno", "section", "percentile98"])
        df.to_hdf(RESULTS / f"boss_bias_{camera}.hdf", "data")

        fg = seaborn.FacetGrid(df, row="section", hue="section")
        fg.map_dataframe(seaborn.scatterplot, x="expno", y="percentile98")

        for ii in range(5):
            if ii == 0:
                section = "global"
            else:
                section = f"q{ii}"

            ax = fg.axes[ii][0]  # type: ignore
            ax.set_ylabel("Percentile 98% [ADU]")

            if ii != 4:
                ax.set_xlabel("")
            else:
                ax.set_xlabel("Expose number")

            mean = df.loc[df.section == section, "percentile98"].mean()
            ax.legend([f"{section} ({mean:.1f})"], loc="upper right")

        fg.figure.set_size_inches(15, 10)

        plt.tight_layout()

        fg.figure.savefig(str(RESULTS / f"boss_bias_{camera}.pdf"))

        plt.close(fg.figure)


if __name__ == "__main__":
    plot_bias_readnoise()
