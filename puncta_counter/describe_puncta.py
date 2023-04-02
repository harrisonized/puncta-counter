#!/usr/bin/env python3

"""Takes nuclei.csv and puncta.csv files, computes boundaries, then plots using Bokeh
This script is still in development and not ready for use
As of 04/01/2023, this script has a basic level of outlier filtering prior to plotting
"""

import os
from os.path import join as ospj
from tqdm import tqdm
import argparse
import logging
import datetime as dt
import numpy as np
import pandas as pd

from puncta_counter.src.preprocessing import preprocess_df, reassign_puncta_to_nuclei
from puncta_counter.src.summarize import (generate_circle, generate_ellipse, compute_mahalanobis_distances,
                                          plot_nuclei_circles_puncta, plot_nuclei_ellipses_puncta)
from puncta_counter.utils.common import dirname_n_times
from puncta_counter.utils.plotting import save_plot_as_png
from puncta_counter.utils.logger import configure_logger
from puncta_counter.etc.columns import ellipse_cols, nuclei_cols, puncta_cols

script_name = 'run_puncta_counter'
this_dir = os.path.realpath(ospj(os.getcwd(), os.path.dirname(__file__)))
base_dir = dirname_n_times(this_dir, 1)
os.chdir(base_dir)


# Functions
# # parse_args
# # main


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="")

    # io
    parser.add_argument("-i", "--input", dest="input_dir", default='data', action="store",
                        required=False, help="input directory")
    parser.add_argument("-o", "--output", dest="output_dir", default='data', action="store",
                        required=False, help="output directory")
    parser.add_argument("-s", "--save", dest="save", default=False, action="store_true",
                        required=False, help="turn this on to output the intermediate csv files")

    parser.add_argument("-a", "--algos", dest="algos", nargs='+',
                        default=['confidence_ellipse',
                                 'min_vol_ellipse',
                                 'circle',  # deprecated
                                ],
                        action="store", required=False, help="limit scope for testing")

    # other
    parser.add_argument("-l", "--log-dir", dest="log_dir", default='log', action="store",
                        required=False, help="set the directory for storing log files")
    parser.add_argument('--debug', default=False, action='store_false',
                        help='print debug messages')

    return parser.parse_args(args)


def main(args=None):
    start_time = dt.datetime.now()

    # ----------------------------------------------------------------------
    # Args

    args = parse_args(args)

    # ----------------------------------------------------------------------
    # Logging

    log_level = logging.DEBUG if args.debug else logging.INFO
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
    log_filename = ospj(args.log_dir, script_name)
    logger = logging.getLogger(__name__)
    logger = configure_logger(logger, log_level, log_filename)

    logger.info(f"running script")

    # ----------------------------------------------------------------------
    # Read and filter data


    # Nuclei
    nuclei = pd.read_csv("data/nuclei.csv")
    nuclei.rename(columns={'ObjectNumber': 'NucleiObjectNumber'}, inplace=True)
    nuclei = preprocess_df(nuclei, nuclei_cols)
    nuclei['effective_radius_nuclei'] = nuclei['area'].apply(lambda x: np.sqrt(x/np.pi))

    nuclei_subset = nuclei[
        (nuclei['eccentricity'] < 0.69)
        & (nuclei['major_axis_length'] < 128)
    ].copy()

    if args.save:
        nuclei_subset.to_csv('data/nuclei_subset.csv', index=None)


    # Puncta
    puncta = pd.read_csv("data/puncta.csv")
    puncta = puncta.rename(columns={'ObjectNumber': 'PunctaObjectNumber'})
    puncta = preprocess_df(puncta, puncta_cols)
    puncta = reassign_puncta_to_nuclei(puncta, nuclei)

    min_scale = 0.7  # rescale to half the intensity ~1/np.sqrt(2) at the lowest brightness
    intensity_col = 'mean_intensity'
    puncta['fill_alpha'] = (puncta[intensity_col]-puncta[intensity_col].min()) / (
        puncta[intensity_col].max()-puncta[intensity_col].min()
    )*(1-min_scale)+(min_scale)

    puncta_subset = pd.merge(
        left=nuclei_subset[["image_number", 'nuclei_object_number']],
        right=puncta.loc[:, puncta.columns != 'puncta_object_number'],
        left_on=["image_number", 'nuclei_object_number'],
        right_on=['image_number', 'nuclei_object_number'],
        how="left",
    ).dropna(subset=['parent_manual_nuclei'])  # use nuclei_subset to filter puncta

    if args.save:
        puncta_subset.to_csv('data/puncta_subset.csv', index=None)

    filename_for_image_number = dict(zip(nuclei_subset['image_number'], nuclei_subset['file_name_tif']))


    # ----------------------------------------------------------------------
    # Confidence Ellipse

    if 'confidence_ellipse' in args.algos:

        logger.info(f"Generating confidence ellipse...")

        # used to filter outliers
        unweighted_ellipses = generate_ellipse(
            puncta_subset,
            algo='confidence_ellipse',
            aweights=None,
            n_std=1
        )

        # filter outliers mahalanobis_distances >= 1.5
        unweighted_ellipses = compute_mahalanobis_distances(unweighted_ellipses)
        unweighted_ellipses['center_x'] = unweighted_ellipses['centers'].apply(lambda x: x[0])
        unweighted_ellipses['center_y'] = unweighted_ellipses['centers'].apply(lambda x: x[1])
        unweighted_ellipses.drop(columns=['centers'], inplace=True)
        unweighted_ellipses = unweighted_ellipses[(unweighted_ellipses['mahalanobis_distances'] < 1.5)]  # filter

        # compute final boundaries
        # unweighted_ellipses['integrated_intensity_sq'] = unweighted_ellipses['integrated_intensity'].apply(
        #     lambda x: np.array(x)**2
        # )  # this didn't actually work well
        ellipses = generate_ellipse(
            unweighted_ellipses,
            algo='confidence_ellipse',
            aweights='integrated_intensity',
            n_std=2
        )

        if args.save:
            ellipses[['image_number', 'nuclei_object_number'] + ellipse_cols].to_csv(
                'data/ellipses/confidence_ellipse.csv', index=None
            )

        for image_number in tqdm(nuclei_subset['image_number'].unique()):
            title = filename_for_image_number[image_number].split('.')[0]

            plot = plot_nuclei_ellipses_puncta(
                nuclei=nuclei_subset.loc[(nuclei_subset['image_number']==image_number)],
                ellipses=ellipses.loc[(ellipses['image_number']==image_number)],
                puncta=puncta_subset.loc[(puncta_subset['image_number']==image_number)],
                title=title
            )

            save_plot_as_png(plot, f"figures/confidence_ellipse/{title}.png")

    # ----------------------------------------------------------------------
    # Mimimum Bounding Ellipse

    if 'min_vol_ellipse' in args.algos:

        logger.info(f"Generating minimum bounding ellipse...")
        ellipses = generate_ellipse(puncta_subset, algo='min_vol_ellipse')
        if args.save:
            ellipses[['image_number', 'nuclei_object_number'] + ellipse_cols].to_csv(
                'data/ellipses/min_vol_ellipse.csv', index=None
            )

        for image_number in tqdm(nuclei_subset['image_number'].unique()):
            title = filename_for_image_number[image_number].split('.')[0]
            
            plot = plot_nuclei_ellipses_puncta(
                nuclei=nuclei_subset.loc[(nuclei_subset['image_number']==image_number)],
                ellipses=ellipses.loc[(ellipses['image_number']==image_number)],
                puncta=puncta_subset.loc[(puncta_subset['image_number']==image_number)],
                title=title
            )

            save_plot_as_png(plot, f"figures/min_vol_ellipse/{title}.png")

    # ----------------------------------------------------------------------
    # Circles
    # This should be deprecated. I'm making it available for now just for comparison.

    if 'circle' in args.algos:

        logger.info(f"Generating gaussian circles...")
        circles = generate_circle(puncta_subset)
        if args.save:
            circles[['image_number', "nuclei_object_number",
                     "center_x_mean", "center_y_mean",
                     "effective_radius_puncta"]].to_csv('data/ellipses/circles.csv', index=None)

        for image_number in tqdm(nuclei_subset['image_number'].unique()):
            title = filename_for_image_number[image_number].split('.')[0]

            plot = plot_nuclei_circles_puncta(
                nuclei=nuclei_subset.loc[(nuclei_subset['image_number']==image_number)],
                circles=circles.loc[(circles['image_number']==image_number)],
                puncta=puncta_subset.loc[(puncta_subset['image_number']==image_number)],
                title=title
            )

            save_plot_as_png(plot, f"figures/circle/{title}.png")

    # ----------------------------------------------------------------------
    # End

    runtime = (dt.datetime.now() - start_time).total_seconds()
    logger.info(f"Script completed in {int(runtime // 60)} min {round(runtime % 60, 2)} sec")


if __name__ == "__main__":
    main()
