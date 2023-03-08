#!/usr/bin/env python3

"""Takes nuclei.csv and puncta.csv files, computes boundaries, then plots using Bokeh
This script is still in development and not ready for use
As of 03/08/2023, this script is almost ready to use, but has some bugs.
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
from puncta_counter.src.summarize import (generate_circle, generate_ellipse,
                                          plot_nuclei_circles_puncta, plot_nuclei_ellipses_puncta)
from puncta_counter.utils.common import dirname_n_times
from puncta_counter.utils.plotting import save_plot_as_png
from puncta_counter.utils.logger import configure_logger
from puncta_counter.etc.columns import nuclei_cols, puncta_cols

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

    parser.add_argument("-a", "--algos", dest="algos", nargs='+',
    	                default=['confidence_ellipse',
    	                		 # 'min_vol_ellipse',  # bugs
    	                		 # 'circle',  # deprecate
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
    nuclei = preprocess_df(nuclei, nuclei_cols)
    nuclei['effective_radius_nuclei'] = nuclei['area'].apply(lambda x: np.sqrt(x/np.pi))
    nuclei['angle'] = nuclei['orientation'].apply(lambda x: x/180*np.pi)
    nuclei_subset = nuclei[
        (nuclei['eccentricity'] < 0.69)
        & (nuclei['major_axis_length'] < 128)
    ].copy()
    nuclei_subset.to_csv('data/nuclei_subset.csv', index=None)

    # Puncta
    puncta = pd.read_csv("data/puncta.csv")
    puncta = preprocess_df(puncta, puncta_cols)
    puncta = reassign_puncta_to_nuclei(puncta, nuclei)
    puncta['angle'] = puncta['orientation'].apply(lambda x: x/180*np.pi)
    puncta_subset = pd.merge(
        left=nuclei_subset[["image_number", 'object_number']],
        right=puncta.loc[:, puncta.columns != 'object_number'],
        left_on=["image_number", 'object_number'],
        right_on=['image_number', 'nuclei_object_number'],
        how="left",
    ).dropna(subset=['nuclei_object_number'])  # use nuclei_subset to filter puncta
    puncta.to_csv('data/puncta_subset.csv', index=None)

    filename_for_image_number = dict(zip(nuclei_subset['image_number'], nuclei_subset['file_name_tif']))

    # ----------------------------------------------------------------------
    # Confidence Ellipse

    if 'confidence_ellipse' in args.algos:

        logger.info(f"Generating confidence ellipse...")
        ellipses = generate_ellipse(puncta_subset, algo='confidence_ellipse')
        ellipses['angle'] = ellipses['orientation'].apply(lambda x: x/180*np.pi)

        for image_number in tqdm(nuclei_subset['image_number'].unique()):
            title = filename_for_image_number[image_number].split('.')[0]

            plot = plot_nuclei_ellipses_puncta(
                nuclei=nuclei_subset.loc[(nuclei_subset['image_number']==image_number)],
                ellipses=ellipses.loc[(ellipses['image_number']==image_number)],
                puncta=puncta_subset.loc[(puncta_subset['image_number']==image_number)],
            )

            save_plot_as_png(plot, f"figures/confidence_ellipse/{title}.png")

    # ----------------------------------------------------------------------
    # Mimimum Bounding Ellipse
    # There's a bug in the current implementation

    if 'min_vol_ellipse' in args.algos:

        logger.info(f"Generating minimum bounding ellipse...")


        ellipses = generate_ellipse(puncta_subset, algo='min_vol_ellipse')  # this has a bug
        ellipses['angle'] = ellipses['orientation'].apply(lambda x: x/180*np.pi)

        for image_number in tqdm(nuclei_subset['image_number'].unique()):
            title = filename_for_image_number[image_number].split('.')[0]
            
            plot = plot_nuclei_ellipses_puncta(
                nuclei=nuclei_subset.loc[(nuclei_subset['image_number']==image_number)],
                ellipses=ellipses.loc[(ellipses['image_number']==image_number)],
                puncta=puncta_subset.loc[(puncta_subset['image_number']==image_number)],
            )

            save_plot_as_png(plot, f"figures/min_vol_ellipse/{title}.png")

    # ----------------------------------------------------------------------
    # Circles
    # This should be deprecated. I'm making it available for now just for comparison

    if 'circle' in args.algos:

        logger.info(f"Generating gaussian circles...")
        circles = generate_circle(puncta_subset)
        for image_number in tqdm(nuclei_subset['image_number'].unique()):
            title = filename_for_image_number[image_number].split('.')[0]

            plot = plot_nuclei_circles_puncta(
                nuclei=nuclei_subset.loc[(nuclei_subset['image_number']==image_number)],
                circles=circles.loc[(circles['image_number']==image_number)],
                puncta=puncta_subset.loc[(puncta_subset['image_number']==image_number)],
            )

            save_plot_as_png(plot, f"figures/circle/{title}.png")

    # ----------------------------------------------------------------------
    # End

    runtime = (dt.datetime.now() - start_time).total_seconds()
    logger.info(f"Script completed in {int(runtime // 60)} min {round(runtime % 60, 2)} sec")


if __name__ == "__main__":
    main()
