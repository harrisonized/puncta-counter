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
from puncta_counter.utils.common import dirname_n_times, expand_dataframe, collapse_dataframe
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
                                 # 'min_vol_ellipse',
                                 # 'circle',  # deprecated
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


    # ----------------------------------------------------------------------
    # Process Nuclei Data

    logger.info(f"Processing nuclei...")

    # Nuclei
    nuclei = pd.read_csv("data/nuclei.csv")
    nuclei.rename(columns={'ObjectNumber': 'NucleiObjectNumber'}, inplace=True)
    nuclei = preprocess_df(nuclei, nuclei_cols)  # rename columns and subset

    # add qc flags
    nuclei['effective_radius_nuclei'] = nuclei['area'].apply(lambda x: np.sqrt(x/np.pi))
    eccentricity_threshold = 0.69
    nuclei['potential_doublet'] = (nuclei['eccentricity'] >= eccentricity_threshold)
    nuclei_major_axis_length_threshold = 128
    nuclei['major_axis_too_long'] = (nuclei['major_axis_length'] >= nuclei_major_axis_length_threshold)

    # filter
    problem_nuclei = nuclei[
        (nuclei['potential_doublet'] == True) |
        (nuclei['major_axis_too_long'] == True)
    ]  # troubleshooting only
    nuclei_subset = nuclei[
        (nuclei['potential_doublet'] == False) &
        (nuclei['major_axis_too_long'] == False)
    ].copy()
    
    problem_nuclei.to_csv('data/problem_nuclei.csv', index=None)
    nuclei_subset.to_csv('data/nuclei_subset.csv', index=None)
    

    # ----------------------------------------------------------------------
    # Process Puncta Data

    logger.info(f"Processing puncta...")

    # Puncta
    puncta = pd.read_csv("data/puncta.csv")
    puncta = puncta.rename(columns={'ObjectNumber': 'PunctaObjectNumber'})
    puncta = preprocess_df(puncta, puncta_cols)  # rename columns and subset

    # reassign puncta to nuclei
    index_cols = ['image_number', 'parent_manual_nuclei']
    value_cols = [item for item in puncta.columns if item not in index_cols]  # we may not need all the value_cols
    puncta_short = collapse_dataframe(puncta, index_cols, value_cols)
    puncta_short['mean_center_x'] = puncta_short['center_x'].apply(np.mean)
    puncta_short['mean_center_y'] = puncta_short['center_y'].apply(np.mean)
    puncta_short['center'] = puncta_short[["mean_center_x", "mean_center_y"]].apply(list, axis=1)
    puncta_short = reassign_puncta_to_nuclei(puncta_short, nuclei)
    puncta = expand_dataframe(puncta_short, value_cols)
    puncta = puncta.sort_values(['image_number', 'puncta_object_number']).reset_index(drop=True)

    # can be used for confidence_ellipse weight
    # however, this ended up not actually working as well as vanilla intensity
    puncta['integrated_intensity_sq'] = puncta['integrated_intensity'].apply(lambda x: np.array(x)**2)

    # generate fill_alpha for plotting
    # (intensity-min_intensity) / (max_intensity-min_intensity) * (1-0.7) + 0.7
    puncta['fill_alpha'] = (puncta['mean_intensity']-puncta['mean_intensity'].min()) / (
        puncta['mean_intensity'].max()-puncta['mean_intensity'].min()
    )*(1-1/np.sqrt(2))+(1/np.sqrt(2))  # rescale to half the intensity ~1/np.sqrt(2) at the lowest brightness


    # add qc flags
    puncta['puncta_out_of_bounds'] = (
        (puncta["center_x"] < puncta["bounding_box_min_x_nuclei"]) |
        (puncta["center_x"] > puncta["bounding_box_max_x_nuclei"]) |
        (puncta["center_y"] < puncta["bounding_box_min_y_nuclei"]) |
        (puncta["center_y"] > puncta["bounding_box_max_y_nuclei"])
    )
    puncta = pd.merge(
        puncta,
        nuclei[["image_number", 'nuclei_object_number',
                'potential_doublet', 'major_axis_too_long',
                'path_name_tif', 'file_name_tif']],
        left_on=["image_number", 'nuclei_object_number'],
        right_on=['image_number', 'nuclei_object_number'],
        how='left',  
    ).rename(columns={
        'potential_doublet': 'nuclei_potential_doublet',
        'major_axis_too_long': 'nuclei_major_axis_too_long'
    })  # bring in nuclei flags


    # filter
    problem_puncta = puncta[
        (puncta['puncta_out_of_bounds'] == True) |
        (puncta['nuclei_potential_doublet'] == True) |
        (puncta['nuclei_major_axis_too_long'] == True)
    ]  # troubleshooting only
    puncta_subset = puncta[
        (puncta['puncta_out_of_bounds'] == False) &
        (puncta['nuclei_potential_doublet'] == False) &
        (puncta['nuclei_major_axis_too_long'] == False)
    ].copy()

    problem_puncta.to_csv('data/problem_puncta.csv', index=None)
    puncta_subset.to_csv('data/puncta_subset.csv', index=None)


    # ----------------------------------------------------------------------
    # Confidence Ellipse

    filename_for_image_number = dict(zip(nuclei_subset['image_number'], nuclei_subset['file_name_tif']))

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
        # )  
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
