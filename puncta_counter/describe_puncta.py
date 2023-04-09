#!/usr/bin/env python3

"""Takes nuclei.csv and puncta.csv files as inputs
1. Computes metrics on nuclei
2. Assigns puncta to nearest nuclei
3. Computes metrics on puncta
4. Splits nuclei into problem_nuclei and nuclei_subset (and same for puncta)
5. Uses first pass confidence_ellipse to filter outliers
6. Uses second pass confidence_ellipse to calculate diptest
7. Plot confidence ellipse
"""

import os
from os.path import join as ospj
from tqdm import tqdm
import argparse
import logging
import datetime as dt
import json
import numpy as np
import pandas as pd
import diptest

from puncta_counter.src.preprocessing import preprocess_df, reassign_puncta_to_nuclei, compute_puncta_metrics
from puncta_counter.src.summarize import (generate_circle, generate_ellipse, two_pass_confidence_ellipse, 
                                          compute_mahalanobis_distances,
                                          plot_nuclei_circles_puncta, plot_nuclei_ellipses_puncta)
from puncta_counter.utils.common import dirname_n_times, expand_dataframe, collapse_dataframe
from puncta_counter.utils.plotting import save_plot_as_png
from puncta_counter.utils.logger import configure_logger
from puncta_counter.etc.columns import ellipse_cols, nuclei_cols, puncta_cols, two_pass_ellipse_list_cols

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
    # Preprocess Nuclei Data

    logger.info(f"Preprocessing nuclei...")

    # nuclei
    # note that for nuclei, image_number and nuclei_object_number uniquely defines the nuclei
    nuclei = pd.read_csv("data/nuclei.csv")
    nuclei.rename(columns={'ObjectNumber': 'NucleiObjectNumber'}, inplace=True)
    nuclei = preprocess_df(nuclei, nuclei_cols)  # rename columns and subset

    # nuclei metrics
    nuclei['effective_radius_nuclei'] = nuclei['area'].apply(lambda x: np.sqrt(x/np.pi))
    nuclei['potential_doublet'] = (nuclei['eccentricity'] >= 0.69)  # eccentricity threshold
    nuclei['major_axis_too_long'] = (nuclei['major_axis_length'] >= 128)  # cells aren't this big
    

    # ----------------------------------------------------------------------
    # Preprocess Puncta Data

    logger.info(f"Preprocessing puncta...")

    # puncta
    # note that for puncta, image_number and puncta_object_number uniquely defines the puncta
    puncta = pd.read_csv("data/puncta.csv")
    puncta = puncta.rename(columns={'ObjectNumber': 'PunctaObjectNumber'})
    puncta = preprocess_df(puncta, puncta_cols)  # rename columns and subset

    # collapse, reassign nuclei, bring in extra nuclei columns, then expand
    # this algorithm has the potential to assign the same nuclei_object_number to two different parent_manual_nuclei
    # also adds nuclei qc cols to puncta
    puncta = reassign_puncta_to_nuclei(
        puncta,
        nuclei.rename(
            columns={'potential_doublet': 'nuclei_potential_doublet',
                     'major_axis_too_long': 'nuclei_major_axis_too_long'}
        ),
        extra_cols = [
            'path_name_tif', 'file_name_tif',
            "bounding_box_min_x", "bounding_box_max_x",
            "bounding_box_min_y", "bounding_box_max_y",
            'nuclei_potential_doublet', 'nuclei_major_axis_too_long',]
    )

    # 'puncta_out_of_bounds', 'num_clean_puncta_in_nucleus', 'high_background_puncta', 'fill_alpha'
    puncta = compute_puncta_metrics(puncta)  


    # ----------------------------------------------------------------------
    # Split Data

    logger.info(f"Splitting data...")


    # Split Puncta Data

    qc_cols = ['nuclei_potential_doublet', 'nuclei_major_axis_too_long', 'puncta_out_of_bounds', 'high_background_puncta']
    puncta_failed_qc = puncta[qc_cols].any(axis=1)
    problem_puncta, puncta_subset = puncta[puncta_failed_qc], puncta[~puncta_failed_qc].copy()

    if args.save:
        problem_puncta.to_csv('data/troubleshooting/problem_puncta.csv', index=None)
        puncta_subset.to_csv('data/puncta_subset.csv', index=None)


    # Split Nuclei Data

    # Add high_background_puncta to nuclei
    index_cols = ['image_number', 'nuclei_object_number']
    puncta_short = collapse_dataframe(puncta, index_cols+['high_background_puncta'])
    nuclei = pd.merge(nuclei, puncta_short[(puncta_short['high_background_puncta']==True)],
                      left_on=index_cols,  right_on=index_cols, how='left')
    nuclei['high_background_puncta'].fillna(False, inplace=True)

    qc_cols = ['potential_doublet', 'major_axis_too_long', 'high_background_puncta']
    nuclei_failed_qc = nuclei[qc_cols].any(axis=1)
    problem_nuclei, nuclei_subset = nuclei[nuclei_failed_qc], nuclei[~nuclei_failed_qc].copy()

    if args.save:
        problem_nuclei.to_csv('data/troubleshooting/problem_nuclei.csv', index=None)
        nuclei_subset.to_csv('data/nuclei_subset.csv', index=None)


    # Ellipse algos require collapsed dataframe
    puncta_short = collapse_dataframe(
        puncta_subset.rename(
            columns={'center_x': 'center_x_puncta',
                     'center_y': 'center_y_puncta'}),
        index_cols=['image_number', 'nuclei_object_number'],
        value_cols=['parent_manual_nuclei', 'puncta_object_number',  # we need these to map the ellipse back to the original data
                    'center_x_puncta', 'center_y_puncta', 'integrated_intensity']
    )


    # ----------------------------------------------------------------------
    # Confidence Ellipse


    logger.info(f"Generating confidence ellipses...")

    ellipses = two_pass_confidence_ellipse(puncta_short)  # generate ellipses

    if args.save:
        # fix this later
        # for col in two_pass_ellipse_cols:
        #     puncta_short[col] = puncta_short[col].apply(json.dumps)
        ellipses.to_csv('data/confidence_ellipse_test.csv', index=None)

    # ----------------------------------------------------------------------
    # Puncta Doublet Detection

    # ellipses['mahalanobis_x'] = ellipses['mahalanobis_coordinates_second_pass'].apply(lambda x: x[0])
    # ellipses.rename(columns={
    #     'major_axis_length_second_pass': 'major_axis_length',
    #     'minor_axis_length_second_pass': 'minor_axis_length'
    # }, inplace=True)

    # 'minor_axis_length', 'center_y', 'major_axis_length', 'orientation', 'center_x'


    # ellipses['num_puncta'] = ellipses['mahalanobis_x'].apply(len)
    # ellipses['eccentricity'] = np.sqrt(1-(ellipses['minor_axis_length']/ellipses['major_axis_length'])**2)

    # ellipses['dip'] = ellipses['mahalanobis_x'].apply(
    #     lambda x: diptest.diptest(np.array(x)) if len(x) > 3 else (np.nan, np.nan)  # diptest is invalid for three points
    # )
    # ellipses['diptest_dip'] = ellipses['dip'].apply(lambda x: x[0])
    # ellipses['diptest_pval'] = ellipses['dip'].apply(lambda x: x[1])

    # ellipses['puncta_doublet'] = (
    #     (ellipses['num_puncta'] > 3) &  # diptest is invalid for three points
    #     (ellipses['eccentricity'] > np.sqrt(1-1/2**2)) &  # major_axis_length at least 2x the minor_axis_length
    #     (ellipses['major_axis_length'] > 54) &   # the min minor_axis_length
    #     (ellipses['diptest_dip'] < 0.1) &  # note: the dip statistic is at least 0.5/n, where n is the number of items
    #     (ellipses['diptest_pval'] < 0.25)  # less than 25% confident that the distribution is univariate
    # )

    # # find a better way to do this...
    # ellipses['center_x'] = ellipses['mean_center_x']
    # ellipses['center_y'] = ellipses['mean_center_y']

    # # Save
    # if args.save:
    #     # ellipses_first_pass[
    #     #     ['image_number', 'nuclei_object_number'] + ellipse_cols + ['mahalanobis_outlier']
    #     # ].to_csv('data/troubleshooting/confidence_ellipses_first_pass.csv', index=None)
    #     ellipses[
    #         ['image_number', 'nuclei_object_number'] +
    #         ellipse_cols + 
    #         ['eccentricity', 'diptest_dip', 'diptest_pval', 'puncta_doublet']
    #     ].to_csv('data/confidence_ellipse.csv', index=None)


    # ----------------------------------------------------------------------
    # Plot Confidence Ellipses

    filename_for_image_number = dict(zip(nuclei_subset['image_number'], nuclei_subset['file_name_tif']))
    ellipses.rename(columns=dict(zip([f'{col}_second_pass' for col in ellipse_cols], ellipse_cols)), inplace=True)

    logger.info(f"Plotting...")
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
    # This might also be deprecated?

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
