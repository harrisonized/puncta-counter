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
import numpy as np
import pandas as pd
import diptest

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

    nuclei = pd.read_csv("data/nuclei.csv")
    nuclei.rename(columns={'ObjectNumber': 'NucleiObjectNumber'}, inplace=True)
    nuclei = preprocess_df(nuclei, nuclei_cols)  # rename columns and subset

    # metrics
    nuclei['effective_radius_nuclei'] = nuclei['area'].apply(lambda x: np.sqrt(x/np.pi))
    nuclei['potential_doublet'] = (nuclei['eccentricity'] >= 0.69)  # eccentricity threshold
    nuclei['major_axis_too_long'] = (nuclei['major_axis_length'] >= 128)
    

    # ----------------------------------------------------------------------
    # Process Puncta Data

    logger.info(f"Processing puncta...")

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
    puncta_short = reassign_puncta_to_nuclei(puncta_short, nuclei)  # use all nuclei for this for better assignment
    puncta = expand_dataframe(puncta_short, value_cols)
    puncta = puncta.sort_values(['image_number', 'puncta_object_number']).reset_index(drop=True)

    # bring in nuclei flags and nuclei_object_number
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
    })


    # metrics

    # can be used for confidence_ellipse weight
    # however, this ended up not actually working as well as vanilla intensity
    # puncta['integrated_intensity_sq'] = puncta['integrated_intensity'].apply(lambda x: np.array(x)**2)

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

    # count puncta per nucleus
    puncta.set_index(['image_number', 'nuclei_object_number'], inplace=True)
    puncta['num_total_puncta'] = puncta.groupby(['image_number', 'nuclei_object_number'])['puncta_object_number'].count()
    puncta['num_clean_puncta'] = puncta[
        (puncta[
            ['puncta_out_of_bounds', 'nuclei_potential_doublet', 'nuclei_major_axis_too_long']
         ].any(axis=1)==False)
    ].groupby(['image_number', 'nuclei_object_number'])['puncta_object_number'].count()
    puncta['num_clean_puncta'] = puncta['num_clean_puncta'].fillna(0).astype(int)
    puncta.reset_index(inplace=True)

    puncta['high_background_puncta'] = (puncta['num_clean_puncta'] > 70)  # should filter nuclei associated with this too


    # ----------------------------------------------------------------------
    # Filter

    logger.info(f"Filtering...")
    index_cols=['image_number', 'nuclei_object_number', 'num_total_puncta', 'num_clean_puncta', 'high_background_puncta']
    puncta_short = collapse_dataframe(puncta, index_cols, value_cols=[])
    nuclei = pd.merge(
        nuclei, puncta_short,
        left_on=['image_number', 'nuclei_object_number'],
        right_on=['image_number', 'nuclei_object_number'],
        how='left'
    )
    nuclei[['num_total_puncta', 'num_clean_puncta']] = nuclei[['num_total_puncta', 'num_clean_puncta']].fillna(0)
    nuclei['high_background_puncta'] = nuclei['high_background_puncta'].fillna(False)

    nuclei_passed_qc = (
        (nuclei['potential_doublet'] == False) &
        (nuclei['major_axis_too_long'] == False) &
        (nuclei['high_background_puncta'] == False)
    )
    problem_nuclei = nuclei[~nuclei_passed_qc]  # troubleshooting only
    nuclei_subset = nuclei[nuclei_passed_qc].copy()

    if args.save:
        problem_nuclei.to_csv('data/troubleshooting/problem_nuclei.csv', index=None)
        nuclei_subset.to_csv('data/nuclei_subset.csv', index=None)


    puncta_passed_qc = (
        (puncta['puncta_out_of_bounds'] == False) &
        (puncta['nuclei_potential_doublet'] == False) &
        (puncta['nuclei_major_axis_too_long'] == False) &
        (puncta['high_background_puncta'] == False)
    )
    problem_puncta = puncta[~puncta_passed_qc]  # troubleshooting only
    puncta_subset = puncta[puncta_passed_qc].copy()

    if args.save:
        problem_puncta.to_csv('data/troubleshooting/problem_puncta.csv', index=None)
        puncta_subset.to_csv('data/puncta_subset.csv', index=None)


    # ----------------------------------------------------------------------
    # Confidence Ellipse

    filename_for_image_number = dict(zip(nuclei_subset['image_number'], nuclei_subset['file_name_tif']))

    logger.info(f"Generating confidence ellipses...")

    # first pass
    ellipses_first_pass = generate_ellipse(
        puncta_subset,
        algo='confidence_ellipse',
        aweights=None,
        n_std=1
    )  # note: this collapses the dataframe
    ellipses_first_pass = compute_mahalanobis_distances(ellipses_first_pass)  # note: this expands the dataframe
    ellipses_first_pass['center_x'] = ellipses_first_pass['centers'].apply(lambda x: x[0])
    ellipses_first_pass['center_y'] = ellipses_first_pass['centers'].apply(lambda x: x[1])
    ellipses_first_pass.drop(columns=['centers'], inplace=True)
    ellipses_first_pass['mahalanobis_outlier'] = (ellipses_first_pass['mahalanobis_distances'] >= 1.5)

    # second pass
    # detect doublets
    ellipses = generate_ellipse(
        ellipses_first_pass[(ellipses_first_pass['mahalanobis_outlier']==False)].copy(),
        algo='confidence_ellipse',
        aweights='integrated_intensity',
        n_std=2
    )
    ellipses = compute_mahalanobis_distances(ellipses)  # note: this expands the dataframe
    ellipses.rename(columns={'center_x': 'mean_center_x', 'center_y': 'mean_center_y'}, inplace=True)
    ellipses['center_x'] = ellipses['centers'].apply(lambda x: x[0])
    ellipses['center_y'] = ellipses['centers'].apply(lambda x: x[1])
    ellipses.drop(columns=['centers'], inplace=True)
    
    ellipses['mahalanobis_x'] = ellipses['mahalanobis_coordinates'].apply(lambda x: np.nan if x is None else x[0])
    ellipses['mahalanobis_y'] = ellipses['mahalanobis_coordinates'].apply(lambda x: np.nan if x is None else x[1])
    ellipses.drop(columns=['mahalanobis_distances'], inplace=True)
    ellipses = collapse_dataframe(
        ellipses,
        index_cols=['image_number', 'nuclei_object_number'] +
            ['major_axis_length', 'minor_axis_length', 'orientation', 'mean_center_x', 'mean_center_y'],
        value_cols=['center_x', 'center_y', 'mahalanobis_x', 'mahalanobis_y']
    )
    

    # ----------------------------------------------------------------------
    # Puncta Doublet Detection

    ellipses['num_puncta'] = ellipses['mahalanobis_x'].apply(len)
    ellipses['eccentricity'] = np.sqrt(1-(ellipses['minor_axis_length']/ellipses['major_axis_length'])**2)

    ellipses['dip'] = ellipses['mahalanobis_x'].apply(
        lambda x: diptest.diptest(np.array(x)) if len(x) > 3 else (np.nan, np.nan)
    )
    ellipses['diptest_dip'] = ellipses['dip'].apply(lambda x: x[0])
    ellipses['diptest_pval'] = ellipses['dip'].apply(lambda x: x[1])

    ellipses['puncta_doublet'] = (
        (ellipses['eccentricity'] > np.sqrt(1-1/2**2)) &   # major_axis_length at least 2x the minor_axis_length
        (ellipses['major_axis_length'] > 54) &   # the min minor_axis_length
        (ellipses['diptest_dip'] < 0.1) &
        (ellipses['diptest_pval'] < 0.25)  # less than 25% confident that the distribution is univariate
    )

    # find a better way to do this...
    ellipses['center_x'] = ellipses['mean_center_x']
    ellipses['center_y'] = ellipses['mean_center_y']

    # Save
    if args.save:
        # ellipses_first_pass[
        #     ['image_number', 'nuclei_object_number'] + ellipse_cols + ['mahalanobis_outlier']
        # ].to_csv('data/troubleshooting/confidence_ellipses_first_pass.csv', index=None)
        ellipses[
            ['image_number', 'nuclei_object_number'] +
            ellipse_cols + 
            ['eccentricity', 'diptest_dip', 'diptest_pval', 'puncta_doublet']
        ].to_csv('data/confidence_ellipse.csv', index=None)


    # ----------------------------------------------------------------------
    # Plot Confidence Ellipse

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
