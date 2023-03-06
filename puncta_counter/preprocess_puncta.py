#!/usr/bin/env python3

"""Takes data/puncta.csv and data/nuclei.csv files and outputs a puncta_summary.csv
Plots the items in nuclei.csv and puncta.csv

This script is still in development and not ready for use
"""

import os
from os.path import join as ospj
from tqdm import tqdm
import argparse
import logging
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import t

from puncta_counter.src.preprocessing import reassign_puncta_to_nuclei
from puncta_counter.src.columns import nuclei_cols, puncta_cols
from puncta_counter.src.utils import dirname_n_times, camel_to_snake_case, flatten_columns
from puncta_counter.src.plotting import save_fig_as_png, plot_single_scatter
from puncta_counter.src.logger import configure_logger

script_name = 'run_puncta_counter'
this_dir = os.path.realpath(ospj(os.getcwd(), os.path.dirname(__file__)))
base_dir = dirname_n_times(this_dir, 1)
os.chdir(base_dir)


# Functions
# # parse_args
# # preprocess_df
# # generate_circle_bounding_boxes
# # plot_circle_puncta_using_plotly
# # main


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="")

    # io
    parser.add_argument("-i", "--input", dest="input_dir", default='data', action="store",
                        required=False, help="input directory")
    parser.add_argument("-o", "--output", dest="output_dir", default='data', action="store",
                        required=False, help="output directory")

    # other
    parser.add_argument("-l", "--log-dir", dest="log_dir", default='log', action="store",
                        required=False, help="set the directory for storing log files")
    parser.add_argument('--debug', default=False, action='store_false',
                        help='print debug messages')

    return parser.parse_args(args)


def preprocess_df(df, columns):
    """General preprocessing steps"""

    # clean column names
    df.columns = [camel_to_snake_case(col).replace("__", "_") for col in df.columns]
    df.columns = [
        (camel_to_snake_case(col)
         .replace("area_shape_", "")
         .replace('minimum', 'min')
         .replace("maximum", 'max')
         .replace('_masked_xist', "")
         .replace('_intensity', '')
        )
        for col in df.columns
    ]
    
    # puncta only
    intensity_cols = [col for col in df.columns if 'intensity_' in col]
    df = df.rename(columns=dict(
        zip(intensity_cols, ['_'.join(col.split('_')[::-1]) for col in intensity_cols])
    ))

    df_subset = df[columns].copy()

    return df_subset


def generate_cirle_bounding_boxes(puncta):

    puncta_summary = puncta.groupby(["image_number", "nuclei_object_number"]).agg(
        {
            "area": [sum, "count"],
            "integrated_intensity": sum,
            "center_x": [np.mean, np.std],
            "center_y": [np.mean, np.std],
        }
    ).reset_index()
    puncta_summary.columns = flatten_columns(puncta_summary.columns)

    # derive effective radius
    puncta_summary["center_std"] = np.sqrt(puncta_summary["center_x_std"]**2+puncta_summary["center_y_std"]**2)
    puncta_summary["effective_radius_puncta"] = puncta_summary["center_std"].apply(lambda x: x*t.ppf(0.90, 2))  # 90% CI

    # fillna
    puncta_summary.loc[puncta_summary["effective_radius_puncta"].isna(), "effective_radius_puncta"
    ] = puncta_summary.loc[puncta_summary["effective_radius_puncta"].isna(), "area_sum"].apply(
        lambda x: np.sqrt(x/np.pi)
    )
    puncta_summary["bounding_box_min_x"] = puncta_summary["center_x_mean"] - puncta_summary["effective_radius_puncta"]
    puncta_summary["bounding_box_max_x"] = puncta_summary["center_x_mean"] + puncta_summary["effective_radius_puncta"]
    puncta_summary["bounding_box_min_y"] = puncta_summary["center_y_mean"] - puncta_summary["effective_radius_puncta"]
    puncta_summary["bounding_box_max_y"] = puncta_summary["center_y_mean"] + puncta_summary["effective_radius_puncta"]

    return puncta_summary


def plot_circle_puncta_using_plotly(puncta_summary, puncta, image_number, title):

    # get nuclei boundaries
    shapes = list(
        puncta_summary.loc[
            (puncta_summary['image_number']==image_number)
            , ["bounding_box_min_x",
              "bounding_box_max_x",
              "bounding_box_min_y",
              "bounding_box_max_y",]
        ]
        .rename(columns={
                    "bounding_box_min_x": "x0",
                    "bounding_box_max_x": "x1",
                    "bounding_box_min_y": "y0",
                    "bounding_box_max_y": "y1",}
               )
        .apply(lambda x: {**{"type": "circle", 'xref':"x", 'yref':"y", 'line':{'width':1.5}}, **dict(x)}, axis=1)
    )

    # plot puncta
    fig = plot_single_scatter(
        puncta[puncta['image_number']==image_number].copy(),
        x='center_x',
        y='center_y',
        title=title,
        xlabel='x',
        ylabel='y'
    )

    fig.layout.update(
        xaxis = {'range': [-50, 1250], 'constrain': "domain"},
        yaxis = {'range': [1050, -50], 'scaleanchor': 'x', 'scaleratio': 1},
        shapes=shapes,
        height=700,
    )
    fig.update_traces(
        marker=dict(size=3)
    )

    return fig


def main(args=None):
    """
    """
    
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
    # Read and preprocess data

    # nuclei
    nuclei = pd.read_csv("data/nuclei.csv")
    nuclei = preprocess_df(nuclei, nuclei_cols)
    nuclei['effective_radius_nuclei'] = nuclei['area'].apply(lambda x: np.sqrt(x/np.pi))
    nuclei.to_csv('data/nuclei_subset.csv', index=None)
    
    # filter nuclei on eccentricity
    nuclei_subset = nuclei[
        (nuclei['eccentricity'] < 0.69)
        & (nuclei['major_axis_length'] < 128)
    ].copy()
    filename_from_image_number = dict(zip(nuclei_subset['image_number'], nuclei_subset['file_name_tif']))


    # puncta
    puncta = pd.read_csv("data/puncta.csv")
    puncta = preprocess_df(puncta, puncta_cols)
    puncta = reassign_puncta_to_nuclei(puncta, nuclei)
    puncta.to_csv('data/puncta_subset.csv', index=None)


    # ----------------------------------------------------------------------
    # Puncta Only using Plotly

    # filter puncta for nuclei
    puncta = pd.merge(
        left=nuclei_subset[["image_number", 'object_number']],
        right=puncta.loc[:, puncta.columns != 'object_number'],
        left_on=["image_number", 'object_number'],
        right_on=['image_number', 'nuclei_object_number'],
        how="left",
    ).dropna(subset=['nuclei_object_number'])  # left join without duplicates

    puncta_summary = generate_cirle_bounding_boxes(puncta)

    # plot puncta
    logger.info(f"generating figures")
    for image_number in tqdm(nuclei_subset['image_number'].unique()):
        file_name_tif = filename_from_image_number[image_number].split('.')[0]
        fig = plot_circle_puncta_using_plotly(puncta_summary, puncta, image_number, file_name_tif)
        save_fig_as_png(fig, f'figures/circle/{file_name_tif}.png', height=800, scale=1)


    # ----------------------------------------------------------------------
    # Nuclei and Puncta using Bokeh



    runtime = (dt.datetime.now() - start_time).total_seconds()
    logger.info(f"Script completed in {int(runtime // 60)} min {round(runtime % 60, 2)} sec")


if __name__ == "__main__":
    main()
