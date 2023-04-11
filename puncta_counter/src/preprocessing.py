import numpy as np
import pandas as pd
from puncta_counter.etc.columns import index_cols, puncta_qc_cols
from puncta_counter.utils.common import camel_to_snake_case
from puncta_counter.utils.clustering_algos import find_nearest_point
from puncta_counter.utils.common import collapse_dataframe, expand_dataframe


# Functions
# # preprocess_df
# # reassign_puncta_to_nuclei
# # compute_puncta_metrics
# # merge_exclusion_list


def preprocess_df(df, columns):
    """General preprocessing steps"""

    # clean column names
    df.columns = [camel_to_snake_case(col).replace("__", "_") for col in df.columns]

    # replace extra prefixes and suffixes
    df.columns = [
        (camel_to_snake_case(col)
         .replace("area_shape_", "")
         .replace('_masked_xist', "")
         .replace('_xist', "")
         .replace('_intensity', '')
         .replace('minimum', 'min')
         .replace("maximum", 'max')
         .replace("parent_manual_nuclei", "parent_nuclei_object_number")
         .replace('parent_filtered_nuclei', "parent_nuclei_object_number")
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


def merge_nuclei_and_puncta(puncta, nuclei, extra_cols=[], reassign_puncta=False):
    """If the nuclei and puncta were generated at different times, they could be numbered differently
    
    This uses the find_nearest_point algorithm to match each (center_x_puncta, center_y_puncta)
    to the nearest (center_x_nuclei, center_y_nuclei) coordinate, then uses that coordinate to assign
    a nuclei_object_number.
    
    In general, this shouldn't be an issue.
    """

    puncta_index_cols = ['image_number', 'parent_nuclei_object_number']
    value_cols = [item for item in puncta.columns if item not in puncta_index_cols]
    
    puncta_short = collapse_dataframe(puncta, puncta_index_cols, value_cols)  # collapse so that each row is one cloud
    puncta_short['num_total_puncta_in_nucleus'] = puncta_short['puncta_object_number'].apply(len)
    puncta_short['mean_center_x_puncta'] = puncta_short['center_x'].apply(np.mean)
    puncta_short['mean_center_y_puncta'] = puncta_short['center_y'].apply(np.mean)

    if reassign_puncta:

        # use the find_nearest_point algorithm to find the center of the closest nuclei
        # there are more nuclei than puncta, so this is fine
        puncta_short[["_center_x_nuclei", "_center_y_nuclei"]] = pd.DataFrame(
            puncta_short[["image_number", "mean_center_x_puncta", "mean_center_y_puncta"]].apply(
            lambda x: find_nearest_point(
                point=(x['mean_center_x_puncta'], x['mean_center_y_puncta']),
                points=nuclei.loc[(nuclei["image_number"]==x["image_number"]),
                                  ["center_x", "center_y"]].to_records(index=False)
            )
            , axis=1).to_list(),
            columns=["_center_x_nuclei", "_center_y_nuclei"],
        )

        # left join nuclei_table on closest_nuclei_x and closest_nuclei_y
        puncta_short = pd.merge(
            left=puncta_short,
            right=nuclei[index_cols+["center_x", "center_y"]+extra_cols],
            left_on=["image_number", "_center_x_nuclei", "_center_y_nuclei"],
            right_on=["image_number", "center_x", "center_y"],
            how="left",
            suffixes=("", "_nuclei")
        ).drop(columns=["_center_x_nuclei", "_center_y_nuclei"])

    else:

        puncta_short = pd.merge(
            left=puncta_short,
            right=nuclei[index_cols+["center_x", "center_y"]+extra_cols],
            left_on=["image_number", "parent_nuclei_object_number"],
            right_on=index_cols,
            how="left",
            suffixes=("", "_nuclei")
        )

    puncta = expand_dataframe(puncta_short, value_cols)
    puncta = puncta.sort_values(puncta_index_cols).reset_index(drop=True)

    return puncta


def compute_puncta_metrics(puncta):
    """General purpose dumping ground for metrics
    'puncta_out_of_bounds', 'num_clean_puncta_in_nucleus', 'high_background_puncta', 'fill_alpha'
    """
    
    # puncta metrics
    puncta['puncta_out_of_bounds'] = (
        (puncta["center_x"] < puncta["bounding_box_min_x_nuclei"]) |
        (puncta["center_x"] > puncta["bounding_box_max_x_nuclei"]) |
        (puncta["center_y"] < puncta["bounding_box_min_y_nuclei"]) |
        (puncta["center_y"] > puncta["bounding_box_max_y_nuclei"]))

    # find background puncta
    # puncta_qc_cols[0:3] = ['nuclei_potential_doublet', 'nuclei_major_axis_too_long', 'puncta_out_of_bounds']
    puncta.set_index(index_cols, inplace=True)
    puncta['num_clean_puncta_in_nucleus'] = (puncta
        .loc[(puncta[puncta_qc_cols[0:3]].any(axis=1)==False)]
        .groupby(index_cols)['puncta_object_number']
        .count()
    )
    puncta['num_clean_puncta_in_nucleus'] = puncta['num_clean_puncta_in_nucleus'].fillna(0).astype(int)
    puncta.reset_index(inplace=True)

    puncta['high_background_puncta'] = (puncta['num_clean_puncta_in_nucleus'] > 70)

    # generate fill_alpha (for plotting only)
    # (intensity-min_intensity) / (max_intensity-min_intensity) * (1-0.7) + 0.7
    puncta['fill_alpha'] = (puncta['mean_intensity']-puncta['mean_intensity'].min()) / (
        puncta['mean_intensity'].max()-puncta['mean_intensity'].min()
    )*(1-1/np.sqrt(2))+(1/np.sqrt(2))  # rescale to half the intensity ~1/np.sqrt(2) at the lowest brightness
    
    return puncta


def merge_exclusion_list(df, exclusion_list):
    """Convenience function, because you cannot set a dataframe from a for loop
    """
    df = pd.merge(
        df, exclusion_list[index_cols+['manually_exclude']],
        left_on=index_cols, right_on=index_cols, how='left'
    )
    df['manually_exclude'].fillna(False, inplace=True)
    
    return df
