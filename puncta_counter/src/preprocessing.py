import numpy as np
import pandas as pd
from puncta_counter.etc.columns import puncta_cols
from puncta_counter.utils.common import camel_to_snake_case
from puncta_counter.utils.ellipse_algos import find_nearest_point


# Functions
# # preprocess_df
# # reassign_puncta_to_nuclei


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


def reassign_puncta_to_nuclei(puncta, nuclei):
    """If the nuclei and puncta were generated at different times, they could be numbered differently
    This reassigns the parent_manual_nuclei so that it lines up with nuclei_object_number
    """

    # use the find_nearest_point algorithm to find the center of the closest nuclei
    # there are more nuclei than puncta, so this is fine
    puncta[["closest_nuclei_x", "closest_nuclei_y"]] = pd.DataFrame(
        puncta[["image_number", "center"]].apply(
        lambda x: find_nearest_point(
            point=x["center"],
            points=nuclei.loc[(nuclei["image_number"]==x["image_number"]),
                              ["center_x", "center_y"]].to_records(index=False)
        )
        , axis=1).to_list(),
        columns=["closest_nuclei_x", "closest_nuclei_y"],
    )

    # left join nuclei_table on closest_nuclei_x and closest_nuclei_y
    puncta = pd.merge(
        left=puncta,
        right=nuclei[[
            "center_x", "center_y", "image_number", "nuclei_object_number",
            "bounding_box_min_x", "bounding_box_max_x",
            "bounding_box_min_y", "bounding_box_max_y"
        ]],
        left_on=["image_number", "closest_nuclei_x", "closest_nuclei_y"],
        right_on=["image_number", "center_x", "center_y"],
        how="left",
        suffixes=("", "_nuclei")
    )
    
    puncta = puncta.drop(columns=['closest_nuclei_x', 'closest_nuclei_y'])

    return puncta
