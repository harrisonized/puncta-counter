import numpy as np
import pandas as pd
from puncta_counter.src.columns import puncta_cols


# Functions
# # find_nearest_point
# # reassign_puncta_to_nuclei


def find_nearest_point(point, points:list):
    """O(n^2) algorithm to find the nearest point
    Can make this faster with binary search on one of the variables
    However, since this is a small dataset (20 nuclei per image), this is whatever
    
    >>> find_nearest_point(
        point=(281.415801, 135.945238),
        points=[(693.094713, 59.080090), (295.184921, 118.996760), (282.528024, 182.998269)],
    )
    (295.184921, 118.99676)
    """
    
    d = np.inf
    for x, y in points:
        d_current = np.sqrt((point[0]-x)**2+(point[1]-y)**2)
        if d_current < d:
            closest_point = (x, y)
            d = d_current
        
    return closest_point


def reassign_puncta_to_nuclei(puncta, nuclei):
    """If the nuclei and puncta were generated at different times, they could be numbered differently
    This reassigns the nuclei_id so that the indexes line up
    """

    puncta_centers = (
        puncta
        .groupby(["image_number", "parent_manual_nuclei"])[["center_x", "center_y"]]
        .mean()
        .reset_index()
    )
    puncta_centers["center"] = puncta_centers[["center_x", "center_y"]].apply(list, axis=1)


    # use find_nearest_point to find the center of the closest nuclei
    # there are more nuclei than puncta, so this is fine
    puncta_centers[["closest_nuclei_x", "closest_nuclei_y"]] = pd.DataFrame(
        puncta_centers[["image_number", "center"]].apply(
        lambda x: find_nearest_point(
            point=x["center"],
            points=nuclei.loc[(nuclei["image_number"]==x["image_number"]),
                              ["center_x", "center_y"]].to_records(index=False)
        )
        , axis=1).to_list(),
        columns=["closest_nuclei_x", "closest_nuclei_y"],
    )

    # left join nuclei_table on closest_nuclei_x and closest_nuclei_y
    puncta_centers["nuclei_object_number"] = pd.merge(
        left=puncta_centers[["closest_nuclei_x", "closest_nuclei_y",
                             "image_number", "parent_manual_nuclei"]],
        right=nuclei[["center_x", "center_y", "image_number", "object_number"]],
        left_on=["closest_nuclei_x", "closest_nuclei_y", "image_number",],
        right_on=["center_x", "center_y", "image_number",],
        how="left",
        suffixes=("", "_nuclei")
    )["object_number"]


    # add back to puncta
    puncta = pd.merge(
        left=puncta[puncta_cols],
        right=puncta_centers[["image_number", "parent_manual_nuclei", "nuclei_object_number"]],
        left_on=["image_number", "parent_manual_nuclei"],
        right_on=["image_number", "parent_manual_nuclei",],
        how="left",
        suffixes=("", "_")
    )


    # filter puncta that are too far away from the nuclei
    puncta = pd.merge(
        left=puncta[list(puncta_cols)+["nuclei_object_number"]],
        right=nuclei[["image_number", "object_number",
                      "bounding_box_min_x", "bounding_box_max_x",
                      "bounding_box_min_y", "bounding_box_max_y"]],
        left_on=["image_number", "nuclei_object_number"],
        right_on=["image_number", "object_number"],
        how="left",
        suffixes=("", "_nuclei")
    )  # left join nuclei data

    puncta = puncta[
        (puncta["center_x"] >= puncta["bounding_box_min_x_nuclei"]) & 
        (puncta["center_x"] <= puncta["bounding_box_max_x_nuclei"]) &
        (puncta["center_y"] >= puncta["bounding_box_min_y_nuclei"]) &
        (puncta["center_y"] <= puncta["bounding_box_max_y_nuclei"])
    ].copy()  # filter


    # regenerate puncta_centers using filtered data
    puncta_centers = (
        puncta
        .groupby(["image_number", "nuclei_object_number"])[["center_x", "center_y"]]
        .mean()
        .reset_index()
    )

    return puncta
