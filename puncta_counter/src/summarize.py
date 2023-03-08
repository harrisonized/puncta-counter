import numpy as np
import pandas as pd
from scipy.stats import t

from puncta_counter.src.ellipse_algos import min_vol_ellipse, confidence_ellipse
from puncta_counter.src.utils import flatten_columns


# Functions
# # generate_circle
# # generate_ellipse


def generate_circle(puncta):
    """Generates an "effective radius", which is then plotted using plotly
    """

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


def generate_ellipse(puncta, algo='confidence_ellipse'):
    """Generates an ellipse, which comes with the following dimensions:
    ["center_x",
     "center_y",
     "minor_axis_length",
     "major_axis_length",
     "orientation"]

    This is plotted using Bokeh, because Plotly currently cannot plot rotated ellipses
    In the future, I may change this to matplotlib instead of Bokeh, as Bokeh is very resource heavy
    """
        
    if algo == 'min_vol_ellipse':
        ellipse_func = min_vol_ellipse
    elif algo == 'confidence_ellipse':
        ellipse_func = confidence_ellipse
    else:
        raise ValueError("Choose one: ['min_vol_ellipse', 'confidence_ellipse']")

    puncta['center'] = puncta[['center_x', 'center_y']].apply(list, axis=1)
    puncta['total_intensity'] = puncta["integrated_intensity"] * puncta["area"]

    puncta_summary = (
        puncta
        .groupby(['image_number', 'object_number'])[['center', "total_intensity"]]
        .agg(list)
        .reset_index()
    ).copy()

    puncta_summary[
        ["center_x", "center_y", "major_axis_length", "minor_axis_length", "orientation"]
    ] = pd.DataFrame(
        puncta_summary[["center", 'total_intensity']]
        .apply(lambda x: ellipse_func(
            np.transpose(np.array(x['center'])),
            aweights=x['total_intensity'], n_std=2.5,  # algo='confidence_ellipse'
            tolerance=0.05,  # algo='min_vol_ellipse'
        ), axis=1)
        .to_list()
    )
    
    return puncta_summary
