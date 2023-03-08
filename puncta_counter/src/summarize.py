import numpy as np
import pandas as pd
from scipy.stats import t

from puncta_counter.utils.common import flatten_columns
from puncta_counter.utils.ellipse_algos import confidence_ellipse, min_vol_ellipse
from puncta_counter.utils.plotting import (plot_circle_using_bokeh,
                                           plot_ellipse_using_bokeh)


# Functions
# # generate_circle
# # generate_ellipse
# # plot_nuclei_circles_puncta
# # plot_nuclei_ellipses_puncta


def generate_circle(puncta):
    """Generates an "effective radius" by assuming that the standard deviations in each dimension are uncorrelated
    This was a first pass used to build plotting capabilities
    This algorithm should be deprecated, as it is highly sensitive to outliers
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
    In the future, I may change this to matplotlib instead of Bokeh, as Bokeh is very resource heavy
    """

    puncta['center'] = puncta[['center_x', 'center_y']].apply(list, axis=1)
    puncta['total_intensity'] = puncta["integrated_intensity"] * puncta["area"]

    puncta_summary = (
        puncta
        .groupby(['image_number', 'object_number'])[['center', "total_intensity"]]
        .agg(list)
        .reset_index()
    ).copy()


    if algo == 'confidence_ellipse':
        puncta_summary[
            ["center_x", "center_y", "major_axis_length", "minor_axis_length", "orientation"]
        ] = pd.DataFrame(
            puncta_summary[["center", 'total_intensity']]
            .apply(lambda x: confidence_ellipse(
                np.transpose(np.array(x['center'])),
                aweights=x['total_intensity'], n_std=2.5,  # algo='confidence_ellipse'
                tolerance=0.05,  # algo='min_vol_ellipse'
            ), axis=1)
            .to_list()
        )

    elif algo == 'min_vol_ellipse':
        puncta_summary[
            ["center_x", "center_y", "major_axis_length", "minor_axis_length", "orientation"]
        ] = pd.DataFrame(
            puncta_summary["center"]
            .apply(lambda x: np.transpose(np.array(x)))
            .apply(lambda x: min_vol_ellipse(x, tolerance=0.05))
            .to_list()
        )

    else:
        raise ValueError("Choose one: ['min_vol_ellipse', 'confidence_ellipse']")
    
    return puncta_summary


def plot_nuclei_circles_puncta(nuclei, circles, puncta, title=None):
    """Build plot
    """

    # nuclei
    nuclei_data = nuclei[["object_number", "center_x", "center_y", "major_axis_length", "minor_axis_length", "angle"]]
    plot = plot_ellipse_using_bokeh(
        nuclei_data,
        nuclei_data,
        x='center_x',
        y='center_y',
        height="major_axis_length",
        width="minor_axis_length",
        angle='angle',
        text="object_number",
        title=title,
        fill_color='#000fff',  # blue
    )

    # circle
    circles_data = circles[["nuclei_object_number", "center_x_mean", "center_y_mean", "effective_radius_puncta"]]
    plot = plot_circle_using_bokeh(
        circles_data,
        circles_data,
        x='center_x_mean',
        y='center_y_mean',
        size="effective_radius_puncta",
        text="nuclei_object_number",
        text_color='orange',
        fill_color='#097969',  # green
        line_alpha=0,
        plot=plot
    )

    # puncta
    puncta_data = puncta[["object_number", "center_x", "center_y", "major_axis_length", "minor_axis_length", "angle"]]
    plot = plot_ellipse_using_bokeh(
        puncta_data,
        x='center_x',
        y='center_y',
        height="major_axis_length",
        width="minor_axis_length",
        angle='angle',
        fill_color='#ff2b00',  # red
        line_alpha=0,
        plot=plot
    )

    return plot


def plot_nuclei_ellipses_puncta(nuclei, ellipses, puncta, title=None):
    """Build plot
    """

    # nuclei
    nuclei_data = nuclei[["object_number", "center_x", "center_y", "major_axis_length", "minor_axis_length", "angle"]]
    plot = plot_ellipse_using_bokeh(
        nuclei_data,
        nuclei_data,
        x='center_x',
        y='center_y',
        height="major_axis_length",
        width="minor_axis_length",
        angle='angle',
        text="object_number",
        title=title,
        fill_color='#000fff',  # blue
    )

    # confidence_ellipse
    ellipses_data = ellipses[["object_number", "center_x", "center_y", "major_axis_length", "minor_axis_length", "angle"]]
    plot = plot_ellipse_using_bokeh(
        ellipses_data,
        ellipses_data,
        x='center_x',
        y='center_y',
        height="major_axis_length",
        width="minor_axis_length",
        angle='angle',
        text="object_number",
        text_color='orange',
        fill_color='#097969',  # green
        line_alpha=0,
        plot=plot
    )

    # puncta
    puncta_data = puncta[["object_number", "center_x", "center_y", "major_axis_length", "minor_axis_length", "angle"]]
    plot = plot_ellipse_using_bokeh(
        puncta_data,
        x='center_x',
        y='center_y',
        height="major_axis_length",
        width="minor_axis_length",
        angle='angle',
        fill_color='#ff2b00',  # red
        line_alpha=0,
        plot=plot
    )

    return plot
