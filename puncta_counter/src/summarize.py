import numpy as np
import pandas as pd
from scipy.stats import t

from puncta_counter.utils.common import flatten_columns, expand_dataframe
from puncta_counter.utils.ellipse_algos import (confidence_ellipse, min_vol_ellipse,
	                                            mahalanobis_transform, compute_euclidean_distance_from_origin)
from puncta_counter.utils.plotting import (plot_circle_using_bokeh,
                                           plot_ellipse_using_bokeh)
from puncta_counter.etc.columns import ellipse_cols


# Functions
# # generate_ellipse
# # generate_circle
# # compute_mahalanobis_distances
# # plot_nuclei_ellipses_puncta
# # plot_nuclei_circles_puncta

# Note
# # ellipse_cols = ["center_x", "center_y", "major_axis_length", "minor_axis_length", "orientation"]


def generate_ellipse(
        puncta,
        algo='confidence_ellipse',
        aweights=None,
        n_std=2,  # algo='confidence_ellipse'
        tolerance=0.01  # algo='min_vol_ellipse'
    ):
    """Generates an ellipse, which comes with the following dimensions:
    ["center_x",
     "center_y",
     "minor_axis_length",
     "major_axis_length",
     "orientation"]
    """

    puncta['centers'] = puncta[['center_x', 'center_y']].apply(list, axis=1)
    cols = list(set(["centers", "integrated_intensity"]+([] if aweights is None else [aweights])))
    
    ellipses = (
        puncta
        .groupby(['image_number', 'nuclei_object_number'])[cols]
        .agg(list)
        .reset_index()
    ).copy()
    
    # convert this: [[330.3, 52.7], [329.6, 54.8], [333.9, 54.8, 54.9]]
    # to this: [[330.3, 329.6 , 333.9],
    #           [52.7, 54.8, 54.9]]    
    ellipses['centers'] = ellipses['centers'].apply(lambda x: np.transpose(np.array(x)))
    
    if algo == 'min_vol_ellipse':
        ellipses[ellipse_cols] = pd.DataFrame(
            ellipses["centers"]
            .apply(lambda x: min_vol_ellipse(x, tolerance=tolerance))
            .to_list()
        )

    elif algo == 'confidence_ellipse':
        ellipses[ellipse_cols] = pd.DataFrame(
            ellipses[cols]
            .apply(lambda x: confidence_ellipse(
                x['centers'],
                aweights=aweights if aweights is None else x[aweights],
                n_std=n_std,
            ), axis=1)
            .to_list()
        )
        
    else:
        raise ValueError("Choose one: ['min_vol_ellipse', 'confidence_ellipse']")

    # The chosen ellipse algos work for normal x, y coordinates.
    # Images have a reversed y coordinate.
    ellipses['orientation'] = -ellipses['orientation']
    
    return ellipses


def generate_circle(puncta):
    """Generates an "effective radius" by assuming that the standard deviations in each dimension are uncorrelated
    This was a first pass used to build plotting capabilities
    This algorithm should be deprecated, as it is highly sensitive to outliers
    """

    circles = puncta.groupby(["image_number", "nuclei_object_number"]).agg(
        {
            "area": [sum, "count"],
            "integrated_intensity": sum,
            "center_x": [np.mean, np.std],
            "center_y": [np.mean, np.std],
        }
    ).reset_index()
    circles.columns = flatten_columns(circles.columns)

    # derive effective radius
    circles["center_std"] = np.sqrt(circles["center_x_std"]**2+circles["center_y_std"]**2)
    circles["effective_radius_puncta"] = circles["center_std"].apply(lambda x: x*t.ppf(0.90, 2))  # 90% CI

    # fillna
    circles.loc[circles["effective_radius_puncta"].isna(), "effective_radius_puncta"
    ] = circles.loc[circles["effective_radius_puncta"].isna(), "area_sum"].apply(
        lambda x: np.sqrt(x/np.pi)
    )
    circles["bounding_box_min_x"] = circles["center_x_mean"] - circles["effective_radius_puncta"]
    circles["bounding_box_max_x"] = circles["center_x_mean"] + circles["effective_radius_puncta"]
    circles["bounding_box_min_y"] = circles["center_y_mean"] - circles["effective_radius_puncta"]
    circles["bounding_box_max_y"] = circles["center_y_mean"] + circles["effective_radius_puncta"]

    return circles


def compute_mahalanobis_distances(ellipses, explode=True):
    
    # center, rotate, and rescale the coordinates of the puncta (centers)
    # such that the x-axis is major_axis and the y-axis is the minor_axis
    ellipses['mahalanobis_coordinates'] = ellipses[
        ['centers', 'major_axis_length', 'minor_axis_length', 'orientation']
    ].apply(lambda x: mahalanobis_transform(
        x['centers'], x['major_axis_length'], x['minor_axis_length'], x['orientation']
    ), axis=1)
    ellipses['mahalanobis_distances'] = ellipses['mahalanobis_coordinates'].apply(
        compute_euclidean_distance_from_origin
    )

    if explode:
        # For expand_dataframe, need arrays to be in this format:
        #     [[330.3, 52.7], [329.6, 54.8], [333.9, 54.8, 54.9]]
        # Not this:
        #     [[330.3, 329.6 , 333.9],
        #      [52.7, 54.8, 54.9]]
        for col in ['centers', 'mahalanobis_coordinates']:
            ellipses[col] = ellipses[col].apply(lambda x: np.transpose(np.array(x)))
        
        # make it such that each row is a single entity
        ellipses = expand_dataframe(
            ellipses,
            ['centers', 'integrated_intensity', 'mahalanobis_coordinates', 'mahalanobis_distances']
        )
        
    return ellipses


def plot_nuclei_ellipses_puncta(nuclei, ellipses, puncta, title=None):
    """Construct a resemblance of the original image from the extracted features
    """

    # nuclei
    nuclei_data = nuclei[["nuclei_object_number"]+ellipse_cols]
    plot = plot_ellipse_using_bokeh(
        nuclei_data,
        nuclei_data,
        x='center_x',
        y='center_y',
        height="major_axis_length",
        width="minor_axis_length",
        angle="orientation",
        angle_units='deg',
        text="nuclei_object_number",
        title=title,
        fill_color='#000fff',  # blue
    )

    # confidence_ellipse
    # may need to redo this...
    ellipses_data = ellipses[["nuclei_object_number"]+ellipse_cols]
    plot = plot_ellipse_using_bokeh(
        ellipses_data,
        ellipses_data,
        x='center_x',
        y='center_y',
        height="major_axis_length",
        width="minor_axis_length",
        angle="orientation",
        angle_units='deg',
        text="nuclei_object_number",
        text_color='orange',
        fill_color='#097969',  # green
        fill_alpha=0.9,
        line_alpha=0,
        plot=plot
    )

    # puncta
    puncta_data = puncta[["nuclei_object_number"] + ellipse_cols + ["fill_alpha"]]
    plot = plot_ellipse_using_bokeh(
        puncta_data,
        x='center_x',
        y='center_y',
        height="major_axis_length",
        width="minor_axis_length",
        angle="orientation",
        angle_units='deg',
        fill_color='#ff2b00',  # red
        fill_alpha='fill_alpha',
        line_alpha=0,
        plot=plot
    )

    return plot


def plot_nuclei_circles_puncta(nuclei, circles, puncta, title=None):
    """Construct a resemblance of the original image from the extracted features
    """

    # nuclei
    nuclei_data = nuclei[["nuclei_object_number"] + ellipse_cols]
    plot = plot_ellipse_using_bokeh(
        nuclei_data,
        nuclei_data,
        x='center_x',
        y='center_y',
        height="major_axis_length",
        width="minor_axis_length",
        angle="orientation",
        angle_units='deg',
        text="nuclei_object_number",
        title=title,
        fill_color='#000fff',  # blue
    )

    # circle
    circles_data = circles[
        ["nuclei_object_number", "center_x_mean", "center_y_mean", "effective_radius_puncta"]
    ]
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
    puncta_data = puncta[["nuclei_object_number"] + ellipse_cols + ["fill_alpha"]]
    plot = plot_ellipse_using_bokeh(
        puncta_data,
        x='center_x',
        y='center_y',
        height="major_axis_length",
        width="minor_axis_length",
        angle="orientation",
        angle_units='deg',
        fill_color='#ff2b00',  # red
        fill_alpha='fill_alpha',
        line_alpha=0,
        plot=plot
    )

    return plot
