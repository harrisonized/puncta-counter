from functools import reduce
import numpy as np
import pandas as pd
from scipy.stats import t
import diptest

from puncta_counter.utils.common import flatten_columns, expand_dataframe, collapse_dataframe
from puncta_counter.utils.clustering_algos import two_cluster_kmeans
from puncta_counter.utils.ellipse_algos import (confidence_ellipse, min_vol_ellipse,
	                                            mahalanobis_transform, compute_euclidean_distance_from_origin)
from puncta_counter.utils.plotting import (plot_circle_using_bokeh,
                                           plot_ellipse_using_bokeh)
from puncta_counter.etc.columns import ellipse_cols


# Functions
# # compute_mahalanobis_distances
# # generate_ellipse
# # two_pass_confidence_ellipse
# # compute_diptest
# # plot_nuclei_ellipses_puncta

# Note
# # ellipse_cols = ["center_x", "center_y", "major_axis_length", "minor_axis_length", "orientation"]


# might deprecate
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


def generate_ellipse(
        puncta_short,  # note: This must be a COLLAPSED ellipse!
        algo='confidence_ellipse',
        suffix='',
        aweights=None,
        n_std=1,
        mahalanobis_threshold = 1.5,
        # algo='min_vol_ellipse'
        tolerance=0.01,
    ):
    """Requires a condensed dataframe

    Generates an ellipse, which comes with the following dimensions:
    ["center_x",
     "center_y",
     "minor_axis_length",
     "major_axis_length",
     "orientation"]
    """

    if algo == 'confidence_ellipse':
        # The bread-and-butter of this pipeline

        # compute confidence ellipse
        cols = [f'{col}{suffix}' for col in ellipse_cols]
        puncta_short['confidence_ellipse'] = (
            puncta_short.apply(
                lambda x: confidence_ellipse(
                    x[f'center_puncta{suffix}'],
                    aweights=aweights if aweights is None else x[aweights],
                    n_std=n_std), axis=1)
        )
        for idx, col in enumerate(cols):
            puncta_short[col] = puncta_short['confidence_ellipse'].apply(lambda x: x[idx])
        puncta_short.drop(columns=['confidence_ellipse'], inplace=True)

        # compute ellipse metrics
        puncta_short[f'num_puncta{suffix}'] = puncta_short[f'center_puncta{suffix}'].apply(lambda x: x.shape[1])
        puncta_short[f'eccentricity{suffix}'] = np.sqrt(
            1-(puncta_short[f'minor_axis_length{suffix}']/puncta_short[f'major_axis_length{suffix}'])**2
        )

        # mahalanobis transform
        # center, rotate, and rescale the coordinates of the puncta_short (centers)
        # such that the x-axis is major_axis and the y-axis is the minor_axis
        ellipse_args = [
            f'{col}{suffix}' for col in
            ['center_puncta', 'major_axis_length', 'minor_axis_length', 'orientation']
        ]
        puncta_short[f'mahalanobis_coordinates{suffix}'] = puncta_short[ellipse_args].apply(
            lambda x: mahalanobis_transform(*x), axis=1)
        puncta_short[f'mahalanobis_distances{suffix}'] = puncta_short[f'mahalanobis_coordinates{suffix}'].apply(
            compute_euclidean_distance_from_origin)
        puncta_short[f'is_mahalanobis_outlier{suffix}'] = puncta_short[f'mahalanobis_distances{suffix}'].apply(
            lambda x: np.array(x >= mahalanobis_threshold))
        puncta_short[f'any_mahalanobis_outlier{suffix}'] = puncta_short[f'is_mahalanobis_outlier{suffix}'].apply(
            lambda x: np.any(x==True)
        )

    elif algo == 'min_vol_ellipse':

        cols = [f'{col}{suffix}' for col in ellipse_cols]
        puncta_short['center_puncta'] = puncta_short[['center_x_puncta', 'center_y_puncta']].apply(lambda x: np.array(list(x)), axis=1)
        puncta_short['min_vol_ellipse'] = puncta_short['center_puncta'].apply(
            lambda x: min_vol_ellipse(x, tolerance=tolerance)
        )
        for idx, col in enumerate(cols):
            puncta_short[col] = puncta_short['min_vol_ellipse'].apply(lambda x: x[idx])
        puncta_short.drop(columns=['min_vol_ellipse'], inplace=True)

    elif algo=='circle':
        """Generates an "effective radius" by assuming that the standard deviations in each dimension are uncorrelated
        This was a first pass used to build plotting capabilities
        This algorithm should be deprecated, as it is highly sensitive to outliers
        """

        # compute some metrics
        puncta_short['total_area'] = puncta_short['area'].apply(lambda x: sum(x))
        puncta_short['center_x_mean'] = puncta_short['center_x_puncta'].apply(lambda x: np.mean(x))
        puncta_short['center_y_mean'] = puncta_short['center_y_puncta'].apply(lambda x: np.mean(x))
        puncta_short['center_x_std'] = puncta_short['center_x_puncta'].apply(lambda x: np.std(x))
        puncta_short['center_y_std'] = puncta_short['center_y_puncta'].apply(lambda x: np.std(x))
        
        # derive effective radius
        puncta_short["center_std"] = np.sqrt(puncta_short["center_x_std"]**2+puncta_short["center_y_std"]**2)
        puncta_short[f"effective_radius{suffix}"] = puncta_short["center_std"].apply(lambda x: x*t.ppf(0.90, 2))  # 90% CI
        puncta_short.loc[puncta_short[f"effective_radius{suffix}"].isna(), f"effective_radius{suffix}"
        ] = puncta_short.loc[puncta_short[f"effective_radius{suffix}"].isna(), "total_area"].apply(
            lambda x: np.sqrt(x/np.pi)
        )

        return puncta_short

    else:
        raise ValueError("Choose one: ['min_vol_ellipse', 'confidence_ellipse', 'circle']")

    return puncta_short


def two_pass_confidence_ellipse(puncta_short):
    """Required columns: center_x_puncta
    """

    # first pass ellipse
    puncta_short[f'center_puncta_first_pass'] = puncta_short[
        ['center_x_puncta', 'center_y_puncta']
    ].apply(lambda x: x if isinstance(x, np.ndarray) else np.array(list(x)), axis=1)
    puncta_short = generate_ellipse(
        puncta_short,
        suffix='_first_pass'
    )

    # filter mahalanobis_outliers from first pass
    cols = ['parent_nuclei_object_number', 'puncta_object_number',
            'center_x_puncta', 'center_y_puncta', 'integrated_intensity']
    for col in cols:
        puncta_short[col] = puncta_short[col].apply(np.array)
        puncta_short[f'{col}_second_pass'] = puncta_short[
            [col, 'is_mahalanobis_outlier_first_pass']
        ].apply(lambda x: x[0][~x[1]], axis=1)

    # second pass ellipse
    puncta_short['center_puncta_second_pass'] = puncta_short[
        ['center_x_puncta_second_pass', 'center_y_puncta_second_pass']
    ].apply(lambda x: x if isinstance(x, np.ndarray) else np.array(list(x)), axis=1)
    puncta_short = generate_ellipse(
        puncta_short,
        aweights='integrated_intensity_second_pass',
        n_std=2,
        suffix='_second_pass'
    )

    return puncta_short


def compute_diptest(
        ellipses,
        eccentricity_col='eccentricity_second_pass',
        major_axis_length_col='major_axis_length_second_pass',
        minor_axis_length_col='minor_axis_length_second_pass',
        mahalanobis_coordinates_col='mahalanobis_coordinates_second_pass',
    ):
    
    # get relevant x coordinate
    ellipses['diptest_mahalanobis_x'] = ellipses[mahalanobis_coordinates_col].apply(lambda x: x[0])
    
    # comute diptest
    ellipses['dip'] = ellipses['diptest_mahalanobis_x'].apply(
        lambda x: diptest.diptest(np.array(x)) if len(x) > 3 else (np.nan, np.nan)
    )
    ellipses['diptest_dip'] = ellipses['dip'].apply(lambda x: x[0])
    ellipses['diptest_pval'] = ellipses['dip'].apply(lambda x: x[1])
    ellipses.drop(columns=['dip'], inplace=True)
    
    # compute filter
    ellipses['puncta_doublet'] = (
        (ellipses[eccentricity_col] > np.sqrt(1-1/2**2)) &  # major_axis_length at least 2x the minor_axis_length
        (ellipses[major_axis_length_col] > 54) &   # the min minor_axis_length
        (ellipses['diptest_dip'] < 0.1) &  # note: the dip statistic is at least 0.5/n, where n is the number of items
        (ellipses['diptest_pval'] < 0.25)  # less than 25% confident that the distribution is univariate
    )
    
    return ellipses


def cluster_doublets(doublets):
    """Currently, as we do not expect to run MEFs through the pipeline,
    the input should be the minority of the data.
    
    Uses KMeans from sklearn
    See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """

    # compute centroids for initializing k means
    for i, side in enumerate(['left', 'right']):
        if side=='left':
            func = lambda x: [(x[0]-(x[2]*np.sin(x[3]/180*np.pi)/4)), (x[1]-(x[2]*np.cos(x[3]/180*np.pi)/4))]
        else:
            func = lambda x: [(x[0]+(x[2]*np.sin(x[3]/180*np.pi)/4)), (x[1]+(x[2]*np.cos(x[3]/180*np.pi)/4))]
        params = ["center_x_second_pass", "center_y_second_pass",
                  "major_axis_length_second_pass", "orientation_second_pass"]
        doublets[f'{side}_centroid'] = doublets[params].apply(func, axis=1)

    
    # Peform the K-Means Clustering
    doublets.loc[:, ['cluster_label', 'kmeans_centroids']] = np.array(
        doublets[['center_puncta_first_pass', 'left_centroid', 'right_centroid']].apply(
        lambda x: two_cluster_kmeans(x['center_puncta_first_pass'], x['left_centroid'], x['right_centroid']),
        axis=1
    ).to_list(), dtype=object)

    # construct final output array
    list_cols = ['parent_nuclei_object_number', 'puncta_object_number',
                 'center_x_puncta', 'center_y_puncta', 'integrated_intensity', 'area']
    singlets = doublets[
        ['image_number', 'nuclei_object_number']  # index cols
        + list_cols
        + ['kmeans_centroids', 'cluster_label']  # kmeans outputs
    ].copy()
    singlets['cluster_id'] = [[0, 1] for i in range(len(doublets))]
    singlets = singlets.explode('cluster_id')  # duplicate rows

    # grab only the items for the row
    singlets['kmeans_centroids'] = singlets[['kmeans_centroids', 'cluster_id']].apply(lambda x: x[0][x[1]], axis=1)
    for metric in ['parent_nuclei_object_number', 'puncta_object_number', 'integrated_intensity', 'center_x_puncta', 'center_y_puncta']:
        singlets[metric] = singlets[[metric, 'cluster_label', 'cluster_id']].apply(
            lambda x: x[metric][x['cluster_label']==x['cluster_id']], axis=1)
    singlets.drop(columns=['cluster_label'], inplace=True)
    
    return singlets


def plot_nuclei_ellipses_puncta(nuclei, ellipses, puncta, title=None, is_circle=False):
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

    if is_circle:

        circles_data = ellipses[
            ["nuclei_object_number", "center_x_mean", "center_y_mean", "effective_radius_circle"]
        ]
        plot = plot_circle_using_bokeh(
            circles_data,
            circles_data,
            x='center_x_mean',
            y='center_y_mean',
            size="effective_radius_circle",
            text="nuclei_object_number",
            text_color='orange',
            fill_color='#097969',  # green
            line_alpha=0,
            plot=plot
        )

    else:
        # confidence_ellipse
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
