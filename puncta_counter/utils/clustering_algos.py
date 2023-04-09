import numpy as np
from sklearn.cluster import KMeans

# Functions
# # find_nearest_point
# # two_cluster_kmeans


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
            nearest_point = (x, y)
            d = d_current
        
    return nearest_point


def two_cluster_kmeans(centers, left=[0, 0], right=[1, 1], random_state=37):
    """Wrapper around kmeans so it can return both labels_ and cluster_centers_ in one operation
    See: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    """

    if centers.shape[0]==2:
        # need a list of coordinates
        # ie. [[0, 1], [2, 3], [4, 5]], [[0, 2, 4], [1, 3, 5]]
        centers = np.transpose(centers)
    
    kmeans = KMeans(
        n_clusters=2, n_init=1, random_state=37,
        init=np.array(
            [left, right]
        )).fit(centers)
    
    return kmeans.labels_, kmeans.cluster_centers_
