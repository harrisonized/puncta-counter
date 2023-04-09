import numpy as np


# Functions
# # confidence_ellipse
# # min_vol_ellipse
# # scale_invariant_ellipse_transform
# # center_points
# # rotate_points
# # rescale_xy
# # mahalanobis_transform
# # compute_euclidean_distance_from_origin


def confidence_ellipse(P, aweights=None, n_std=1, **kwargs):
    """Source code adapted from: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    Note that aweights are required to prevent this algorithm from being sensitive to outliers
    This is only meant to be used for 2D
    By default, the n_std should be 1 to generate the correct mahal distance
    """
    x, y = P
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    # dimension check
    # for a 2d image, d=2 and N is the number of points
    if x.size <= 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    cov = np.cov(x, y, aweights=aweights)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    # Use a special case to obtain the eigenvalues of this 2d dataset.
    minor_radius, major_radius = sorted(
        [np.sqrt(max(0, 1 - pearson)), np.sqrt(max(0, 1 + pearson))]
    )
    
    # calculate weighted center
    center_x = sum(x*aweights)/sum(aweights) if aweights is not None else np.mean(x)
    center_y = sum(y*aweights)/sum(aweights) if aweights is not None else np.mean(y)
    
    # 2D PCA Only
    # Note from Harrison: I spent a lot of time verifying that the signs and angles here are correct.
    # Note that this is NOT the equivalent of SVD,
    # because the sign and order of the eigenvectors and eigenvalues can change!
    # See: https://stats.stackexchange.com/questions/197034/different-order-and-signs-of-eigenvectors-when-doing-pca-via-eig-or-svd-func
    eig_vals, eig_vecs = np.linalg.eig(cov)
    if eig_vals[0] >= eig_vals[1]:
        major_axis_length = (2 * major_radius) * np.sqrt(cov[0, 0]) * n_std
        minor_axis_length = (2 * minor_radius) * np.sqrt(cov[1, 1]) * n_std
        # np.arccos outputs range of [0, pi], so orientation range is in [-180, 0]
        orientation = -np.arccos(eig_vecs[1, 0])/np.pi*180
        # correct the orientation range to be [-90, 90]
        if orientation < -90:
            orientation = orientation + 180
    else:
        major_axis_length = (2 * major_radius) * np.sqrt(cov[1, 1]) * n_std
        minor_axis_length = (2 * minor_radius) * np.sqrt(cov[0, 0]) * n_std
        # np.arcsin outputs range of [-pi/2, pi/2], so orientation range is in [-90, 90]
        orientation = -np.arcsin(eig_vecs[1, 0])/np.pi*180

    return center_x, center_y, major_axis_length, minor_axis_length, orientation


def min_vol_ellipse(P, tolerance=0.05, **kwargs):
    """See: https://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid
    Original author: Nima Moshtagh (nima@seas.upenn.edu)
    Translated to Python by Harrison Wang
    
    Note about the orientation: This is the  angle (in degrees ranging from -90 to 90 degrees)
    between the x-axis and the major axis of the ellipse. Since I had to pick a convention, I chose
    to use the standard x, y coordinate system, where x is increasing left to right and y is
    increasing from bottom to top. However, in standard images, y=0 is at the top left corner and
    increases down. This should be taken care of by the scale_invariant_ellipse_transform.
    """

    # Data Points
    # -----------------------------------
    
    d, N = P.shape
    if N <= d:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Q = np.zeros((d+1,N))
    # Q(1:d,:) = P(1:d,1:N)
    # Q(d+1,:) = np.ones(1,N)
    Q = np.vstack([P, np.ones((1, N))])


    # Initializations
    # -----------------------------------
    count = 1
    err = 1
    u = (1/N) * np.ones((N, 1))  # 1st iteration
    

    # Khachiyan Algorithm
    # -----------------------------------
    while err > tolerance:
        try:
            X = np.dot(np.dot(Q, u* np.identity(N)), np.transpose(Q))
            M = np.diag(np.dot(np.dot(np.transpose(Q), np.linalg.inv(X)), Q))  # M the np.diagonal vector of an NxN matrix

            j = np.argmax(M)
            maximum = max(M)

            step_size = (maximum - d -1)/((d+1)*(maximum-1))
            new_u = (1 - step_size)*u 
            new_u[j] = new_u[j] + step_size

            count = count + 1
            err = np.linalg.norm(new_u - u)
            u = new_u
        except:
            print(P)
            break


    ################### Computing the Ellipse parameters######################
    # Finds the ellipse equation in the 'center form': 
    # (x-c)' * A * (x-c) = 1
    # It computes a dxd matrix 'A' and a d dimensional vector 'c' as the center
    # of the ellipse. 
    U = u * np.identity(N)
    # return P, u
    
    # the A matrix for the ellipse
    # A = (1/d) * inv(P * U * P' - (P * u)*(P*u)' );
    # --------------------------------------------
    A = (1/d) * np.linalg.inv(
        np.dot(np.dot(P, U), np.transpose(P)) - np.dot(np.dot(P, u), np.transpose(np.dot(P, u)))
    )
    
    # center of the ellipse 
    # --------------------------------------------
    c = np.dot(P, u)
    
    # original return value
    # return A, c

    center_x = c[0][0]
    center_y = c[1][0]

    eig_vals, eig_vecs = np.linalg.eig(A)
    if eig_vals[0] <= eig_vals[1]:
        major_axis_length = 2/np.sqrt(eig_vals[1])
        minor_axis_length = 2/np.sqrt(eig_vals[0])
        orientation = np.arcsin(eig_vecs[0, 1])/np.pi*180
    else:
        major_axis_length = 2/np.sqrt(eig_vals[0])
        minor_axis_length = 2/np.sqrt(eig_vals[1])
        orientation = np.arccos(eig_vecs[0, 1])/np.pi*180
        if orientation > 90:
            orientation = orientation - 180

    return center_x, center_y, major_axis_length, minor_axis_length, orientation



def scale_invariant_ellipse_transform(major_axis_length, minor_axis_length, angle, height_to_width_ratio):
    """In Bokeh, the angle uses screen pixel ratio rather than the data range ratio to draw ellipses
    As a result, ellipses are only drawn correctly if your axes are perfectly square
    See: https://stackoverflow.com/questions/61877110/how-to-plot-an-ellipse-with-bokeh-and-a-correct-orientation-angle
    
    This is a geometric workaround that's only an approximate solution.
    
    Consider a triangle that is squished in half horizontally,
    such that 2 times the number of pixels fit in the same x range, ie. (x, y) -> (x', y') = (2x, y),
    where x' and y' denote the new coordinates.
    
    For a triangle of length 10 that makes an angle of 30 degrees with the x axis,
    What does this look like in the new coordinate system?
    1. The angle will increase, because the x axis is squished.
        To find the new angle, observe that tan(theta)=1 in the original coordinate system,
        whereas tan(theta')=1/(1/2)=2, because the x axis has been squished by 2.
    2. The new major_axis_length L' is found by using the Euclidean distance in the new space
        L'/L = sqrt(x'**2 + y'**2) / sqrt(x**2 + y**2). Assigning x'=2x and x=y=y'=1, we get:
        L'/L = sqrt(2 + 1) / sqrt(2). In other words, in the new coordinate system, it's "stretched"
    3. Unfortunately, the minor_axis_length will be inaccurate regardless how how this goes,
        because once you squeeze or stretch a rotated object, the major axis length and minor axis length
        will no longer be orthogonal! But a good approximation is the opposite transform of
        the major_axis_length.
        
    Provide the angle in degrees and the height_to_width_ratio
    For example, if height=30 and width=60, then the height_to_width_ratio=1/2
    
    Copyright: Harrison Wang, 2023
    """
    new_angle = np.arctan(height_to_width_ratio*np.tan(angle/180*np.pi))*180/np.pi
    new_major_axis_length = major_axis_length * np.sqrt((height_to_width_ratio)**2 + 1) / np.sqrt(2)
    new_minor_axis_length = minor_axis_length * np.sqrt(2) / np.sqrt((height_to_width_ratio)**2 + 1) 
    
    return new_major_axis_length, new_minor_axis_length, new_angle


def center_points(A):
    return np.array((A[0]-A[0].mean(), A[1]-A[1].mean()))


def rotate_points(A, orientation):
    """A is a list of points
    Provide the orientation in degrees
    To unrotate points, rotate by negative orientation
    Make sure points are centered before using this
    """
    # Rotation matrix
    R = np.array(
        [[np.cos(orientation/180*np.pi), -np.sin(orientation/180*np.pi)],
         [np.sin(orientation/180*np.pi), np.cos(orientation/180*np.pi)]]
    )
    
    return np.dot(R, A)


def rescale_xy(A, scale_x=1, scale_y=1):
    """Make sure points have been unrotated before using this
    """
    return np.array((A[0]/scale_x, A[1]/scale_y))


def mahalanobis_transform(P, major_axis_length, minor_axis_length, orientation):
    """Transform points such that they are centered about 0
    and the distance from the origin is the Mahalanobis distance
    """
    centered = center_points(P)
    rotated = rotate_points(centered, -orientation)
    scaled = rescale_xy(rotated, major_axis_length, minor_axis_length)
    
    return scaled


def compute_euclidean_distance_from_origin(P):
    """Euclidean distances in Mahalanobis space is the number of standard deviations from the mean
    """
    return np.sqrt(P[0]**2+P[1]**2)
