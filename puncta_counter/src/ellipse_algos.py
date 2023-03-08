import numpy as np
import matplotlib.pyplot as plt


# Functions
# # confidence_ellipse
# # min_vol_ellipse


def confidence_ellipse(P, aweights=None, n_std=2.5, **kwargs):
    """Source code adapted from: https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    Note that aweights are required to prevent this algorithm from being sensitive to outliers
    This is only meant to be used for 2D
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
    center_x = sum(x*aweights)/sum(aweights) if aweights else np.mean(x)
    center_y = sum(y*aweights)/sum(aweights) if aweights else np.mean(y)
    
    # PCA
    eig_vals, eig_vecs = np.linalg.eig(cov)
    if eig_vals[0] >= eig_vals[1]:
        major_axis_length = (2 * major_radius) * np.sqrt(cov[0, 0]) * n_std
        minor_axis_length = (2 * minor_radius) * np.sqrt(cov[1, 1]) * n_std
        orientation = np.arccos(eig_vecs[1, 0])/np.pi*180
    else:
        major_axis_length = (2 * major_radius) * np.sqrt(cov[1, 1]) * n_std
        minor_axis_length = (2 * minor_radius) * np.sqrt(cov[0, 0]) * n_std
        orientation = np.arcsin(eig_vecs[1, 0])/np.pi*180

    return center_x, center_y, major_axis_length, minor_axis_length, orientation


def min_vol_ellipse(P, tolerance=0.01, **kwargs):
    # See: https://www.mathworks.com/matlabcentral/fileexchange/9542-minimum-volume-enclosing-ellipsoid
    # Original author: Nima Moshtagh (nima@seas.upenn.edu)
    # Translated to Python by Harrison Wang


    # data points
    # -----------------------------------
    
    d, N = P.shape
    if N <= d:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Q = np.zeros((d+1,N))
    # Q(1:d,:) = P(1:d,1:N)
    # Q(d+1,:) = np.ones(1,N)
    Q = np.vstack([P, np.ones((1, N))])

    # initializations
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

    u, s, vh = np.linalg.svd(A)
    # return u, s, vh  # troubleshooting
    
    try:
        items = np.sort(np.sqrt(1/s))
        minor_axis_length, major_axis_length = items[0], items[1]
    except:
        print(items)
        minor_axis_length, major_axis_length = np.nan, np.nan
    
    orientation = -np.arcsin(u[0][0])*180/np.pi
    
    return center_x, center_y, minor_axis_length, major_axis_length, orientation

    # bounding box
    inv_A = np.linalg.inv(A)  # diagonals are the important terms

    min_x = center_x - np.sqrt(inv_A[0][0])
    max_x = center_x + np.sqrt(inv_A[0][0])
    min_y = center_y - np.sqrt(inv_A[1][1])
    max_y = center_y + np.sqrt(inv_A[1][1])
    
    return center_x, center_y, min_x, max_x, min_y, max_y
