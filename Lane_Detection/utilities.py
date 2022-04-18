import numpy as np
import random
import math

import scipy
from scipy import signal


def RANSAC(point):
    """Fit parabola.

    Args:
        point: An array

    Returns: parameter of parabola
    """

    beta = []  # best parameter of Parabola

    # pre-set param
    tr = 4  # tolerance for squared residual
    epsilon = 0.99  # threshold of the percentage of the inliers
    iters = 10000000  # maximum number of iterations

    for i in range(iters):
        # random select 5 candidates
        sample_index = random.sample(range(point.shape[0]), 5)
        RandomPoint = np.array([point[i] for i in sample_index])

        # fit parabola
        beta = np.polyfit(RandomPoint[:, 0], RandomPoint[:, 1], deg=2)

        # calculate the numbers of inliers and remove outliers
        total_inlier = 0
        outlier_index = []
        for index in range(point.shape[0]):
            value_estimate = np.polyval(beta, point[index, 0])
            if abs(value_estimate - point[index, 1]) < tr:
                total_inlier = total_inlier + 1
            else:
                outlier_index.append(index)
        if total_inlier > epsilon * point.shape[0]:
            break
        point = np.delete(point, outlier_index, 0)

    return beta


def Edge_Detection(Sobel_Image, threshold,height, width, Vanishing_y0, Vanishing_x_Accumulator, Vanishing_y,
                   Orientation_Accumulator):
    GX = np.array([
        [4, 3, 2, 1, 0, -1, -2, -3, -4],
        [5, 4, 3, 2, 0, -2, -3, -4, -5],
        [6, 5, 4, 3, 0, -3, -4, -5, -6],
        [7, 6, 5, 4, 0, -4, -5, -6, -7],
        [8, 7, 6, 5, 0, -5, -6, -7, -8],
        [7, 6, 5, 4, 0, -4, -5, -6, -7],
        [6, 5, 4, 3, 0, -3, -4, -5, -6],
        [5, 4, 3, 2, 0, -2, -3, -4, -5],
        [4, 3, 2, 1, 0, -1, -2, -3, -4]
    ])

    # GX = np.array([
    #     [2, 1, 0, -1, -2],
    #     [3, 2, 0, -2, -3],
    #     [4, 3, 0, -3, -4],
    #     [3, 2, 0, -2, -3],
    #     [2, 1, 0, -1, -2]
    # ])
    # GX = np.array([
    #
    #     [2, 0, -2],
    #     [3, 0, -3],
    #     [2, 0, -2]
    #
    # ])



    GY = GX.T
    grad_x = signal.convolve2d(Sobel_Image, GX, mode="same")
    grad_y = signal.convolve2d(Sobel_Image, GY, mode="same")

    for y in np.arange(Vanishing_y0, height):
        for x in np.arange(width):
            # if grad_x[y, x] != 0 and grad_y[y, x] != 0:
            if abs(grad_x[y, x]) > threshold and abs(grad_y[y, x]) > threshold :

                tan_theta = grad_y[y, x] / grad_x[y, x]
                Orientation_Accumulator[y, x] = -grad_x[y, x]
                a = round(x + (y - Vanishing_y[y]) / tan_theta)
                if 0 <= a < width:
                    Vanishing_x_Accumulator[y - Vanishing_y0, a] += 1
    return grad_x, grad_y



