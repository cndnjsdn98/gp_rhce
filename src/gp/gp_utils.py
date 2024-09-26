import numpy as np


def distance_maximizing_points_1d(points, n_train_points, dense_gp=None):
    """
    Heuristic function for sampling training points in 1D (one input feature and one output prediction dimensions)
    :param points: dataset points for the current cluster. Array of shape Nx1
    :param n_train_points: Integer. number of training points to sample.
    :param dense_gp: A GP object to sample the points from, or None of the points will be taken directly from the data.
    :return:
    """
    closest_points = np.zeros(n_train_points, dtype=int if dense_gp is None else float)

    if dense_gp is not None:
        n_train_points -= 1

    # Fit histogram in data with as many bins as the number of training points
    a, b = np.histogram(points, bins=n_train_points)
    hist_indices = np.digitize(points, b) - 1

    # Pick as training value the median or mean value of each bin
    for i in range(n_train_points):
        bin_values = points[np.where(hist_indices == i)]
        if len(bin_values) < 1:
            closest_points[i] = np.random.choice(np.arange(len(points)), 1)
            continue
        if divmod(len(bin_values), 2)[1] == 0:
            bin_values = bin_values[:-1]

        if dense_gp is None:
            # If no dense GP, sample median points in each bin from training set
            bin_median = np.median(bin_values)
            median_point_id = np.where(points == bin_median)[0]
            if len(median_point_id) > 1:
                closest_points[i] = median_point_id[0]
            else:
                closest_points[i] = median_point_id
        else:
            # If with GP, sample mean points in each bin from GP
            bin_mean = np.min(bin_values)
            closest_points[i] = bin_mean

    if dense_gp is not None:
        # Add dimension axis 0
        closest_points[-1] = np.max(points)
        closest_points = closest_points[np.newaxis, :]

    return closest_points