"""Module for learning and predicting pairwise relation types through K-means
clustering."""

import numpy as np

from sklearn.cluster import KMeans


def from_dataset(joints, k):
    """Takes the joints from a set and learns a set of k different pairwise
    relation types for each joint.

    :param joints: a ``Joints`` instance.
    :param k: number of "types" to learn for each possible joint.
    :return: ``PairwiseRelations`` j * k * d ndarray, where j is the number of
    joints, k is given, and d is the dimension of each cluster center (in this
    case, 2)."""
    # Cluster centers for each joint
    cluster_centers = []

    for first_idx, second_idx in joints.pairs:
        # Take x and y locations (:2) of each joint (first_idx, second_idx)
        # over every image in the dataset (:) and calculate the difference
        first_locations = joints.locations[:, first_idx, :2]
        second_locations = joints.locations[:, second_idx, :2]
        # X will be a two-axis ndarray with the first axis corresponding to
        # samples and the second axis corresponding to (2D) relative positions
        # within an image.
        X = second_locations - first_locations
        assert X.ndim == 2
        assert X.shape == (len(joints.locations), 2)

        # Now we fit a model!
        model = KMeans(n_clusters=k)
        model.fit(X)

        # Extract the cluster centers
        centers = model.cluster_centers_
        assert centers.ndim == 2
        assert centers.shape == (k, 2)
        cluster_centers.append(centers)

    return np.array(cluster_centers)
