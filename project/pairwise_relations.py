"""Module for learning and predicting pairwise relation types through K-means
clustering."""

import numpy as np

from sklearn.cluster import KMeans


def from_dataset(joints, k, scales, template_size):
    """Takes the joints from a set and learns a set of k different pairwise
    relation types for each joint.

    :param joints: a ``Joints`` instance.
    :param k: number of "types" to learn for each possible joint.
    :param scales: n-dimensional array giving a scale for each sample in the
    dataset.
    :param template_size: floating point number representing the length of one
    side on a (square) part template.
    :return: j * k * d ndarray, where j is the number of joints, k is given,
    and d is the dimension of each cluster center (in this case, 2)."""
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

        # This is the "normalisation" applied by Chen & Yuille.
        scaled_X = template_size * X / (2 * scales + 1).reshape((-1, 1))

        # XXX: Chen & Yuille apply mean-centering before doing K-means. I'm not
        # sure why they do that, so I haven't done it here. I should figure out
        # whether it's necessary.

        # Now we fit a model!
        model = KMeans(n_clusters=k)
        model.fit(scaled_X)

        # Extract the cluster centers
        centers = model.cluster_centers_
        assert centers.ndim == 2
        assert centers.shape == (k, 2)
        cluster_centers.append(centers)

    return np.array(cluster_centers)


def _make_id_dict(adjacency, k):
    """Makes a dictionary of IDs for each combination of part and neighbour
    mixtures for that part.

    :param adjacency: p*p adjacency matrix for the parts.
    :param k: number of clusters per part->part relationship.
    :return: list of matrices in which the p-th (corresponding to row p of the
    adjacency matrix) entry is an k*k*k*...-dimensional (k^n_p, where n_p is
    the number of neighbours of p) ndarray assigning unique IDs to each (part 1
    mixture, part 2 mixture, ...) combination."""
    num_parts = len(adjacency)
    assert adjacency.shape == (num_parts, num_parts)
    rv = []
    current_idx = 0

    for adj in adjacency:
        num_neighbours = np.sum(adj)
        ids = np.arange(k ** num_neighbours).reshape((k,) * num_neighbours)
        rv.append(ids)
        current_idx += ids.size

    return rv


class TrainingLabels(object):
    """Class for producing training labels from a dataset and some cluster
    centroids."""
    def __init__(self, dataset, centroids):
        """Take a dataset and some clusters and produce global labels for
        them."""
        self.scales = dataset.scales
        self.joints = dataset.joints
        self.centroids = centroids
        self.ids = _make_id_dict(self.joints.adjacent, centroids.shape[1])
        self.template_size = dataset.template_size

    def id_for(self, sample_idx, part_idx):
        """Produces an ID for a specific sample and part."""
        adj = self.joints.adjacent[part_idx]
        neighbours = np.flatnonzero(adj)
        locs = self.joints.locations[sample_idx]
        scale = self.scales[sample_idx]
        cluster_ids = np.zeros_like(neighbours)

        for cluster_ids_idx, neighbour in enumerate(neighbours):
            assert neighbour != part_idx
            pair = (part_idx, neighbour)
            limb_id = self.joints.pair_indices[pair]
            clusters = self.centroids[limb_id]

            # XXX: This relative position calculation is potentially VERY
            # buggy. I should figure out a better way of doing this.
            first_loc_id, second_loc_id = self.joints.pairs[limb_id]
            relative_pos = locs[second_loc_id, :2] - locs[first_loc_id, :2]
            normed_rp = self.template_size * relative_pos / (2 * scale + 1)
            assert normed_rp.shape == (2,)

            diffs = clusters - normed_rp.reshape((1, 2))
            assert diffs.shape == clusters.shape
            mags = np.sum(diffs, axis=1) ** 2
            assert mags.shape == (clusters.shape[0],)
            cluster_ids[cluster_ids_idx] = np.argmin(mags)

        cluster_ids_key = tuple(cluster_ids)

        return self.ids[part_idx][cluster_ids_key]
