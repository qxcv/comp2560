"""Module for learning and predicting pairwise relation types through K-means
clustering."""

from sklearn.cluster import KMeans


class PairwiseRelations(object):
    # TODO
    pass


def from_dataset(joints, k):
    """Takes the joints from a set and learns a set of k different pairwise
    relation types for each joint.

    :param joints: a ``Joints`` instance.
    :param k: number of "types" to learn for each possible joint.
    :return: ``PairwiseRelations`` instance containing all of the relevant
    clusters."""
    # TODO
    pass
