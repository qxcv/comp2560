"""Code for training and using relevant CNNs. Uses pycaffe underneath."""

import caffe as cf


def make_patches(dataset, cluster_centroids):
    """Takes a dataset (including images and joint labels) and a set of learned
    cluster centroids for each limb. The images are then spliced according to
    the joint data and labelled using the cluster centroids, before being
    written to an LMDB file."""
    pass


def train_on_patches():
    """Uses patches stored in an LMDB file by ``make_patches()`` to train a
    CNN."""
