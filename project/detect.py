"""Glue code for running the detector. Analogous to ``detect_fast.m`` in
Cherian et al.'s released code."""

from collections import namedtuple

from cnn import evaluate_cnn

Detection = namedtuple('Detection', ['joint_coords', 'score'])


def detect(image, threshold):
    """Run a supplied model on the given image and return a list of detections
    with scores exceeding the given threshold.

    :param image: A ``h*w*c`` array representing a single input image.
    :param threshold: Score threshold for detection.
    :returns: List of ``Detection`` instances."""
    # Run CNN
    feature_pyramid, unaries, idprs = evaluate_cnn(image)
    rv = []
    return rv
