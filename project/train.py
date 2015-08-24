#!/usr/bin/env python2

"""Trains a complete model from a supplied dataset."""

# Standard library imports
from __future__ import print_function

from argparse import ArgumentParser
import logging
from os import path
import sys

# Now try to get Caffe on the PATH if we need to
try:
    import caffe                                                         # noqa
except ImportError:
    best_guess = path.expanduser('~/repos/caffe/distribute/python')
    success = False
    if path.isdir(best_guess):
        sys.path.append(best_guess)
        try:
            import caffe                                                 # noqa
            success = True
        except ImportError:
            success = False

    if not success:
        print("Couldn't find pycaffe!", file=sys.stderr)

from config import from_files
from cnn import compute_mean_pixel, make_patches, train_dcnn
import datasets
from pairwise_relations import from_dataset, TrainingLabels
from util import create_dirs

DESCRIPTION = """Trains a model from beginning to end, including datset \
loading, clustering of limbs, training of the part detection CNN and \
training of the graphical model."""
LOG_LEVELS = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}
LOG_LEVEL_OPTIONS = ', '.join(LOG_LEVELS.keys())

parser = ArgumentParser(description=DESCRIPTION)
parser.add_argument(
    '--conf', type=open, action='append', metavar='FILE', default=[],
    help="extra .cfg file to use; can be specified multiple times"
)
parser.add_argument(
    '--cache', type=str, default='cache/', metavar='DIRECTORY',
    help="directory to store cached training data in"
)
parser.add_argument(
    '--gpu', type=int, default=-1, metavar='ID',
    help="ID of training GPU (disabled by default; -1 to disable manually)"
)
parser.add_argument(
    '--loglevel', '-l', type=str, default='warning', choices=LOG_LEVELS.keys(),
    metavar='LEVEL', help="logging level for Python's logging module (one of "
    + ', '.join(LOG_LEVELS.keys()) + ")"
)

if __name__ == '__main__':
    # Grab the configuration file
    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVELS[args.loglevel])
    logging.info('Parsing configuration')
    cfg = from_files(args.conf)

    # Load the dataset
    loader_name = cfg.get('dataset', 'loader')
    if loader_name not in datasets.ALLOWED_LOADERS:
        print("'{}' is not a valid loader. Allowed loaders: {}".format(
            loader_name, ', '.join(datasets.ALLOWED_LOADERS)
        ), file=sys.stderr)
        sys.exit(1)
    loader = getattr(datasets, loader_name)
    whole_dataset = loader(cfg.get('dataset', 'path'))
    # Training set will contain 1/2 of the set
    train_set, others = whole_dataset.split(2)
    # Validation and test sets will each contain 1/4 of the set
    validate_set, test_set = others.split(2)

    # Caching
    cache_dir = path.join(args.cache, loader_name)
    logging.info("Checking cache directory '{}'".format(cache_dir))
    create_dirs(cache_dir)

    # Derive pairwise relations
    K = cfg.getint('graphical_model', 'centroids_per_limb')
    logging.info("Deriving limb relative position centroids")
    centroids = from_dataset(
        train_set.joints, K, train_set.scales, train_set.template_size
    )
    logging.info("Labelling training set")
    train_labels = TrainingLabels(train_set, centroids)
    logging.info("Labelling validation set")
    validate_labels = TrainingLabels(validate_set, centroids)
    logging.info("Labelling test set")
    test_labels = TrainingLabels(test_set, centroids)

    # Train the CNN
    logging.info("Generating training image patches for CNN")
    train_patch_dir = path.join(cache_dir, 'train_patches.lmdb')
    validate_patch_dir = path.join(cache_dir, 'validate_patches.lmdb')
    make_patches(
        train_set, train_labels, train_patch_dir
    )
    logging.info("Generating validation image patches for CNN")
    make_patches(
        validate_set, validate_labels, validate_patch_dir
    )
    logging.info("Computing mean pixel value for training set")
    mean_pixel_path = path.join(cache_dir, "mean_pixel")
    compute_mean_pixel(train_patch_dir, mean_pixel_path)
    logging.info("Training Caffe model")
    gpu_id = None if args.gpu == -1 else args.gpu
    train_dcnn(
        train_patch_dir, validate_patch_dir, mean_pixel_path,
        cfg.get('cnn', 'solver_path'), gpu_id=gpu_id
    )
