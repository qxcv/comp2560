#!/usr/bin/env python2

"""Trains a complete model from a supplied dataset."""

# Standard library imports
from __future__ import print_function

from argparse import ArgumentParser
import logging
from os import path
import pickle
import sys

# External imports
import numpy as np

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

# Local imports
from config import from_files
from cnn import (customize_solver, customize_train_net, make_patches,
                 train_dcnn_patches)
import datasets
from pairwise_relations import from_dataset, TrainingLabels
from util import create_dirs

DESCRIPTION = """Trains a model from beginning to end, including dataset \
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
parser.set_defaults(check_cache=True)
parser.add_argument(
    '--no-cache', dest='check_cache', action='store_false',
    help="Prevent trainer from using cached data."
)


def load_cache_and_dataset(loader_name, cache_prefix, dataset_path,
                           check_cache=True):
    """"Finds a loader by name, builds a cache prefix from the loader name,
    loads a dataset and returns the set split into train, test and validate
    subsets.

    :param loader_name: Name of the dataset loader to use (e.g. LSP, FLIC).
    :param cache_prefix: Prefix of the cache directory.
    :param dataset_path: Path to dataset.
    :param check_cache: Should the cache be checked for picked datasets?
    :returns: Tuple of ``(fresh_cache, train_set, validate_set, test_set,
        cache_path)``, where ``fresh_cache`` is a boolean indicating whether
        the cache is still appropriate to use in later stages of the training
        pipeline (will be false iff the cache was written to during loading),
        ``{train, validate, test}_set`` are the relevant chunks of the data
        set, and ``cache_path`` is a complete path to the cache directory
        (including the appropriate prefix)."""
    # Get loader
    if loader_name not in datasets.ALLOWED_LOADERS:
        print("'{}' is not a valid loader. Allowed loaders: {}".format(
            loader_name, ', '.join(datasets.ALLOWED_LOADERS)
        ), file=sys.stderr)
        sys.exit(1)
    loader = getattr(datasets, loader_name)

    # Caching
    cache_dir = path.join(args.cache, loader_name)
    logging.info("Checking cache directory '{}'".format(cache_dir))
    create_dirs(cache_dir)

    # Now actually load the dataset, if possible
    pickle_path = path.join(cache_dir, 'dataset_meta.pickle')

    if check_cache and path.exists(pickle_path):
        logging.info("Loading pickled dataset")
        with open(pickle_path) as fp:
            train_set, validate_set, test_set = pickle.load(fp)
    else:
        if check_cache:
            msg = "Pickled dataset not found"
        else:
            msg = "Ignoring cached dataset, if any"
        logging.info(msg + '; loading full dataset')

        whole_dataset = loader(dataset_path)
        # Training set will contain 1/2 of the set
        train_set, others = whole_dataset.split(2)
        # Validation and test sets will each contain 1/4 of the set
        validate_set, test_set = others.split(2)

        logging.info("Pickling dataset for future use")
        with open(pickle_path, 'w') as fp:
            pickle.dump((train_set, validate_set, test_set), fp)
        check_cache = False

    return check_cache, train_set, validate_set, test_set, cache_dir


def do_pairwise_clustering(train_set, validate_set, test_set, cache_dir,
                           check_cache=True):
    centroids_path = path.join(cache_dir, 'centroids.npy')
    if check_cache and path.exists(centroids_path):
        logging.info("Loading centroids from cache")
        centroids = np.load(centroids_path)
    else:
        if check_cache:
            msg = "Saved centroids not found"
        else:
            msg = "Ignoring saved centroids, if any"
        logging.info(msg + '; deriving centroids')

        centroids = from_dataset(
            train_set.joints, K, train_set.scales, train_set.template_size
        )

        logging.info('Caching centroids')
        np.save(centroids_path, centroids)
        check_cache = False

    labels_path = path.join(cache_dir, 'labels.pickle')
    if check_cache and path.exists(labels_path):
        with open(labels_path) as fp:
            train_labels, validate_labels, test_labels = pickle.load(fp)
    else:
        if check_cache:
            msg = "Cached labels not found"
        else:
            msg = "Ignoring cached labels, if any"
        logging.info(msg + '; calculating labels')
        logging.info("Labelling training set")
        train_labels = TrainingLabels(train_set, centroids)
        logging.info("Labelling validation set")
        validate_labels = TrainingLabels(validate_set, centroids)
        logging.info("Labelling test set")
        test_labels = TrainingLabels(test_set, centroids)

        logging.info("Pickling labels")
        with open(labels_path, 'w') as fp:
            pickle.dump((train_labels, validate_labels, test_labels), fp)
        check_cache = False

    return check_cache, centroids, train_labels, validate_labels, test_labels


def do_make_patches(name, dataset, labels, cache_dir, check_cache=True):
    db_path = path.join(cache_dir, '{}_patches.lmdb'.format(name))

    if check_cache and path.exists(db_path):
        logging.info("Using cached {} patches".format(name))
    else:
        logging.info("Generating {} patches for CNN".format(name))
        make_patches(
            dataset, labels, db_path
        )
        check_cache = False

    return check_cache, db_path


if __name__ == '__main__':
    # Grab the configuration file
    args = parser.parse_args()
    logger = logging.getLogger()
    logger.setLevel(LOG_LEVELS[args.loglevel])
    logging.info('Parsing configuration')
    cfg = from_files(args.conf)
    check_cache = args.check_cache

    # Load the dataset
    loader_name = cfg.get('dataset', 'loader')
    dataset_path = cfg.get('dataset', 'path')
    check_cache, train_set, validate_set, test_set, cache_dir = \
        load_cache_and_dataset(
            loader_name, args.cache, dataset_path, check_cache
        )

    # Derive pairwise relations
    K = cfg.getint('graphical_model', 'centroids_per_limb')
    check_cache, centroids, train_labels, validate_labels, test_labels = \
        do_pairwise_clustering(
            train_set, validate_set, test_set, cache_dir, check_cache
        )

    # Make training and validation patches for the CNN
    check_cache, train_patch_dir = do_make_patches(
        'train', train_set, train_labels, cache_dir, check_cache
    )
    check_cache, validate_patch_dir = do_make_patches(
        'validate', validate_set, validate_labels, cache_dir, check_cache
    )

    # TODO: Mean pixel. This isn't needed for the the Chen & Yuille net (which
    # uses a fixed mean of #808080).
    # logging.info("Computing mean pixel value for training set")
    # mean_pixel_path = path.join(cache_dir, "mean_pixel")
    # compute_mean_pixel(train_patch_dir, mean_pixel_path)

    # Now we load & update the solver and net definitions
    model_source_path = cfg.get('cnn', 'train_net')
    model_dest_path = path.join(cache_dir, 'train_val.prototxt')
    customize_train_net(
        model_source_path, model_dest_path, train_patch_dir, validate_patch_dir
    )

    solver_source_path = cfg.get('cnn', 'solver')
    solver_dest_path = path.join(cache_dir, 'solver.prototxt')
    customize_solver(
        solver_source_path, solver_dest_path, cache_dir, model_dest_path
    )

    # caffe train!
    logging.info("Training Caffe model")
    gpu_id = None if args.gpu == -1 else args.gpu
    train_dcnn_patches(model_dest_path, solver_dest_path, gpu=gpu_id)
