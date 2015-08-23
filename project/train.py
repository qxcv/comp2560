#!/usr/bin/env python2

"""Trains a complete model from a supplied dataset."""

from __future__ import print_function

from argparse import ArgumentParser
import logging
from os import path
from sys import stderr, exit

from config import from_files
from cnn import make_patches
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
        ), file=stderr)
        exit(1)
    loader = getattr(datasets, loader_name)
    dataset = loader(cfg.get('dataset', 'path'))

    # Caching
    cache_dir = path.join(args.cache, loader_name)
    logging.info("Checking cache directory '{}'".format(cache_dir))
    create_dirs(cache_dir)

    # Derive pairwise relations
    K = cfg.getint('graphical_model', 'centroids_per_limb')
    logging.info("Deriving limb relative position centroids")
    centroids = from_dataset(
        dataset.joints, K, dataset.scales, dataset.template_size
    )
    logging.info("Labelling dataset")
    labels = TrainingLabels(dataset, centroids)

    # Train the CNN
    logging.info("Generating image patches for CNN")
    make_patches(dataset, labels, path.join(cache_dir, 'train_patches.lmdb'))
