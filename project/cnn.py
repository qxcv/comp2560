"""Code for training and using relevant CNNs. Uses pycaffe underneath."""

import logging
from multiprocessing import cpu_count, Pool

import caffe as cf
import lmdb
from scipy.misc import imresize

from util import unique_key, sample_patch


def process_patch(args):
    destination, dataset, labels, sample_id = args
    logging.info('Generating patches for sample {}/{}'.format(
        sample_id, dataset.num_samples
    ))
    data = sample_to_data(dataset, labels, sample_id)
    # map_size is 1TiB. Hopefully this won't cause problems ;)
    with lmdb.open(destination, create=True, map_size=1 << 40) as env:
        with env.begin(write=True, buffers=True) as txn:
            for datum in data:
                datum_string = datum.SerializeToString()
                datum_name = '{}_{:08}'.format(unique_key(), sample_id)
                txn.put(datum_name, datum_string)


def make_patches(dataset, labels, destination):
    """Takes a dataset (including images and joint labels) and a set of learned
    cluster centroids for each limb. The images are then spliced according to
    the joint data and labelled using the cluster centroids, before being
    written to an LMDB file.

    :param dataset: ``Dataset`` instance to train on.
    :param labels: ``TrainingLabels`` instance giving a label to each
    on each limb.
    :param destination: path to LMDB file."""
    pool = Pool(cpu_count() + 1)
    gen = ((destination, dataset, labels, n)
           for n in xrange(dataset.num_samples))
    pool.map(process_patch, gen)


def sample_to_data(dataset, labels, sample_id):
    """Take a dataset, a set ot training labels, and a training ID from within
    the dataset and produce a list of ``Datum`` instances which can be shoved
    into LMDB."""
    locs = dataset.joints.locations[sample_id]
    img = dataset.load_image(sample_id)
    side_length = dataset.template_size * dataset.STEP
    assert isinstance(side_length, int)
    rv = []

    for part_idx in xrange(len(locs)):
        part_x, part_y, visible = locs[part_idx]
        if not visible:
            continue

        scale = dataset.scales[part_idx]
        cropped = sample_patch(img, part_x, part_y, scale, scale, mode='edge')
        scaled = imresize(cropped, (side_length,) * 2)
        assert scaled.shape == (side_length,) * 2 + cropped.shape[2:]
        assert scaled.dtype.name == 'uint8'
        label = labels.id_for(sample_id, part_idx)
        datum = cf.io.array_to_datum(scaled, label=label)
        rv.append(datum)

    return rv


def compute_image_mean(lmdb_path, destination):
    """Uses the ``compute_image_mean`` Caffe tool to compute the mean pixel
    value of a training database."""
    raise NotImplemented()


def train_dcnn_patches(train_db, val_db, mu_pix_db, solver, gpu=None):
    """Uses patches stored in an LMDB file by ``make_patches()`` to train a
    CNN."""
    raise NotImplemented()
