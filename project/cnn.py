"""Code for training and using relevant CNNs. Uses pycaffe underneath."""

import caffe as cf
import lmdb

from util import unique_key, sample_patch


def make_patches(dataset, labels, destination):
    """Takes a dataset (including images and joint labels) and a set of learned
    cluster centroids for each limb. The images are then spliced according to
    the joint data and labelled using the cluster centroids, before being
    written to an LMDB file.

    :param dataset: ``Dataset`` instance to train on.
    :param labels: ``TrainingLabels`` instance giving a label to each
    on each limb.
    :param destination: path to LMDB file."""
    with lmdb.open(destination) as env:
        for sample_id in xrange(len(dataset.num_samples)):
            data = sample_to_data(dataset, sample_id)
            with env.begin(write=True, buffers=True) as txn:
                for datum in data:
                    datum_string = datum.SerializeToString()
                    datum_name = '{}_{:08}'.format(unique_key(), sample_id)
                    txn.put(datum_name, datum_string)


def sample_to_data(dataset, labels, sample_id):
    """Take a dataset, a set ot training labels, and a training ID from within
    the dataset and produce a list of ``Datum`` instances which can be shoved
    into LMDB."""
    locs = dataset.joints.locations[sample_id]
    img = dataset.load_image(sample_id)
    rv = []

    for part_idx in xrange(len(locs)):
        part_x, part_y, visible = locs[part_idx]
        if not visible:
            continue

        scale = dataset.scales[part_idx]
        cropped = sample_patch(img, part_x, part_y, scale, scale, mode='edge')
        label = labels.id_for(sample_id, part_idx)
        datum = cf.io.array_to_datum(cropped, label=label)
        rv.append(datum)

    return rv


def train_on_patches():
    """Uses patches stored in an LMDB file by ``make_patches()`` to train a
    CNN."""
    # TODO
