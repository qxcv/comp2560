"""Code for training and using relevant CNNs. Uses pycaffe underneath."""

import logging
from subprocess import call as pcall
from distutils.spawn import find_executable

import caffe as cf
from google.protobuf.text_format import Merge, MessageToString
import lmdb
from scipy.misc import imresize

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
    with lmdb.open(destination, create=True, map_size=1 << 40) as env:
        for sample_id in xrange(dataset.num_samples):
            logging.info('Generating patches for sample {}/{}'.format(
                sample_id + 1, dataset.num_samples
            ))
            data = sample_to_data(dataset, labels, sample_id)
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
    side_length = dataset.template_size * dataset.STEP
    assert isinstance(side_length, int)
    rv = []

    for part_idx in xrange(len(locs)):
        part_x, part_y, visible = locs[part_idx]
        if not visible:
            continue

        scale = dataset.scales[part_idx]
        cropped = sample_patch(
            img, int(part_x), int(part_y), int(scale), int(scale), mode='edge'
        )
        scaled = imresize(cropped, (side_length,) * 2)
        assert scaled.shape == (side_length,) * 2 + cropped.shape[2:]
        assert scaled.dtype.name == 'uint8'
        label = labels.id_for(sample_id, part_idx)
        datum = cf.io.array_to_datum(scaled, label=label)
        rv.append(datum)

    return rv


def read_prototxt(ptxt_path, message):
    """Takes a path to a ``.prototxt`` file and a protobuf message and merges
    the two."""
    with open(ptxt_path) as fp:
        ptxt_contents = fp.read()

    return Merge(ptxt_contents, message)


def save_prototxt(message, dest_path):
    """Save protobuf message as ``.prototxt``."""
    msg_str = MessageToString(message)

    with open(dest_path, 'w') as fp:
        fp.write(msg_str)


def customize_solver(source_path, dest_path, snapshot_prefix, net):
    """Customize a Caffe solver ``.prototxt`` so that it has the right snapshot
    prefix and net path.

    :param source_path: Path to the ``.prototxt`` file describing the solver.
    :param dest_path: Where the updated ``.prototxt`` will be written to.
    :param snapshot_prefix: Path prefix for net snapshots during training.
    :param net: Path to the actual net to train."""
    solver = read_prototxt(source_path, cf.proto.caffe_pb2.SolverParameter())
    solver.snapshot_prefix = snapshot_prefix
    solver.net = net
    save_prototxt(solver, dest_path)


def customize_train_net(source_path, dest_path, train_db_path, test_db_path):
    """Customize a Caffe net definition in ``.prototxt`` form so that it reads
    from the right databases.

    :param source_path: Path to ``.prototxt`` describing the net.
    :param dest_path: Where the updated net spec ``.prototxt`` will be written.
    :param train_db_path: Path to the training LMDB.
    :param test_db_path: Path to the validation LMDB."""
    # TODO: Mean file implementation. The nets I'm working with don't bother
    # with a mean file (it DOES look rather cosmetic), so I don't need it for
    # now.
    net = read_prototxt(source_path, cf.proto.caffe_pb2.NetParameter())

    # Get the data layers. This will fail if we have more than two test layers
    # (we shouldn't; we assume that we have ONE train layer and ONE test layer)
    train_layer, test_layer = [l for l in net.layers if l.type == l.DATA]

    # Make sure that we have the train layer and the test layer
    assert len(train_layer.include) == 1, len(test_layer.include) == 1
    TRAIN = cf.proto.caffe_pb2.TRAIN
    TEST = cf.proto.caffe_pb2.TEST
    if train_layer.include[0].phase != TRAIN:
        # Swap them!
        train_layer, test_layer = test_layer, train_layer
    assert train_layer.include[0].phase == TRAIN
    assert test_layer.include[0].phase == TEST

    # Now we can modify the layers in-place
    train_layer.data_param.source = train_db_path
    test_layer.data_param.source = test_db_path

    # Save the net again
    save_prototxt(net, dest_path)


def caffe_binary(name):
    """Computes the full path to a Caffe binary."""
    for ext in ['', '.bin']:
        full_path = find_executable(name + ext)
        if full_path is not None:
            return full_path


def compute_image_mean(lmdb_path, destination):
    """Uses the ``compute_image_mean`` Caffe tool to compute the mean pixel
    value of a training database."""
    cim_binary = caffe_binary('compute_image_mean')
    assert cim_binary is not None, "Could not find Caffe's compute_image_mean"
    logging.info('Calling compute_image_mean ({})'.format(cim_binary))
    pcall([
        cim_binary, '-backend', 'lmdb', lmdb_path, destination
    ])


def train_dcnn_patches(model, solver, gpu=None):
    """Uses patches stored in an LMDB file by ``make_patches()`` to train a
    CNN."""
    caffe_path = caffe_binary('caffe')
    assert caffe_path is not None
    command = [
        caffe_path, 'train',
        '-solver', solver,
        '-model', model,
    ]
    if gpu is not None:
        command.extend(['-gpu', str(gpu)])
    logging.info("Running Caffe with subprocess.call({})".format(command))
    pcall(command)
