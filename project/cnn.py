"""Code for training and using relevant CNNs. Uses pycaffe underneath."""

import logging
from subprocess import call as pcall
from distutils.spawn import find_executable

import caffe as cf
from google.protobuf.text_format import Merge, MessageToString
import lmdb
import numpy as np
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
            if (sample_id % 50) == 0:
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
        # For array_to_datum, the dimensions are (channels, height, width).
        # This means need to swap width and height, then transpose the entire
        # Numpy array
        trans_scaled = scaled.transpose((2, 0, 1))
        datum = cf.io.array_to_datum(trans_scaled, label=label)
        assert datum.height == scaled.shape[0]
        assert datum.width == scaled.shape[1]
        assert datum.channels == 3
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


def transplant_weights(source_spec, source_params, dest_spec, dest_params):
    """Makes a Caffe net fully convolutional.

    :param source_spec: Path to source (FC) net.
    :param source_modeL: Path to trained source model.
    :param dest_spec: Specification of destination (fully convolutional)
        network.
    :param dest_params: Path to store model in once we've had our way with
    it."""
    # Names of layers to reshape and copy
    names = [('fc6', 'fc6-conv'), ('fc7', 'fc7-conv'), ('fc8', 'fc8-conv')]

    # Load the nets, copy whatever weights we can
    fc_net = cf.Net(source_spec, source_params, cf.TEST)
    conv_net = cf.Net(dest_spec, cf.Test)
    conv_net.copy_from(source_params)

    for fc_name, conv_name in names:
        fc_params = fc_net.params[fc_name]
        conv_params = conv_net.params[conv_name]

        # Usually there will be a weights blob and a biases blob in each layer
        # we want to replace
        assert len(fc_params) == len(conv_params)

        for blob_idx in len(fc_params):
            old_shape = tuple(fc_params[blob_idx].shape)
            new_shape = tuple(conv_params[blob_idx].shape)
            # Reshape FC weights to match convolutional layer dimensions.
            # See net_surgery.ipynb for some notes on how to get this right.
            conv_params[blob_idx][...] = fc_params[blob_idx].reshape(new_shape)
            logging.info('Converting {}->{}, blob {}: {}->{}'.format(
                fc_name, conv_name, blob_idx, old_shape, new_shape
            ))

    conv_net.save(dest_params)


def compute_pyramid(net, psize, step, interval, image, mean_pixel=None):
    """Compute a pyramid of CNN-derived features for the given image. Similar
    to ``impyra.m``, except we haven't bothered upscaling, since Chen & Yuille
    don't upscale anyway.

    :param net: a ``caffe.Net`` instance corresponding to the fully
        convolutional "deploy" model.
    :param psize: parameter from Chen & Yuille. It's actually ``step * tsize``,
        where ``tsize`` is a kind of "natural" template size computed from the
        dimensions of skeletons in the training set. Unlike Chen & Yuille, we
        use **ROW MAJOR ORDER** for ``psize``!
    :param step: yet another parameter from Chen & Yuille. I actually have no
        idea what this corresponds to, intuitively.
    :param interval: how many pyramid levels does it take to halve the data
        size?
    :param image: ``h * w * c`` ``ndarray`` representing a single input image.
    :param mean_pixel: optional mean pixel argument.
    :returns: list of dictionaries with ``output_size``,
        ``{width,height}_pad``, ``scale`` and ``features`` keys. Each entry in
        the list corresponds to a level of the feature pyramid (largest scale
        first). The ``features`` key is an "image" representing the fully
        convolutional netowrk output, where the number of channels in the image
        is equal to the number of softmax outputs in the CNN."""
    assert image.ndim == 3 and image.shape[2] == 3

    if mean_pixel is None:
        mean_pixel = 128 * np.ones((3,))
    else:
        # Flip the mean pixel to BGR
        mean_pixel = mean_pixel[::-1]

    height_pad, width_pad = np.maximum(np.ceil((psize - 1) / 2.0), 0)\
        .astype('int')
    scale = 2 ** (1.0 / interval)
    image_size = np.array(image.shape[:2])
    max_scale = int(1 + np.floor(np.log(np.min(image_size)) / np.log(scale)))
    # This will have keys 'output_size', 'scale', 'height_pad', 'width_pad',
    # 'features'
    rv = [{} for _ in xrange(max_scale)]

    # A natural size, I guess
    max_batch_size = interval
    for batch_level in xrange(0, max_scale, max_batch_size):
        batch_size = np.min(max_batch_size, max_scale - batch_level)
        base_dims = image_size / scale ** (batch_level)
        scaled = cf.io.resize(image, base_dims.astype('int'))

        # This next array will be passed to Caffe
        caffe_input = np.zeros((
            batch_size,
            3,
            scaled.shape[1] + 2 * height_pad,
            scaled.shape[0] + 2 * width_pad,
        ))

        for sublevel in xrange(batch_level, batch_level + batch_size):
            # Pad and add to Caffe input
            pad_dims = (2 * (height_pad,), 2 * (width_pad,), 2 * (0,))
            padded = np.pad(scaled, pad_dims, mode='edge') - mean_pixel
            max_row, max_col = padded.shape[:2]
            caffe_input[sublevel - batch_level, :, :max_row, :max_col] = \
                padded.transpose((2, 0, 1))

            # Store metadata
            info = rv[sublevel]
            info['output_size'] = np.floor(
                (padded.shape[:2] - psize) / float(step)
            ).astype('int') + 1
            info['scale'] = step * scale ** (sublevel - 1)
            info['width_pad'] = width_pad / float(step)
            info['height_pad'] = height_pad / float(step)

            # Resize for the next step
            base_dims /= scale
            scaled = cf.io.resize(image, base_dims.astype('int'))

        # To do a fully convolutional forward pass, we just reshape the data
        # layer and let the rest follow
        net.blobs['data'].reshape(*caffe_input.shape)
        net.blobs['data'].data[...] = caffe_input
        # TODO: What does result contain? Apparently it's a dictionary mapping
        # blob names to ndarrays for those blobs. In this case, I guess we'll
        # have a batch_size * softmax_outputs * something * something_else
        # ndarray, where something and something_else will be decided by
        # some annoying arithmetic on strides, pads and steps. Ugh, gross.
        result = net.forward()['prob']

        for sublevel in xrange(batch_level, batch_level + batch_size):
            info = rv[sublevel]
            max_row, max_col = info['output_size']
            info['features'] = result[
                sublevel - batch_level, :, :max_row, :max_col
            ].transpose((1, 2, 0))

    return rv


def evaluate_cnn(image, net):
    """Run a fully convolutional CNN over the given image, returning feature
    layers and unaries."""
    pass
