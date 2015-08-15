"""Functions for reading data sets (LSP, INRIA, Buffy, etc.)"""

from abc import abstractmethod, ABCMeta
from copy import copy
from io import BytesIO
from zipfile import is_zipfile, ZipFile

import numpy as np

from scipy.io import loadmat
from scipy.misc import imread


def split_items(items, num_groups):
    """Splits a list of items into ``num_groups`` groups fairly (i.e. every
    item is assigned to exactly one group and no group is more than one item
    larger than any other)."""
    per_set = len(items) / float(num_groups)
    assert per_set >= 1, "At least one set will be empty"
    small = int(np.floor(per_set))
    big = small + 1
    num_oversized = len(items) % small

    rv_items = []
    total_allocated = 0
    for i in range(num_groups):
        if i < num_oversized:
            l = items[total_allocated:total_allocated + big]
            total_allocated += big
        else:
            l = items[total_allocated:total_allocated + small]
            total_allocated += small
        rv_items.append(l)

    assert total_allocated == len(items), "Did not assign exactly 100% of " \
        "items to a group"
    assert len(rv_items) == num_groups, "Wrong number of groups"

    return rv_items


class DataSet(object):
    """ABC for datasets"""
    __metaclass__ = ABCMeta

    def post_init(self):
        """Should be called after __init__."""
        self.num_samples = len(self.joints.locations)

        self.scales = self._calculate_scales()
        assert np.all(self.scales >= 18)
        assert np.any(self.scales > 18)
        assert self.scales.shape == (self.num_samples,)

        self.template_size = self._calculate_template_size()
        assert self.template_size.shape == (2,)
        assert np.all(self.template_size > 0)

    def split(self, num_groups):
        """Splits one monolothic dataset into several equally sized
        datasets. May need to be overridden."""
        assert num_groups > 1, "It's not splitting if there's < 2 groups :)"

        # Shallow-copy myself several times
        rv = tuple(copy(self) for i in range(num_groups))

        # Figure out which indices each group will get
        indices = np.arange(self.num_samples)
        np.random.shuffle(indices)
        rv_indices = split_items(indices, num_groups)

        for new_dataset, new_indices in zip(rv, rv_indices):
            new_dataset.joints = self.joints.for_indices(new_indices)
            new_dataset.image_ids = np.array(self.image_ids)[new_indices]

        return rv

    def _calculate_scales(self):
        """Calculates a scale factor for each image in the dataset. This is
        indended to indicate roughly how long the average limb is in each image
        (in pixels), so that images taken at different distances from a person
        can be considered differently for joint RP (relative position)
        clustering and the like. Magic constants (75th percentile, 18px
        minimum) taken from Chen & Yuille's code"""
        lengths = np.zeros((self.num_samples, len(self.joints.pairs)))

        # If the length of a limb is 0, then we'll mark it as invalid for our
        # calculations
        valid = np.ones_like(lengths, dtype=bool)  # array of True

        for idx, pair in enumerate(self.joints.pairs):
            fst_prt, snd_prt = pair
            fst_loc = self.joints.locations[:, fst_prt, :2]
            snd_loc = self.joints.locations[:, snd_prt, :2]
            assert fst_loc.shape == (self.num_samples, 2)
            assert fst_loc.shape == snd_loc.shape
            # lengths stores the length of each limb in the model
            pair_dists = np.linalg.norm(fst_loc - snd_loc, axis=1)
            lengths[:, idx] = pair_dists

            # Mark zeros invalid
            valid[pair_dists == 0, idx] = False

        # The last limb is head-neck (we can consider this the "root" limb,
        # since we assume that the head is the root for graphical model
        # calculations). We will normalise all lengths to this value.
        exp_med = np.zeros(len(self.joints.pairs) - 1)
        for idx, pair in enumerate(self.joints.pairs[:-1]):
            valid_col = valid[:, idx]
            log_neck = np.log(lengths[valid_col, -1])
            log_diff = np.log(lengths[valid_col, idx]) - log_neck
            exp_med[idx] = np.exp(np.median(log_diff))

        print(exp_med)

        # Norm calculated lengths using the exponent of the median of the
        # quantities we calculated above
        norm_factor_nc = exp_med.reshape((1, -1))
        norm_factor = np.concatenate([norm_factor_nc, [[1]]], axis=1)
        assert norm_factor.shape == (1, len(self.joints.pairs))
        normed_lengths = lengths / norm_factor

        percentiles = np.percentile(normed_lengths, 75, axis=1)
        assert percentiles.ndim == 1
        assert len(percentiles) == self.num_samples

        assert not np.any(np.isnan(percentiles) + np.isinf(percentiles))

        # NOTE: Chen & Yuille use scale_x and scale_y, but that seems to be
        # redundant, since scale_x == scale_y in their code (init_scale.m)
        return np.maximum(percentiles, 18)

    def _calculate_template_size(self):
        """Use calculated scales to choose template sizes for body part
        detection. Follows Chen & Yuille formula."""
        # This is a little different to Chen & Yuille's formula (they use a
        # fixed aspect ratio, and calculate a square root which makes no sense
        # in context), but it should yield the same result
        side_lengths = 2 * self.scales + 1
        assert side_lengths.shape == (self.num_samples,)

        bottom_length = np.percentile(side_lengths, 1)
        template_side = np.floor(bottom_length / self.STEP)

        return np.array([template_side, template_side])

    @abstractmethod
    def load_image(self, identifier):
        pass

    @abstractmethod
    def load_all_images(self):
        pass


class Joints(object):
    """Class to store the locations of key points on a person and the
    connections between them."""
    def __init__(self, point_locations, joint_pairs, point_names=None):
        # First, some sanity checks
        as_set = set(tuple(sorted(p)) for p in joint_pairs)
        assert len(as_set) == len(joint_pairs), "There are duplicate joints"
        assert isinstance(point_locations, np.ndarray), "Point locations " \
            "must be expressed as a Numpy ndarray"
        assert point_locations.ndim == 3, "Point location array must be 3D"
        assert point_locations.shape[2] == 3, "Points must have (x, y) " \
            "location and visibility."
        num_points = point_locations.shape[1]
        for first, second in joint_pairs:
            assert 0 <= first < num_points and 0 <= second < num_points, \
                "Joints must be between extant points."
        assert point_locations.shape[1] < 64, "Are there really 64+ points " \
            "in your pose graph?"
        if point_names is not None:
            assert len(point_names) == point_locations.shape[1], "Need as " \
                "many names as points in pose graph."

        # We can access these directly
        self.pairs = joint_pairs
        self.locations = point_locations
        self.point_names = point_names

    def for_indices(self, indices):
        """Takes a series of indices corresponding to data samples and returns
        a new ``Joints`` instance containing only samples corresponding to
        those indices."""
        return Joints(self.locations[indices], self.pairs, self.point_names)

    # TODO: Enable visualisation of points! This would be a good idea if I
    # wanted to check that my skeletons are correct.


class LSP(DataSet):
    """Loads the Leeds Sports Poses dataset from a ZIP file."""
    PATH_PREFIX = 'lsp_dataset/'
    # ID_WIDTH is the number of digits in the LSP image filenames (e.g.
    # im0022.jpg has width 4).
    ID_WIDTH = 4
    # TODO: Clarify what this does. It's analogous to conf.step (in lsp_conf
    # and flic_conf) from Chen & Yuille's code.
    STEP = 4
    POINT_NAMES = [
        "Right ankle",     # 0
        "Right knee",      # 1
        "Right hip",       # 2
        "Left hip",        # 3
        "Left knee",       # 4
        "Left ankle",      # 5
        "Right wrist",     # 6
        "Right elbow",     # 7
        "Right shoulder",  # 8
        "Left shoulder",   # 9
        "Left elbow",      # 10
        "Left wrist",      # 11
        "Neck",            # 12
        "Head top"         # 13
    ]
    # NOTE: "Root" joint should be first!
    JOINTS = [
        (0, 1),    # Right shin
        (1, 2),    # Right thigh
        (2, 8),    # Right side of body
        (3, 4),    # Left thigh
        (3, 9),    # Left side of body
        (4, 5),    # Left shin
        (6, 7),    # Right forearm
        (7, 8),    # Right upper arm
        (8, 12),   # Right shoulder
        (9, 10),   # Left upper arm
        (9, 12),   # Left shoulder
        (10, 11),  # Left forearm
        (12, 13),  # Neck and head
    ]

    def __init__(self, lsp_path):
        assert is_zipfile(lsp_path), "Supplied path must be to lsp_dataset.zip"
        self.lsp_path = lsp_path
        self.joints = self._load_joints()
        self.image_ids = list(range(len(self.joints.locations)))

        self.post_init()

    def _transpose_joints(self, joints):
        return joints.T

    def _load_joints(self):
        """Load ``joints.mat`` from LSP dataset. Return value holds a 2000x14x3
        ndarray. The first dimension selects an image, the second selects a
        joint, and the final dimension selects between an x-coord, a y-coord
        and a visibility."""
        with ZipFile(self.lsp_path) as zip_file:
            target = self.PATH_PREFIX + 'joints.mat'
            buf = BytesIO(zip_file.read(target))
            mat = loadmat(buf)
            # TODO: Return something a little more user-friendly. In
            # particular, I should check whether Numpy supports some sort
            # of naming for fields.
            point_locations = self._transpose_joints(mat['joints'])
            return Joints(point_locations, self.JOINTS, self.POINT_NAMES)

    def load_image(self, zero_ident):
        """Takes an integer image idenifier in 0, 1, ..., 1999 and returns an
        associated image. The image will have dimensions corresponding to row
        number, column number and channels (RGB, usually)."""
        assert isinstance(zero_ident, int)
        ident = zero_ident + 1
        assert ident > 0
        # Images start from 1, not 0
        str_ident = str(ident).zfill(self.ID_WIDTH)
        file_path = self.PATH_PREFIX + 'images/im' + str_ident + '.jpg'
        with ZipFile(self.lsp_path) as zip_file:
            with zip_file.open(file_path) as image_file:
                return imread(image_file)

    def load_all_images(self):
        """Return a list of all images in the archive, ordered to correspond to
        joints matrix."""
        return [self.load_image(ident) for ident in self.image_ids]


class LSPET(LSP):
    """Like LSP, but specific to the Leeds Extended Poses dataset."""
    PATH_PREFIX = ''
    ID_WIDTH = 5

    def _transpose_joints(self, joints):
        return joints.transpose((2, 0, 1))
