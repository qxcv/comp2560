"""Tests for code in ``util.py``"""

from itertools import combinations
from time import sleep

import numpy as np

from util import unique_key, sample_patch


def test_unique_key():
    keys = [unique_key() for i in xrange(10)]
    sleep(0.1)
    keys += [unique_key() for i in xrange(10)]
    for first, second in combinations(keys, 2):
        assert first != second
        # 12 chars corresponds to 8 bytes, base64 encoded
        assert len(first) == 12
        assert len(second) == 12


def test_sample_patch():
    test_array = np.array([
        [84,   29,  39,  67,  10,   3, 111, 150, 236,  91],
        [55,   93, 248, 152, 195, 241, 169,  27, 104, 230],
        [195, 105, 254,  50, 222,  49, 109,  91, 186, 238],
        [134,  36,  46,  54, 218, 179, 179, 103,  93, 199],
        [114, 230, 143, 114, 224,  74, 236,  90, 236, 192],
        [193, 118,  48, 186, 135,  53, 147, 191, 229,  63],
        [100,  93, 192,  80, 212,  59, 186, 215, 242, 159],
        [129, 141, 212,   8, 185,  93, 103, 243, 157, 163],
        [95,  168,  79, 188, 162,  73,  45,  15, 122, 168],
        [9,   170, 129, 245,  89,   0, 204, 229, 218,  96]
    ], dtype='uint8')

    # Sample a 5x3 patch from the near the middle
    standard_sample = sample_patch(test_array, 5, 5, 3, 5)
    standard_result = np.array([
        [218, 179, 179],
        [224,  74, 236],
        [135,  53, 147],
        [212,  59, 186],
        [185,  93, 103]
    ], dtype='uint8')
    assert standard_sample.shape == standard_result.shape
    assert (standard_sample == standard_result).all()

    # Sample a 3x3 patch from the top right-hand corner
    oob_sample = sample_patch(test_array, 0, 0, 3, 3)
    oob_result = np.array([
        [84, 84, 29],
        [84, 84, 29],
        [55, 55, 93]
    ], dtype='uint8')
    assert oob_sample.shape == oob_result.shape
    assert (oob_sample == oob_result).all()
