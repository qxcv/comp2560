"""Test suite for dataset loading code."""

from os import path

import pytest

from datasets import LSP, LSPET

LSP_PATH = "../datasets/lsp/lsp_dataset.zip"
LSPET_PATH = "../datasets/lspet/lspet_dataset.zip"


@pytest.mark.skipif(not path.exists(LSP_PATH), reason="Need LSP .zip")
def test_lsp():
    lsp = LSP(LSP_PATH)
    joints = lsp.load_joints().locations
    assert joints.shape == (2000, 14, 3)
    # Should load im0042.jpg (i.e. image 41 + 1)
    img_42 = lsp.load_image(41)
    # It's a 134 (width) * 201 (height) image, but the image is row-major
    assert img_42.shape == (201, 134, 3)
    # Just skip this because it's slow. It doesn't run into memory issues,
    # though.
    # all_images = lsp.load_all_images()
    # assert len(all_images) == len(joints)
    # assert all_images[41].shape == img_42.shape


@pytest.mark.skipif(not path.exists(LSPET_PATH), reason="Need LSPET .zip")
def test_lspet():
    # As above, but for the larger LSPET dataset
    lsp = LSPET(LSPET_PATH)
    joints = lsp.load_joints().locations
    assert joints.shape == (10000, 14, 3)
    img_412 = lsp.load_image(411)
    # It's 245 (width) * 371 (height) but, again, the matrix is row-major
    assert img_412.shape == (371, 245, 3)
    # Commented out due to memory issues :P
    # all_images = lsp.load_all_images()
    # assert len(all_images) == len(joints)
    # assert all_images[411] == img_412
