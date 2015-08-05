"""Functions for reading data sets (LSP, INRIA, Buffy, etc.)"""

from abc import abstractmethod, ABCMeta
from io import BytesIO
from zipfile import is_zipfile, ZipFile

from scipy.io import loadmat
from scipy.misc import imread


class DataSet(object):
    """ABC for datasets"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_joints(self):
        pass

    @abstractmethod
    def load_image(self, identifier):
        pass

    @abstractmethod
    def load_all_images(self):
        pass


class LSP(DataSet):
    """Loads the Leeds Sports Poses dataset from a ZIP file. Return value holds
    a 2000x14x3 ndarray. The first dimension selects an image, the second
    selects a joint, and the final dimension selects between an x-coord, a
    y-coord and a visibility."""
    PATH_PREFIX = 'lsp_dataset/'
    ID_WIDTH = 4

    def __init__(self, lsp_path):
        assert is_zipfile(lsp_path), "Supplied path must be to lsp_dataset.zip"
        self.lsp_path = lsp_path

    def _transpose_joints(self, joints):
        return joints.T

    def load_joints(self):
        with ZipFile(self.lsp_path) as zip_file:
            target = self.PATH_PREFIX + 'joints.mat'
            buf = BytesIO(zip_file.read(target))
            mat = loadmat(buf)
            # TODO: Return something a little more user-friendly. In
            # particular, I should check whether Numpy supports some sort
            # of naming for fields.
            return self._transpose_joints(mat['joints'])

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
        num_images = len(self.load_joints())
        return [self.load_image(ident) for ident in range(num_images)]


class LSPET(LSP):
    """Like LSP, but specific to the Leeds Extended Poses dataset."""
    PATH_PREFIX = ''
    ID_WIDTH = 5

    def _transpose_joints(self, joints):
        return joints.transpose((2, 0, 1))
