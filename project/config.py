"""Module for dealing with configurations. Also includes a default
configuration. You shouldn't need to edit this file; instead, copy one of the
example ``.cfg``s and make your changes to that, then pass a path to your
modified file to ``train.py``."""

from ConfigParser import SafeConfigParser
from StringIO import StringIO

# Path to Caffe models (saves space in the default config)
MOD_PATH = "./caffe-models/chen-and-yuille-nips-14/lsp"
# Default configuration, in Python's .cfg syntax
# DON'T CHANGE THIS. Instead, write your own .cfg file with the options you
# wish to change (e.g. add a [graphical_model] section with centroids_per_limb
# = 20 if you just need to change the number of centroids per limb) and load
# that with the --conf option.
DEFAULTS = """
[graphical_model]
centroids_per_limb: 13

[dataset]
loader: LSP
path: ../datasets/lsp/lsp_dataset.zip

[cnn]
solver: {MOD_PATH}/lsp_solver.prototxt
train_net: {MOD_PATH}/lsp_train_val.prototxt
test_net: {MOD_PATH}/lsp_deploy.prototxt
fully_conv_test_net: {MOD_PATH}/lsp_deploy_conv.prototxt
""".format(MOD_PATH=MOD_PATH)


def from_files(files):
    """Read a config file from a given list of file pointers."""
    rv = SafeConfigParser()
    rv.readfp(StringIO(DEFAULTS))
    for fp in files:
        rv.readfp(fp)
    return rv
