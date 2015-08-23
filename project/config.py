"""Module for dealing with configurations. Also includes a default
configuration. You shouldn't need to edit this file; instead, copy one of the
example ``.cfg``s and make your changes to that, then pass a path to your
modified file to ``train.py``."""

from ConfigParser import SafeConfigParser
from StringIO import StringIO

DEFAULTS = """
[graphical_model]
centroids_per_limb = 13

[dataset]
loader: LSP
path: ../datasets/lsp/lsp_dataset.zip
"""


def from_files(files):
    """Read a config file from a given list of file pointers."""
    rv = SafeConfigParser()
    rv.readfp(StringIO(DEFAULTS))
    for fp in files:
        rv.readfp(fp)
    return rv
