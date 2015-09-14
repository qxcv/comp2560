# Datasets for training Chen & Yuille's model

If you wish to use Chen & Yuille's code to train a model yourself, then you
should put the appropriate data sets in this directory. Specifically, to train a
FLIC-based model, you should obtain the following:

1. The [basic FLIC data set](http://vision.grasp.upenn.edu/video/FLIC.zip)
   (`FLIC.zip`). This should be extracted into a subfolder of this directory
   named `FLIC/`
2. Negatives from the [INRIA person data
   set](http://pascal.inrialpes.fr/data/human/). Specifically, you should obtain
   the `neg` subdirectory, rename it `INRIA`, and paste it in this directory.
   **TODO:** is this information correct? I'm just assuming that this is what
   Chen & Yuille did.
