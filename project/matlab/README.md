# Matlab code for CNN-based video pose estimation

The aim of this code is to use a CNN and graphical model which are pre-trained
for static image pose estimation to do pose estimation in videos. Specifically,
we take a model trained using Chen & Yuille's code on the FLIC data set, then
use a slightly different graphical model to perform inference on a set of video
frames (at the moment, frames from the Poses in the Wild set).

Whilst a lot of code will have to be (or has been) rewritten, those
modifications are mostly mixing-and-matching of existing code. The one novel
thing is a Python-based system for doing forward passes through the CNN; I found
that Chen & Yuille's code leaked memory quite badly when using fully
convolutional networks, so I rewrote it in the hope that whatever leak was
present in Chen's Caffe fork is no longer present in BVLC's ``master``.

This code base is forked from Cherian, Mairal, Alahari & Schmid's video pose
estimation code base, combined with Chen & Yuille's static pose estimation code
base (in CY). Note that much of Yang & Ramanan's codebase is included as well
(in YR).

## Original README (from Cherian et al.) converted to Markdown

If you use this code, please cite:

```
@InProceedings{Cherian14,
  author    = "Cherian, A. and Mairal, J. and Alahari, K. and Schmid, C.",
  title     = "Mixing Body-Part Sequences for Human Pose Estimation",
  booktitle = "Proceedings of IEEE Conference on Computer Vision and Pattern Recognition",
  year      = "2014"
}
```

How to use this package?

There is a `demo.m` in the package that you should be able to run. I have
included one sample sequence in the package to test if everything works. To run
on the full "poses in the wild" dataset, you need to download and unzip the
dataset and set the correct paths in `set_algo_parameters.m`. As the code caches
intermediate results (such as the flow,  pose candidates, etc.), you might need
to provide a larger disk space for the cache (~15GB free space or more).

I have also included the Yang and Ramanan pose implementation and a Matlab
implementation of the LDOF optical flow, to make the package self-contained.

For any bugs or issues, contact Anoop Cherian at
[anoop.cherian@inria.fr](mailto:anoop.cherian@inria.fr).
