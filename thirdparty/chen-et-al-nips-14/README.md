## Articulated Pose Estimation by a Graphical Model with Image Dependent Pairwise Relations

This is an implementation of the articulated pose estimation algorithm described in [1]. Much of the code is built on top of the implementation of mixtures-of-parts [2] and deformable part models [3].

To illustrate the use of the training code, this package also includes images from the Leeds Sports Pose (LSP) Dataset [4], and the negative images from the INRIAPerson dataset [5].

Prerequisites: The code requires [Caffe](https://github.com/xianjiec/caffe/tree/dev) with customized matlab wrapper (i.e. matcaffe.cpp) to run.

Acknowledgements: We graciously thank the authors of the previous code releases and image benchmarks for making them publically available.

### References

[1] X. Chen, A. Yuille. Articulated Pose Estimation by a Graphical Model with Image Dependent Pairwise Relations. NIPS'14

[2] Y. Yang, D. Ramanan. Articulated Pose Estimation using Flexible Mixtures of Parts. CVPR'11

[3] P. Felzenszwalb, R. Girshick, D. McAllester, D. Ramanan. Object Detection with Discriminatively Trained Part Based Models. TPAMI'10. http://www.cs.berkeley.edu/~rbg/latent/index.html

[4] S. Johnson, M. Everingham. Clustered Pose and Nonlinear Appearance Models for Human Pose Estimation. BMVC'10. http://www.comp.leeds.ac.uk/mat4saj/lsp.html

[5] N. Dalal, B. Triggs. Histograms of Oriented Gradients for Human Detection. CVPR'05.


### Installing and Running

1. Install [Caffe](https://github.com/xianjiec/caffe/tree/dev) with the customized matlab wrapper under external/caffe. A symbolic link will work.
2. Make sure to compile the Caffe MATLAB wrapper, which is not built by default: make matcaffe
3. Start matlab, run 'startup' script.
4. Run the 'compile' script to compile the mex functions.
   (the script is tested under ubuntu 12.04, you may need to edit it depending on your system)
5. Run 'demo_lsp' to doing training and inference on the LSP dataset with benchmark evaluation.

