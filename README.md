# comp2560

Code and papers for COMP2560 (Studies in Advanced Computing R&D) at ANU. This
semester, my project is to implement a system for human pose estimation from
videos. The aim of the system is to take as input as video of a person (or
people) performing some interesting actions, then output an annotation for each
video frame which shows the location of some key points on their body in that
frame. For instance, you might define your key points to be the {left, right}
{wrist, elbow, shoulder}, plus additional keypoints at the base of the neck and
in the middle of the face. Knowing the locations of these body parts can make it
easier to perform high-level tasks like gesture recognition.

Contents of this repository:

- `dcnn/` houses code related to body part detection in single frames. This is
  implemented using the [Caffe](http://caffe.berkeleyvision.org/) deep learning
  framework, and follows the approach of Chen & Yuille in [Articulated Pose
  Estimation by a Graphical Model with Image Dependent Pairwise
  Relations](http://www.stat.ucla.edu/~xianjie.chen/projects/pose_estimation/pose_estimation.html).

  My understanding of Chen & Yuille's reference implementation is that it uses a
  sliding window to create a heatmap for each body part. This technique is
  inefficient, since it requires multiple passes through the convnet for each
  input image. The aim of my implementation is to replace this
  naively-implemented sliding window with a fully convolutional network which
  will output the heat map directly. Hopefully this will make the network much
  faster to train.
- `thirdparty/` is for papers, code and data sets from other researchers. Some
  of those data sets are large, so I've used `git-annex` to store them.

## Copyright

It's Apache v2! Hallelujah!

> Copyright 2015 Sam Toyer
>
> Licensed under the Apache License, Version 2.0 (the "License");
> you may not use this file except in compliance with the License.
> You may obtain a copy of the License at
>
> [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
>
> Unless required by applicable law or agreed to in writing, software
> distributed under the License is distributed on an "AS IS" BASIS,
> WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
> See the License for the specific language governing permissions and
> limitations under the License.
