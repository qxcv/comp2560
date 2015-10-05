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

- At the moment, `project/` houses code related to body part detection in single
  frames. This is implemented using the
  [Caffe](http://caffe.berkeleyvision.org/) deep learning framework, and follows
  the approach of Chen & Yuille in [Articulated Pose Estimation by a Graphical
  Model with Image Dependent Pairwise
  Relations](http://www.stat.ucla.edu/~xianjie.chen/projects/pose_estimation/pose_estimation.html).

  Initially, I attempted to re-implement Chen & Yuille's method in Python, but
  this proved to be too much work for a single-term project, so I instead opted
  to extend their Matlab code, along with that of a few other researchers. The
  result
  is in `project/matlab/`.
- `thirdparty/` is for papers, code and data sets from other researchers. Some
  of those data sets are large, so I've used `git-annex` to store them.

## Copyright

Copyright is complicated. Stuff in `thirdparty/` was written by other
researchers, and licenses vary. The same goes for things in `project/matlab`,
which was largely written by other researchers (Chen, Yuille, Yang, Ramanan,
Cherian, etc.) and adapted by me, so licensing varies there, too. Everything
else (including the Python code in `project/` and the stuff in `report/`) is
Apache v2 licensed:

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
