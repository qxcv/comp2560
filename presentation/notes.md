# COMP2560 presentation notes

## Intro slide

- Introduce self, supervisor
- Name problem, but don't explain
- Move on to motivation as fast as possible

## Motivation

- Pose estimator takes video frames and outputs "skeleton" for each frame
- Skeleton consists of "joint" locations (not really joints)
- Skeleton also has "limbs"; sometimes these are useful for expressing
  constraints between joint positions
- Applications:
  - Use limb locations to figure out what clothing a person is wearing. Apply to
    clothing recommendation, marketing.
  - Recognise actions by looking at way limbs move
  - Skeletons give clues as to identity of objects---if I'm in sitting position,
    there's probably a chair under me
- Aim of project: improve accuracy of pose estimation! Benefits all
  applications.

## Existing approaches

- Common approach is graphical model
  - Use joint detector to figure out which parts of the image "look like" each
    joint
  - Constraints corresponding to limbs make sure that we have anatomically
    plausible poses
- How to figure out which parts of the image "look like" each joint?
  Histogram-of-gradients!
- Just looking at gradients not so informative
- CNNs have produced much better results; better features
- How to use motion data?
- Extending graphical model with links between frames makes it really hard to
  figure out where each joint is
- Instead, use approximations; cite Anoop's work (will cover later)

## A hybrid approach

- Two stages: generate a set of candidate poses for each frame using Chen &
  Yuille detector, then choose appropriate candidates to get temporally
  consistent pose sequence
- Make it clear that this is two stages!

## Pose model

- Two models:
  - To generate candidate sets in each frame, use 18 part model. Lots of parts
    improves accuracy by allowing more accurate modelling of limb deformations.
  - When combining candidate poses, only use 7 part model, since 18 part model
    isn't as helpful
- 7-part model still covers most important parts
- Also introduce notion of types when generating candidate sets; these aren't
  used by later stages of the pipeline, but they constrain limbs positions more
  accurately. Types can be inferred from image patches.

## Per-frame candidate sets

- When generating a set of candidate poses for each frame, use cost depending
  on joints and limbs:
  - Likelihood of seeing joint at location
  - How likely is length and orientation of limb given type?
  - How likely is limb type given image patches at either end?
- Optimise by looping over all combinations of locations and types. Can be sped
  up using dynamic programming and distance transforms
- Instead of just finding the best pose, try to find the N best poses:
  - Optimisation process produces list of head locations with scores for
    best-scoring pose associated with that head location
  - Go back through original model to collect corresponding locations for each
    joint
  - Trim out poses which are just pixel-shifted copies using NMS
- Now we have a reasonably diverse set of candidate poses. Hopefully one of
  these will be correct.
- How do we figure out likelihood of joint being in patch of image? CNNs!

## Convolutional neural networks I

- Look at a tiny little patch
- For each joint in the skeleton, and each possible combination of adjacent limb
  types, tell us how likely the combination is to be present in that patch
- Convolutional neural network learns to do this from examples
- Much more effective than old approaches using HoG

## Convolutional neural networks II

- If you do this for each patch in the image, then you get a heatmap
- One heatmap for each joint and combination of adjacent limb types
- Value of pixel in heatmap gives costs for graphical model

## Recombination

- Now we have plausible poses in each frame, but we want a single best pose for
  each frame
- Instead of choosing the best pose, break the poses up and choose the best
  *limb*
- Start with the head: consider set of head positions from each candidate set
  and find temporally consistent sequence using GM-like cost
- Compute optical flow between frames, and try to make relative positions of
  heads agree with flow
- Proceed to shoulders, then elbows, then wrists
- For shoulders/elbows/wrists, also take into account distance from previous
  joint to beginning of new limb
- You end up with a single best pose in each frame!

## Results

- Evaluate on Poses in the Wild
- Look at accuracy for each joint, averaged over all frames
- Joint is correctly localised if it's within threshold of ground truth
- Vary threshold to see how far we are from the true position
- Improves results over either base approach for fast-moving joints (elbows,
  wrists)
- Not much for shoulders, though (result occluded)
- All approaches much, *much* better than approach from 4 years ago
- Wrists and elbows are the hardest parts to estimate---they move quickly, they
  get confused with clutter in the background, etc. Thus, this is a good result.

## Discussion I

- Limb-like clutter is a problem
- Lots of challenging sequences in data set; second sequence was almost entirely
  incorrect due to out-of-frame limbs and foreground obstruction
- Motion also a problem, but this is to be expected, esp. since flow is not as
  accurate once blurring takes place

## Discussion II

- Performance is biggest issue
- CNN produces huge number of heat maps
  - One for each joint and *every combination of types for its neighbours*!
  - Exacerbated by multi-scale
  - >500MiB of data---takes ages to do anything with (takes as long to copy as
    to forward propagate!)
- Simplify by reducing number of heat maps. Try not using types!
- Maybe try different architecture (regressor?)

## Summary

- Take questions
- Skip to CNN slide, if possible
