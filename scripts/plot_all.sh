#!/bin/sh

# This just plots all of the stuff in the "stats" directory, but with meaningful
# titles

./plot_pck.py --input "300 poses" "../stats/default-conf-300-poses.csv" \
    --input "50 poses" "../stats/default-conf-50-poses.csv" \
    --input "1 pose" "../stats/default-conf-1-pose.csv" \
    --input "No skin/colour weights" "../stats/no-skin-or-colour-100-poses.csv" \
    --input "No skin/colour/motion" "../stats/no-skin-no-colour-no-motion-100-poses.csv" \
    --input "Best shoulders" "../stats/no-skin-or-colour-best-shoulders-150-poses.csv" \
    $@
