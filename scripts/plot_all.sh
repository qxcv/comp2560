#!/bin/sh

# This just plots all of the stuff in the "stats" directory, but with meaningful
# titles

./plot_pck.py --input "300 poses" "../stats/default-conf-300-poses.csv" \
    --input "50 poses" "../stats/default-conf-50-poses.csv" \
    --input "1 pose" "../stats/default-conf-1-pose.csv" \
    --input "Improved shoulders" "../stats/100-poses-no-colour-or-skin-on-shoulders.csv" \
    --input "No skin/colour weights" "../stats/no-skin-or-colour-100-poses.csv" \
    --input "No skin/colour, tweaked" "../stats/no-skin-or-colour-1-shoulder-sumflow-100-poses.csv" \
    --input "No skin/colour, tweaked 2" "../stats/no-skin-or-colour-no-shoulder-motion-100-poses.csv" \
    $@
