#!/bin/sh

dir="../stats/final"

./plot_pck.py --input "Ours" "$dir/final-config.csv" \
    --input "Chen and Yuille" "$dir/final-config-static-only.csv" \
    --input "Cherian et al." "$dir/anoop-results.csv" \
    $@
    # --input "No NMS" "$dir/final-config-no-nms.csv" \
