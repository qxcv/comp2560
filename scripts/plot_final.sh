#!/bin/sh

dir="../stats/final"

./plot_pck.py --input "Default config" "$dir/final-config.csv" \
    --input "Static-only" "$dir/final-config-static-only.csv" \
    --input "Cherian et al." "$dir/anoop-results.csv" \
    $@
    # --input "No NMS" "$dir/final-config-no-nms.csv" \
