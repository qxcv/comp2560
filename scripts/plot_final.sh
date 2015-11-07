#!/bin/bash

dir="../stats/final"

./plot_pck.py --input "Hybrid" "$dir/final-config.csv" \
    --input $'Chen and\nYuille' "$dir/final-config-static-only.csv" \
    --input "Cherian et al." "$dir/anoop-results.csv" \
    --input $'Yang and\nRamanan' "$dir/yang-ramanan-results.csv" \
    $@
    # --input "No NMS" "$dir/final-config-no-nms.csv" \
