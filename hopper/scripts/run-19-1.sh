#!/bin/bash

sudo /scratch/stuart/ncu/ncu \
    --set source \
    --launch-count 1 \
    --launch-skip 0 \
    -f --export ./profiler/19-all-reduce-no-locking-r1.ncu-rep \
    ./bin/19-all-reduce-no-locking 1 1 2 "$1"
