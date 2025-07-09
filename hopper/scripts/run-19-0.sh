#!/bin/bash

set -e

make

sudo /scratch/stuart/ncu/ncu \
    --set source \
    --launch-count 1 \
    --launch-skip 0 \
    -f --export ./profiler/19-all-reduce-no-locking-r0.ncu-rep \
    ./bin/19-all-reduce-no-locking 0 0 2
