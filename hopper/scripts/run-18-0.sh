#!/bin/bash

sudo /scratch/stuart/ncu/ncu \
    --replay-mode application \
    -c 1 \
    -s 0 \
    --kill 1 \
    --set source \
    -f --export ./profiler/18-mpi-less-multiprocess-r0.ncu-rep ./bin/18-mpi-less-multiprocess 0 0 2
