#!/bin/bash

sudo /scratch/stuart/ncu/ncu \
    --replay-mode kernel \
    --set full -f --export ./profiler/18-mpi-less-multiprocess-r1.ncu-rep ./bin/18-mpi-less-multiprocess 1 1 2 "$1"
