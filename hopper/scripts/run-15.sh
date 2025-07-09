#!/bin/bash

set -e 

cd ./15-nccl
make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90a"
cd ..
make clean
make run
