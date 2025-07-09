#!/bin/bash

NSYS_PATH="/usr/local/cuda-12/bin/nsys"
PYTHON_PATH="/scratch/stuart/multi-gpu-experiments/venv/bin/python"
SCRIPT_PATH="14-mha-dist.py"

# Get script path from first argument if provided, otherwise use default
if [ $# -eq 1 ]; then
    SCRIPT_PATH=$1
fi

sudo $NSYS_PATH profile \
	--stats=true \
	--trace cuda,osrt,nvtx \
	--gpu-metrics-devices=all \
	--force-overwrite=true \
	-o ./profiler/${SCRIPT_PATH}.nsys-rep $PYTHON_PATH $SCRIPT_PATH
