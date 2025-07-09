#!/bin/bash

set -e

# Uncomment if using custom NCCL build
# cd ./20-nccl
# make -j src.build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90a"
# cd ..
# make clean

make

# First run Nsys
sudo nsys profile \
    --stats=true \
    --trace cuda,osrt,nvtx,python-gil,syscall \
    --gpu-metrics-devices=all \
    --cuda-memory-usage true \
    --force-overwrite=true \
    -o ./profiler/20-all-reduce-no-locking-f.nsys-rep \
    ./bin/20-all-reduce-no-locking-f 0 0 2 &

sleep 10
echo "Running peer program..."
./bin/20-all-reduce-no-locking-f 1 1 2

wait

# We must run application mode since we must generate the NCCL unique ID every time
sudo /scratch/stuart/ncu/ncu \
    --set full \
    --replay-mode application \
    -f --export ./profiler/20-all-reduce-no-locking-f.ncu-rep \
    ./bin/20-all-reduce-no-locking-f 0 0 2 &

NCU_PID=$!

# Will do about ~40 replays. After that, you must manually quit (Ctrl+C)

inotifywait -m -e modify /scratch/stuart/ncclUniqueId.txt | while read file event; do
    # For every replay that NCU does, we must run the peer program (otherwise it will hang)
    # if ! kill -0 $NCU_PID 2>/dev/null; then
    #     echo "NCU process has completed. Exiting loop."
    #     break
    # fi
    sleep 3
    echo "Running peer program..."
    ./bin/20-all-reduce-no-locking-f 1 1 2
done

wait
