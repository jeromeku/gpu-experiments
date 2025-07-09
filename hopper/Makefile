##########################
# SOURCE FILES
##########################

# Source files
# SRC=00-naive-cpu-gemm.cu
# SRC=01-naive-gpu-gemm.cu
# SRC=01-tk-attn.cu
# SRC=02-basic-nccl.cu
# SRC=03-dumb-dist-matmul.cu
# SRC=04-all-reduce.cu
# SRC=05-pinned-bandwidth-test.cu
# SRC=06-stream-test.cu
# SRC=07-coalescing-test.cu
# SRC=08-shared-memory-test.cu
# SRC=09-transpose-test.cu
# SRC=10-bandwidth-test.cu
# SRC=15-simple-p2p.cu
# SRC=16-raw-p2p.cu
# SRC=18-mpi-less-multiprocess.cu
# SRC=19-all-reduce-no-locking.cu
# SRC=20-all-reduce-no-locking-f.cu
# SRC=21-multicast-object.cu
# SRC=22-multimem.cu
# SRC=23-multimem-reduce.cu
# SRC=24-multimem-vec-reduce.cu
# SRC=25-ring-reduce-impl.cc
# SRC=26-raw-ring-reduce.cu
# SRC=27-bandwidth-test.cu
# SRC=28-all-reduce-proper-bench.cu
# SRC=28-multimem-reduce-optimize.cu
# SRC=29-basic-threadpool.c
# SRC=30-multimem-reduce-opt-tp.cu
# SRC=31-multimem-init-benchmarks.cu
# SRC=32-tk-h100-matmul.cu
# SRC=33-tk-h100-tpmm.cu
# SRC=34-multimem-api.cu
# SRC=35-polished-tk-h100-tpmm.cu
# SRC=36-multimem-minimum.cu
# SRC=37-multimem-benchmarks.cu
SRC=38-multimem-limits.cu

##########################
# COMPILER CONFIGS
##########################

# Compiler
NVCC?=/usr/local/cuda-12/bin/nvcc

# Host C++ compiler options
NVCCFLAGS+=-std=c++20 -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing

# For profiling only
# NVCCFLAGS+=-Xcompiler=-finstrument-functions -Xcompiler=-pg -Xcompiler=-g
# NVCCFLAGS+=-Xcompiler=-pg
# NVCCFLAGS+=-pg

# Optimization (-O applies only to the host code)
NVCCFLAGS+=-O3 --use_fast_math

# Prod/Dev
NVCCFLAGS+=-DNDEBUG # prod
# NVCCFLAGS+=-DDEBUG # dev

# C++ MACROS
NVCCFLAGS+=-DKITTENS_HOPPER

# CUDA options
NVCCFLAGS+=-x cu
NVCCFLAGS+=-arch=sm_90a
NVCCFLAGS+=--expt-extended-lambda --expt-relaxed-constexpr 
NVCCFLAGS+=-Xptxas=--verbose -Xptxas=--warn-on-spills
NVCCFLAGS+=-Xnvlink=--verbose -forward-unknown-to-host-compiler

# Include paths, dynamic library, and forwarding
NVCCFLAGS+=-I./ThunderKittens/include -I./ThunderKittens/prototype # assumes ThunderKittens repo is in the same directory
NVCCFLAGS+=-lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lnvidia-ml
NVCCFLAGS+=-fopenmp -lgomp # faster CPU matmul (for benchmarking and verification)

# Dependency generation
NVCCFLAGS+=-MD -MT -MF

# Use official nccl installation
NVCCFLAGS+=-lnccl

# Use custom built static nccl
# NVCCFLAGS+=-L./15-nccl/build/lib -lnccl_static
# NVCCFLAGS+=-L./20-nccl/build/lib -lnccl_static

##########################
# PROFILER FLAGS
##########################

# UFTFLAGS+=--no-libcall
# UFTFLAGS+=--nest-libcall
# UFTFLAGG+=--kernel
UFTFLAGS+=--no-event
UFTFLAGS+=--max-stack=10
# UFTFLAGS+=--auto-args
UFTFLAGS+=--column-view

# UFTFLAGS+=--notrace=std::abs
# UFTFLAGS+=--notrace=rand
# UFTFLAGS+=--notrace=pthread_mutexattr_init
# UFTFLAGS+=--notrace=pthread_mutexattr_settype
# UFTFLAGS+=--notrace=pthread_mutexattr_setpshared
# UFTFLAGS+=--notrace=pthread_mutex_init
# UFTFLAGS+=--notrace=pthread_mutexattr_destroy
# UFTFLAGS+=--notrace=pthread_once

# UFTFLAGS+=--notrace=usleep
# UFTFLAGS+=--notrace=x64_sys_call

# UFTFLAGS+=--filter=ncclCommInitAll
# UFTFLAGS+=--filter=ncclGroupStart
# UFTFLAGS+=--filter=ncclAllReduce
# UFTFLAGS+=-F ncclRecv
# UFTFLAGS+=-F ncclSend
UFTFLAGS+=-F pncclGroupEnd
# UFTFLAGS+=--filter=ncclCommDestroy
# UFTFLAGS+=--filter=nccl*
# UFTFLAGS+=--filter=cudaMalloc

##########################
# TARGETS
##########################

# Target file
TARGET=$(basename $(SRC))

# Home diretory
#   Requirements:
#   - Thunderkittens repo in $(WORKING_DIR)/thunderkittens
#   - NCU in $(WORKING_DIR)/ncu
# WORKING_DIR=/home/rahul/stuart
WORKING_DIR=/scratch/stuart

# Default target
all: ./bin/$(TARGET)

# Run target
run: ./bin/$(TARGET)
	./bin/$(TARGET)

# Profile target and produce ncu report (likely requires sudo)
ncu: ./bin/$(TARGET)
	$(WORKING_DIR)/ncu/ncu \
		--replay-mode kernel \
		--set full -f --export ./profiler/$(TARGET).ncu-rep ./bin/$(TARGET)
	echo "ncu report saved to ./profiler/$(TARGET).ncu-rep"
# NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET \

# Profile target and produce nsys report (also using sudo highly recommended)
nsys: ./bin/$(TARGET)
	nsys profile \
		--stats=true \
		--trace cuda,osrt,nvtx,python-gil,syscall \
		--gpu-metrics-devices=all \
		--cuda-memory-usage true \
		--force-overwrite=true \
		-o ./profiler/$(TARGET).nsys-rep \
		./bin/$(TARGET)
	echo "nsys report saved to ./profiler/$(TARGET).nsys-rep and ./profiler/$(TARGET).sqlite"
	echo "run `nsys stats ./profiler/$(TARGET).sqlite` to view the stats"

# Check for out of bounds and misaligned memory accesses to global, local and shared mem
memcheck: ./bin/$(TARGET)
	compute-sanitizer --tool memcheck ./bin/$(TARGET)

uftrace: ./bin/$(TARGET)
	sudo uftrace $(UFTFLAGS) --no-pager ./bin/$(TARGET) > ./profiler/$(TARGET)-uftrace.log

# Build target & generate binary dump files
./bin/$(TARGET): $(SRC)
	mkdir -p ./bin
	$(NVCC) $(SRC) $(NVCCFLAGS) -o ./bin/$(TARGET)
# cuobjdump -ptx ./bin/$(TARGET) > ./bin/$(TARGET).ptx
# cuobjdump -elf ./bin/$(TARGET) > ./bin/$(TARGET).sass
# readelf -a ./bin/$(TARGET) > ./bin/$(TARGET).elfdump
# objdump -d ./bin/$(TARGET) > ./bin/$(TARGET).objdump

# Clean target
clean:
	rm -f ./bin/$(TARGET)
	rm -f ./bin/$(TARGET)*
	rm -f ./profiler/$(TARGET)*

# Force using system C++ library; needed for rcXX environments
setup:
	export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

.PHONY: setup
