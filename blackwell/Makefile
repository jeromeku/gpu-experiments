# Source files
# SRC := 00-sanity-check.cu
# SRC := 01-raw-fp8-mm.cu
# SRC := 02-fp8-mm.cu
# SRC := 03-half-raw-fp8-mm.cu
# SRC := 04-tm-ldst-1d-int32.cu
# SRC := 05-tm-alloc.cu
# SRC := 06-tm-ldst-2d-int32.cu
# SRC := 07-tm-cp.cu
# SRC := 08-tm-cp-fp8.cu
# SRC := 09-mxfp8-mm.cu
# SRC := 10-mxfp8-mma.cu
# SRC := 11-fp8-mma-wg-ablation.cu
# SRC := 12-fp8-mma-N-ablation.cu
# SRC := 13-fp8-mma-M-ablation.cu
# SRC := 14-fp8-mma-M-ablation-2.cu
# SRC := 15-fp8-mma-M-ablation-3.cu
SRC := 16-mxfp8-mma-polished.cu

# Compiler configuration
NVCC := nvcc

# ThunderKittens configuration
THUNDERKITTENS_ROOT := ../ThunderKittens

# Standalone configuration
# OUT := $(basename $(SRC)).out
# RUN_CMD := ./$(OUT)

# Python binding configuration
PYTHON_VERSION := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
PYTHON_LIBDIR := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYBIND_INCLUDES := $(shell python3 -m pybind11 --includes)
EXT_SUFFIX := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SO_FLAGS := -shared -fPIC
NVCCFLAGS += $(SO_FLAGS)
NVCCFLAGS += $(PYBIND_INCLUDES) -L${PYTHON_LIBDIR} -lpython${PYTHON_VERSION}
OUT := _C$(EXT_SUFFIX)
RUN_CMD := python3 $(basename $(SRC)).py

# PyTorch binding configuration
# PYTORCH_INCLUDES := $(shell python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
# PYTORCH_LIBDIR := $(shell python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")
# NVCCFLAGS += $(PYTORCH_INCLUDES) ${PYTORCH_LIBDIR}
# NVCCFLAGS += -ltorch -lc10_cuda -ltorch_cuda
# NVCCFLAGS += -DTORCH_EXTENSION_NAME=_C # replicate setup.py behavior

# GPU configuration
GPU := B200

# NVCC flags
NVCCFLAGS += -DNDEBUG -lineinfo
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr 
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing 
NVCCFLAGS += --use_fast_math -O3 # Note that --use_fast_math will harm precision
NVCCFLAGS += -forward-unknown-to-host-compiler 
NVCCFLAGS += -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas
NVCCFLAGS += -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills 
NVCCFLAGS += -Xcudafe "--diag_suppress=68"

# ThunderKittens includes
NVCCFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype

# Conditional setup based on the target GPU
ifeq ($(GPU),H100)
	NVCCFLAGS+= -DKITTENS_HOPPER -gencode arch=compute_90a,code=sm_90a
else ifeq ($(GPU), B200)
	NVCCFLAGS+= -DKITTENS_HOPPER -DKITTENS_BLACKWELL -gencode arch=compute_100a,code=sm_100a
else
	$(error Unsupported GPU: $(GPU). Supported GPUs are H100 and B200)
endif

all: $(OUT)

run: $(OUT)
	$(RUN_CMD)

$(OUT): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(OUT)

clean:
	rm -f $(OUT)
