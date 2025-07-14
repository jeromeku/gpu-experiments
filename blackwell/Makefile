# Source files
# SRC := 00-sanity-check.cu
SRC := 01-mxfp8-mma.cu

# Compiler configuration
NVCC := nvcc

# ThunderKittens configuration
THUNDERKITTENS_ROOT := ../ThunderKittens

# Python binding configuration
# PYTHON_VERSION := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LDVERSION'))")
# PYBIND_INCLUDES := $(shell python3 -m pybind11 --includes)
# PYTORCH_INCLUDES := $(shell python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
# PYTHON_LIBDIR := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
# PYTORCH_LIBDIR := $(shell python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")
# EXT_SUFFIX := $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
# SO_FLAGS := -shared -fPIC

# GPU configuration
GPU := B200

# NVCC flags
NVCCFLAGS := -DNDEBUG -lineinfo
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr 
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing 
NVCCFLAGS += --use_fast_math -O3 # Note that --use_fast_math will harm precision
NVCCFLAGS += -forward-unknown-to-host-compiler 
NVCCFLAGS += -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas
NVCCFLAGS += -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills 

# ThunderKittens includes
# NVCCFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype

# NVCC flags for PyTorch binding
# NVCCFLAGS += $(SO_FLAGS)
# NVCCFLAGS += -ltorch -lc10_cuda -ltorch_cuda
# NVCCFLAGS += $(PYBIND_INCLUDES) -L${PYTHON_LIBDIR} -lpython${PYTHON_VERSION}
# NVCCFLAGS += $(PYTORCH_INCLUDES) ${PYTORCH_LIBDIR}
# NVCCFLAGS += -DTORCH_EXTENSION_NAME=_C # replicate setup.py behavior

# Conditional setup based on the target GPU
ifeq ($(GPU),H100)
	NVCCFLAGS+= -DKITTENS_HOPPER -gencode arch=compute_90a,code=sm_90a
else ifeq ($(GPU), B200)
	NVCCFLAGS+= -DKITTENS_HOPPER -DKITTENS_BLACKWELL -gencode arch=compute_100a,code=sm_100a
else
	$(error Unsupported GPU: $(GPU). Supported GPUs are H100 and B200)
endif

# Output target
OUT := $(basename $(SRC)).out
# OUT := $(basename $(SRC))$(EXT_SUFFIX)

all: $(OUT)

run: $(OUT)
	./$(OUT)

$(OUT): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(OUT)

clean:
	rm -f $(OUT)
