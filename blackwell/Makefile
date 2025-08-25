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
# SRC := 16-mxfp8-mma-polished.cu
# SRC := 17-mxfp8-mma-M256.cu
# SRC := 18-mxfp8-quantize.cu
# SRC := 19-smem-swizzle-8b.cu
# SRC := 20-smem-swizzle-16b.cu
# SRC := 21-mxfp8-quantize-opt.cu
# SRC := 22-mxfp8-quantize-opt-2.cu
# SRC := 23-mxfp8-mma-grid-ablation.cu
# SRC := 24-mxfp8-quantize-grid-ablation.cu
# SRC := 27-raw-binding.cu
# SRC := 28-tma-load.cu
# SRC := 29-tma-store.cu
# SRC := 30-tma-load-bandwidth.cu
# SRC := 31-tma-load-store-bandwidth.cu
# SRC := 32-tma-swiz-load.cu
# SRC := 33-tma-swiz-load-bandwidth.cu
# SRC := 34-dma-bandwidth.cu
# SRC := 35-tma-pipeline-bandwidth.cu
# SRC := 36-mxfp8-quantize-opt-3.cu
# SRC := 37-mxfp8-transpose-quantize.cu
# SRC := 38-mxfp8-quantize-combined.cu
# SRC := 39-fp8-mma-2cta.cu
# SRC := 40-fp8-mma-2cta-factored.cu
# SRC := 41-fp8-mma-2cta-M256.cu
# SRC := 42-fp8-mma-2cta-MN128.cu
# SRC := 43-fp8-mma-2cta-MNK128.cu
# SRC := 44-mxfp8-mm-2cta.cu
# SRC := 45-mxfp8-mma-2cta.cu
# SRC := 46-mxfp8-mma-2cta-MNK128.cu
# SRC := 47-mxfp8-mma-2cta-MNK128-polished.cu
# SRC := 48-bf16-matmul-shape-ablation.cu
# SRC := 49-bf16-matmul-launcher-ablation.cu
# SRC := 50-bf16-mha-D128.cu
# SRC := 51-bf16-mha-opt-D128.cu
SRC := 52-bf16-mha-deepseek.cu
# SRC := 53-microbenchmarks.cu

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
NVCCFLAGS += -shared -fPIC
NVCCFLAGS += $(PYBIND_INCLUDES) -L${PYTHON_LIBDIR} -lpython${PYTHON_VERSION}
OUT := _C$(EXT_SUFFIX)
RUN_CMD := python3 $(basename $(SRC)).py

# PyTorch binding configuration
# PYTORCH_INCLUDES := $(shell python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))")
# PYTORCH_LIBDIR := $(shell python3 -c "from torch.utils.cpp_extension import library_paths; print(' '.join(['-L' + p for p in library_paths()]))")
# NVCCFLAGS += $(PYTORCH_INCLUDES) ${PYTORCH_LIBDIR}
# NVCCFLAGS += -ltorch -lc10_cuda -ltorch_cuda
# NVCCFLAGS += -DTORCH_EXTENSION_NAME=_C # replicate setup.py behavior

# NVCC flags
NVCCFLAGS += -DNDEBUG -lineinfo
NVCCFLAGS += --expt-extended-lambda --expt-relaxed-constexpr 
NVCCFLAGS += -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing 
NVCCFLAGS += -forward-unknown-to-host-compiler --ftemplate-backtrace-limit=0
NVCCFLAGS += -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static
NVCCFLAGS += -O3 --use_fast_math # VERY important to include this flag
NVCCFLAGS += -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills 
NVCCFLAGS += -I${THUNDERKITTENS_ROOT}/include -I${THUNDERKITTENS_ROOT}/prototype
NVCCFLAGS += -DKITTENS_HOPPER -DKITTENS_BLACKWELL -gencode arch=compute_100a,code=sm_100a

# NVCC flags for compatibility with PyTorch extension
NVCCFLAGS += -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
NVCCFLAGS += -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
NVCCFLAGS += -DTORCH_API_INCLUDE_EXTENSION_H -DTORCH_EXTENSION_NAME=_C

all: $(OUT)

run: $(OUT)
	$(RUN_CMD)

$(OUT): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $(OUT)

clean:
	rm -f $(OUT)
