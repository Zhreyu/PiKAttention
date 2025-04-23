CUDA_ARCH ?= sm_86
NVCC = nvcc -std=c++17 -O3 -Xcompiler -fPIC -arch=$(CUDA_ARCH)

all: pk_attn_scalar.so pk_attn_opt.so pk_attn_mma.so pk_attn_dp4a.so pk_attn_bsr.so 

pk_attn_scalar.so: kernels/pk_attn_scalar.cu kernels/pk_attn_common.h
	$(NVCC) -shared -o $@ $<
pk_attn_opt.so:    kernels/pk_attn_opt.cu    kernels/pk_attn_common.h
	$(NVCC) -shared -o $@ $<
pk_attn_mma.so:    kernels/pk_attn_mma.cu    kernels/pk_attn_common.h
	$(NVCC) -shared -o $@ $<
pk_attn_dp4a.so:   kernels/pk_attn_dp4a.cu   kernels/pk_attn_common.h
	$(NVCC) -shared -o $@ $<
pk_attn_bsr.so: kernels/pk_attn_bsr.cu kernels/pk_attn_common.h
	$(NVCC) -DUSE_TENSOR_CORE=1 -shared -o $@ $<
# pk_attn_mma8.so: kernels/pk_attn_mma8x8.cu kernels/pk_attn_common.h
# 	$(NVCC) -shared -o $@ $<
clean: ; rm -f pk_attn_*.so pk_results.csv
