/********************************************************************/
/*   kernels/pk_attn_tc_wt.cu   —  device-only Tensor-Core kernel   */
/********************************************************************/
#define TILE_TOKENS 16
#define USE_TENSOR_CORE 1
#include "pk_attn_common.h"        // rotary + cp.async helpers

/* ────────── host sees only this empty stub ────────── */
#if !defined(__CUDA_ARCH__)
extern "C" __global__
void tc_warp_kernel(const half*,const half*,const half*,half*,int){}

/* ────────── device pass gets the full body ────────── */
#else
#include <cuda_fp16.h>  
#include <mma.h>
using namespace nvcuda;
using namespace nvcuda::wmma;

extern "C" __global__
void tc_warp_kernel(const half* __restrict__ Q,
                    const half* __restrict__ K,
                    const half* __restrict__ V,
                    half*       __restrict__ O,
                    int seq_len)
{
    constexpr int TT = TILE_TOKENS;
    constexpr int HD = HEAD_DIM;
    constexpr int RD = ROTARY_DIM;

    extern __shared__ half sm[];
    half *q = sm, *k = q + TT*HD, *v = k + TT*HD;

    int head = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const half *Qh = Q + (size_t)head*seq_len*HD;
    const half *Kh = K + (size_t)head*seq_len*HD;
    const half *Vh = V + (size_t)head*seq_len*HD;
          half *Oh = O + (size_t)head*seq_len*HD;

    cg::thread_block tb = cg::this_thread_block();
    const float scale = 1.f / sqrtf(float(HD));

    for (int base=0;base<seq_len;base+=TT) {
        /* cp.async tri-stream (8 B) */
        for (int e=warp*32+lane;e<TT*HD;e+=WARPS_PER_CTA*32){
            int t=e/HD,d=e%HD;
            if((d&3)==0){
                asm volatile(
                 "cp.async.ca.shared.global [%0],[%3],8;\n"
                 "cp.async.ca.shared.global [%1],[%4],8;\n"
                 "cp.async.ca.shared.global [%2],[%5],8;\n"::
                 "r"(sm_ofs(q+t*HD+d)),
                 "r"(sm_ofs(k+t*HD+d)),
                 "r"(sm_ofs(v+t*HD+d)),
                 "l"(gm_addr(Qh+(base+t)*HD+d)),
                 "l"(gm_addr(Kh+(base+t)*HD+d)),
                 "l"(gm_addr(Vh+(base+t)*HD+d)));
            }
        }
        cg::sync(tb); asm volatile("cp.async.commit_group; cp.async.wait_group 0;\n");

        /* rotary on first RD dims */
        for (int tok=warp; tok<TT; tok+=WARPS_PER_CTA){
            float th=float(base+tok)*0.0001f;
#pragma unroll
            for(int d=lane*4; d<RD; d+=128)
                rotary_half4(*(half4*)(q+tok*HD+d),
                             *(half4*)(k+tok*HD+d),
                             __sinf(th+d), __cosf(th+d));
        }
        cg::sync(tb);

        /* WMMA dot-product 16×8×16 */
        for (int qt=warp; qt<TT; qt+=WARPS_PER_CTA){
            float acc=0.f;
            for (int col=0; col<HD; col+=16){
                fragment<matrix_a,16,8,16,half,row_major> fa;
                fragment<matrix_b,16,8,16,half,row_major> fb;
                fragment<accumulator,16,8,16,float>      fc;
                fill_fragment(fc,0.f);
                load_matrix_sync(fa,q+qt*HD+col,HD);
                load_matrix_sync(fb,k+qt*HD+col,HD);
                mma_sync(fc,fa,fb,fc);
#pragma unroll
                for(int i=0;i<fc.num_elements;i++) acc+=fc.x[i];
            }
            for(int s=16;s;s>>=1) acc+=__shfl_down_sync(0xffffffff,acc,s);
            float w=__expf(acc*scale);
            for(int d=lane; d<HD; d+=32)
                Oh[(base+qt)*HD+d]=__float2half(w*__half2float(v[qt*HD+d]));
        }
        cg::sync(tb);
    }
}
#endif  /* __CUDA_ARCH__ */

/* ---------- host-side launcher ---------- */
extern "C" void pk_attn(const void*Q,const void*K,const void*V,
                        void*O,int seq){
    dim3 grid(8), block(WARPS_PER_CTA*32);
    constexpr size_t SMEM = 3ULL*TILE_TOKENS*HEAD_DIM*sizeof(half);
    tc_warp_kernel<<<grid,block,SMEM>>>(
        static_cast<const half*>(Q),
        static_cast<const half*>(K),
        static_cast<const half*>(V),
        static_cast<half*>(O), seq);
}
