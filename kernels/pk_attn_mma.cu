/**********************************************************************
*  pk_attn_mma.cu – persistent kernel, Stage-3 uses WMMA m16n16k16
*********************************************************************/
#include "pk_attn_common.h"
#define TILE_TOKENS 32    // same tile size as Opt

/* ─── Host sees only an empty stub (prevents template-related noise) */
#if !defined(__CUDA_ARCH__)
extern "C" __global__
void pk_mma_kernel(const half*,const half*,const half*,half*,int){}

#else  /* ================= DEVICE PASS ============================ */

#include <mma.h>
using namespace nvcuda;
using namespace nvcuda::wmma;

/* warp-reduce utility */
__device__ __forceinline__ float warpReduce(float x){
    for(int s=16;s;s>>=1) x += __shfl_down_sync(0xffffffff,x,s);
    return x;
}

/* -------- real device kernel ------------------------------------ */
extern "C" __global__
void pk_mma_kernel(const half* __restrict__ Q,
                   const half* __restrict__ K,
                   const half* __restrict__ V,
                   half*       __restrict__ O,
                   int seq)
{
    constexpr int TT = TILE_TOKENS;
    constexpr int HD = HEAD_DIM;
    constexpr int RD = ROTARY_DIM;

    extern __shared__ half sm[];
    half *q = sm, *k = q+TT*HD, *v = k+TT*HD;

    int head = blockIdx.x,
        lane = threadIdx.x & 31,
        warp = threadIdx.x >> 5;

    const half *Qh = Q + (size_t)head*seq*HD;
    const half *Kh = K + (size_t)head*seq*HD;
    const half *Vh = V + (size_t)head*seq*HD;
          half *Oh = O + (size_t)head*seq*HD;

    cg::thread_block tb = cg::this_thread_block();
    const float scale = 1.f / sqrtf(float(HD));

    for(int base=0;base<seq;base+=TT){
        /* Stage-1 : cp.async tri-stream */
        for(int e=warp*32+lane; e<TT*HD; e+=WARPS_PER_CTA*32){
            int t=e/HD,d=e%HD;
            if((d&3)==0){
                asm volatile(
                 "cp.async.ca.shared.global [%0],[%3],8;\n"
                 "cp.async.ca.shared.global [%1],[%4],8;\n"
                 "cp.async.ca.shared.global [%2],[%5],8;\n" ::
                 "r"(sm_ofs(q+t*HD+d)),
                 "r"(sm_ofs(k+t*HD+d)),
                 "r"(sm_ofs(v+t*HD+d)),
                 "l"(gm_addr(Qh+(base+t)*HD+d)),
                 "l"(gm_addr(Kh+(base+t)*HD+d)),
                 "l"(gm_addr(Vh+(base+t)*HD+d)));
            }
        }
        cg::sync(tb); asm volatile("cp.async.commit_group; cp.async.wait_group 0;\n");

        /* Stage-2 : rotary */
        for(int tok=warp; tok<TT; tok+=WARPS_PER_CTA){
            float th=float(base+tok)*0.0001f;
#pragma unroll
            for(int d=lane*4; d<RD; d+=128)
                rotary_half4(*(half4*)(q+tok*HD+d),
                             *(half4*)(k+tok*HD+d),
                             __sinf(th+d), __cosf(th+d));
        }
        cg::sync(tb);

        /* Stage-3 : WMMA 16×16×16  F16×F16→F32 */
        for(int qt=warp; qt<TT; qt+=WARPS_PER_CTA){
            float acc=0.f;
            for(int col=0; col<HD; col+=16){
                fragment<matrix_a,16,16,16,half,row_major>  fa;
                fragment<matrix_b,16,16,16,half,row_major>  fb;
                fragment<accumulator,16,16,16,float>        fc;
                fill_fragment(fc,0.0f);
                load_matrix_sync(fa,q+qt*HD+col,HD);
                load_matrix_sync(fb,k+qt*HD+col,HD);
                mma_sync(fc,fa,fb,fc);
#pragma unroll
                for(int i=0;i<fc.num_elements;i++) acc+=fc.x[i];
            }
            acc = warpReduce(acc);
            float w = __expf(acc*scale);
            for(int d=lane; d<HD; d+=32)
                Oh[(base+qt)*HD+d] =
                    __float2half(w * __half2float(v[qt*HD+d]));
        }
        cg::sync(tb);
    }
}
#endif  /* __CUDA_ARCH__ */

/* host launcher */
extern "C" void pk_attn(const void*Q,const void*K,const void*V,
                        void*O,int seq)
{
    dim3 grid(8), block(WARPS_PER_CTA*32);
    size_t smem = 3ULL*TILE_TOKENS*HEAD_DIM*sizeof(half);
    pk_mma_kernel<<<grid,block,smem>>>(
        static_cast<const half*>(Q),
        static_cast<const half*>(K),
        static_cast<const half*>(V),
        static_cast<half*>(O), seq);
}
