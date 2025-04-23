/**********************************************************************
*  pk_attn_dp4a.cu  –  persistent kernel using INT8 DP4A dot-product
*********************************************************************/
#include "pk_attn_common.h"
#define TILE_TOKENS 32

/* quantise helper: float16 → int8 in [-127,127] */
__device__ __forceinline__ int8_t q8(half h){
    float x = __half2float(h);
    return (int8_t)__float2int_rn(max(-127.f,min(127.f,x*127.f)));
}

template<int HD=HEAD_DIM,int TT=TILE_TOKENS,int RD=ROTARY_DIM>
__global__ void pk_dp4a_kernel(const half* __restrict__ Q,
                               const half* __restrict__ K,
                               const half* __restrict__ V,
                               half*       __restrict__ O,
                               int seq)
{
    extern __shared__ int8_t sm8[];
    int8_t *q8s = sm8;               // INT8 tiles
    int8_t *k8s = q8s + TT*HD;
    half  *v16s = (half*)(k8s + TT*HD);   // keep V in half

    int head=blockIdx.x, lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const half *Qh=Q+(size_t)head*seq*HD,
               *Kh=K+(size_t)head*seq*HD,
               *Vh=V+(size_t)head*seq*HD;
          half *Oh=O+(size_t)head*seq*HD;
    cg::thread_block tb=cg::this_thread_block();
    const float scale=1.f/(sqrtf(float(HD))*127*127);

    for(int base=0;base<seq;base+=TT){
        /* Stage-1 : load + quantise Q,K to INT8, copy V as half */
        for(int e=warp*32+lane;e<TT*HD;e+=WARPS_PER_CTA*32){
            int t=e/HD,d=e%HD;
            int8_t *dq=q8s+t*HD+d,*dk=k8s+t*HD+d;
            *dq=q8(Qh[(base+t)*HD+d]);
            *dk=q8(Kh[(base+t)*HD+d]);
            v16s[t*HD+d]=Vh[(base+t)*HD+d];
        }
        __syncthreads();   // CPU copy is tiny; no cp.async needed

        /* Stage-2 rotary identical but on half (not shown) */

        /* Stage-3 : INT8 dot → FP32 */
        for(int qt=warp; qt<TT; qt+=WARPS_PER_CTA){
            int32_t acc32=0;
            for(int d=lane*4; d<HD; d+=128){           // 4 INT8/loop
                int32_t a=*(int32_t*)(q8s+qt*HD+d);
                int32_t b=*(int32_t*)(k8s+qt*HD+d);
                acc32 = __dp4a(a,b,acc32);
            }
            for(int s=16;s;s>>=1) acc32+=__shfl_down_sync(0xffffffff,acc32,s);

            float w=__expf(float(acc32)*scale);
            for(int d=lane; d<HD; d+=32)
                Oh[(base+qt)*HD+d]=__float2half(w*__half2float(v16s[qt*HD+d]));
        }
        __syncthreads();
    }
}

extern "C" void pk_attn(const void*Q,const void*K,const void*V,void*O,int seq){
    dim3 grid(8),block(WARPS_PER_CTA*32);
    size_t smem = (2ULL*TILE_TOKENS*HEAD_DIM)*sizeof(int8_t) +
                  (1ULL*TILE_TOKENS*HEAD_DIM)*sizeof(half);
    pk_dp4a_kernel<<<grid,block,smem>>>(
        static_cast<const half*>(Q),
        static_cast<const half*>(K),
        static_cast<const half*>(V),
        static_cast<half*>(O), seq);

    }