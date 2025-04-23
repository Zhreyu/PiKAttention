#define TILE_TOKENS 16
#include "pk_attn_common.h"

/* --- plain F16 scalar dot -------------------------------------- */
template<int HD=HEAD_DIM,int TT=TILE_TOKENS,int RD=ROTARY_DIM>
__global__ void pk_scalar(const half* __restrict__ Q,
                          const half* __restrict__ K,
                          const half* __restrict__ V,
                          half*       __restrict__ O,
                          int seq)
{
    extern __shared__ half sm[];
    half *q=sm, *k=q+TT*HD, *v=k+TT*HD;
    int head=blockIdx.x, lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const half *Qh=Q+(size_t)head*seq*HD,
               *Kh=K+(size_t)head*seq*HD,
               *Vh=V+(size_t)head*seq*HD;
          half *Oh=O+(size_t)head*seq*HD;
    float scale=1.f/sqrtf(float(HD));
    cg::thread_block tb=cg::this_thread_block();

    for(int base=0;base<seq;base+=TT){
        for(int e=warp*32+lane;e<TT*HD;e+=WARPS_PER_CTA*32){
            int t=e/HD,d=e%HD;
#if USE_CP_ASYNC
            if((d&3)==0){
                asm volatile(
                 "cp.async.ca.shared.global [%0],[%3],8;\n"
                 "cp.async.ca.shared.global [%1],[%4],8;\n"
                 "cp.async.ca.shared.global [%2],[%5],8;\n"::
                 "r"(sm_ofs(q+t*HD+d)),"r"(sm_ofs(k+t*HD+d)),"r"(sm_ofs(v+t*HD+d)),
                 "l"(gm_addr(Qh+(base+t)*HD+d)),
                 "l"(gm_addr(Kh+(base+t)*HD+d)),
                 "l"(gm_addr(Vh+(base+t)*HD+d)));
            }
#else
            q[t*HD+d]=Qh[(base+t)*HD+d];
            k[t*HD+d]=Kh[(base+t)*HD+d];
            v[t*HD+d]=Vh[(base+t)*HD+d];
#endif
        }
#if USE_CP_ASYNC
        cg::sync(tb); asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n");
#else
        __syncthreads();
#endif
        for(int t=warp; t<TT&&(base+t)<seq; t+=WARPS_PER_CTA){
            float θ=float(base+t)*0.0001f;
            for(int d=lane*4; d<RD; d+=128)
                rotary_half4(*(half4*)(q+t*HD+d),
                             *(half4*)(k+t*HD+d),
                             __sinf(θ+d),__cosf(θ+d));
        }
        cg::sync(tb);
        for(int qt=warp; qt<TT&&(base+qt)<seq; qt+=WARPS_PER_CTA){
            float acc=0.f;
            for(int d=lane; d<HD; d+=32)
                acc+=__half2float(q[qt*HD+d])*__half2float(k[qt*HD+d]);
            for(int sh=16;sh;sh>>=1) acc+=__shfl_down_sync(0xffffffff,acc,sh);
            float w=__expf(acc*scale);
            for(int d=lane; d<HD; d+=32)
                Oh[(base+qt)*HD+d]=__float2half(w*__half2float(v[qt*HD+d]));
        }
        cg::sync(tb);
    }
}

extern "C" void pk_attn(const void*Q,const void*K,const void*V,void*O,int seq){
    dim3 grid(8),block(WARPS_PER_CTA*32);
    size_t smem=3ULL*TILE_TOKENS*HEAD_DIM*sizeof(half);
    pk_scalar<<<grid,block,smem>>>((const half*)Q,(const half*)K,(const half*)V,
                                   (half*)O,seq);
}
