#define TILE_TOKENS 32
#include "pk_attn_common.h"

/* --- half2-vectorised & ping-pong cp.async ---------------------- */
template<int HD=HEAD_DIM,int TT=TILE_TOKENS,int RD=ROTARY_DIM>
__global__ void pk_opt(const half* __restrict__ Q,
                       const half* __restrict__ K,
                       const half* __restrict__ V,
                       half*       __restrict__ O,
                       int seq)
{
    extern __shared__ half sm[];
    half *buf0=sm, *buf1=buf0+TT*HD*3;  /* double buffer */
    bool toggle=false;
    int head=blockIdx.x, lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const half *Qh=Q+(size_t)head*seq*HD,
               *Kh=K+(size_t)head*seq*HD,
               *Vh=V+(size_t)head*seq*HD;
          half *Oh=O+(size_t)head*seq*HD;
    float scale=1.f/sqrtf(float(HD));
    cg::thread_block tb=cg::this_thread_block();

    auto loader=[&](half*dst,int base){
        for(int e=warp*32+lane;e<TT*HD;e+=WARPS_PER_CTA*32){
            int t=e/HD,d=e%HD;
            const half *sQ=Qh+(base+t)*HD+d;
            const half *sK=Kh+(base+t)*HD+d;
            const half *sV=Vh+(base+t)*HD+d;
#if USE_CP_ASYNC
            if((d&3)==0){
                asm volatile(
                 "cp.async.ca.shared.global [%0],[%3],8;\n"
                 "cp.async.ca.shared.global [%1],[%4],8;\n"
                 "cp.async.ca.shared.global [%2],[%5],8;\n"::
                 "r"(sm_ofs(dst+0*TT*HD+t*HD+d)),
                 "r"(sm_ofs(dst+1*TT*HD+t*HD+d)),
                 "r"(sm_ofs(dst+2*TT*HD+t*HD+d)),
                 "l"(gm_addr(sQ)),"l"(gm_addr(sK)),"l"(gm_addr(sV)));
            }
#else
            dst[0*TT*HD+t*HD+d]=*sQ;
            dst[1*TT*HD+t*HD+d]=*sK;
            dst[2*TT*HD+t*HD+d]=*sV;
#endif
        }
    };

    /* prefetch first tile */
    loader(buf0,0);
#if USE_CP_ASYNC
    cg::sync(tb); asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n");
#else
    __syncthreads();
#endif

    for(int base=0;base<seq;base+=TT){
        half *cur = toggle?buf1:buf0;
        half *nxt = toggle?buf0:buf1;
        toggle=!toggle;
        /* preload next tile while computing current */
        if(base+TT<seq) loader(nxt,base+TT);
        cg::sync(tb);
#if USE_CP_ASYNC
        if(base+TT<seq){
            cg::sync(tb); asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n");
        }
#endif
        half *q=cur, *k=q+TT*HD, *v=k+TT*HD;
        /* rotary */
        for(int t=warp; t<TT&&(base+t)<seq; t+=WARPS_PER_CTA){
            float θ=float(base+t)*0.0001f;
#pragma unroll
            for(int d=lane*4; d<RD; d+=128)
                rotary_half4(*(half4*)(q+t*HD+d),
                             *(half4*)(k+t*HD+d),
                             __sinf(θ+d),__cosf(θ+d));
        }
        cg::sync(tb);
        /* dot-product with half2 */
        for(int qt=warp; qt<TT&&(base+qt)<seq; qt+=WARPS_PER_CTA){
            float acc=0.f;
            for(int d=lane*2; d<HD; d+=64){
                half2 a=*reinterpret_cast<half2*>(q+qt*HD+d);
                half2 b=*reinterpret_cast<half2*>(k+qt*HD+d);
                acc += __low2float(__hmul2(a,b)) +
                       __high2float(__hmul2(a,b));
            }
            for(int s=16;s;s>>=1) acc+=__shfl_down_sync(0xffffffff,acc,s);
            float w=__expf(acc*scale);
            for(int d=lane; d<HD; d+=32)
                Oh[(base+qt)*HD+d]=__float2half(w*__half2float(v[qt*HD+d]));
        }
        cg::sync(tb);
    }
}

extern "C" void pk_attn(const void*Q,const void*K,const void*V,void*O,int seq){
    dim3 grid(8),block(WARPS_PER_CTA*32);
    size_t smem=2ULL*3*HEAD_DIM*TILE_TOKENS*sizeof(half); /* double buf */
    pk_opt<<<grid,block,smem>>>((const half*)Q,(const half*)K,(const half*)V,
                                (half*)O,seq);
}
