#pragma once
#include <cuda_fp16.h>
#include <cooperative_groups.h>

/* ---- knobs ---------------------------------------------------- */
#ifndef HEAD_DIM
#define HEAD_DIM 128            /* multiple of 4 */
#endif
#ifndef TILE_TOKENS
#define TILE_TOKENS 32          /* override per .cu */
#endif
#ifndef WARPS_PER_CTA
#define WARPS_PER_CTA 4
#endif
#ifndef ROTARY_DIM
#define ROTARY_DIM 64
#endif

static_assert(HEAD_DIM % 4 == 0, "HEAD_DIM must be mult of 4");

namespace cg = cooperative_groups;

/* ---- cp.async helpers (8-byte) -------------------------------- */
#if (__CUDACC_VER_MAJOR__*10+__CUDACC_VER_MINOR__)>=114
#define USE_CP_ASYNC 1
__device__ __forceinline__ uint32_t sm_ofs(const void* p){
    return uint32_t(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ unsigned long long gm_addr(const void* p){
    return (unsigned long long)p;
}
#else
#define USE_CP_ASYNC 0
#endif

struct __align__(8) half4 { half x,y,z,w; };

__device__ __forceinline__
void rotary_half4(half4 &q, half4 &k, float s, float c){
    float2 q0{__half2float(q.x),__half2float(q.y)};
    float2 q1{__half2float(q.z),__half2float(q.w)};
    float2 k0{__half2float(k.x),__half2float(k.y)};
    float2 k1{__half2float(k.z),__half2float(k.w)};
    q0={q0.x*c-q0.y*s, q0.x*s+q0.y*c};
    q1={q1.x*c-q1.y*s, q1.x*s+q1.y*c};
    k0={k0.x*c-k0.y*s, k0.x*s+k0.y*c};
    k1={k1.x*c-k1.y*s, k1.x*s+k1.y*c};
    q={__float2half(q0.x),__float2half(q0.y),
       __float2half(q1.x),__float2half(q1.y)};
    k={__float2half(k0.x),__float2half(k0.y),
       __float2half(k1.x),__float2half(k1.y)};
}
