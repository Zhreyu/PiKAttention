//═══════════════════════════════════════════════════════════════════════════════
//  Persistent-Kernel Attention  (RTX-3060 / GA10x safe)
//
//  Build (async): nvcc -std=c++17 -arch=sm_86 -O3 -lineinfo pk_attention.cu -o pk_attn
//  Build (sync) : nvcc -std=c++17 -arch=sm_86 -O3 -DNO_CP_ASYNC pk_attention.cu -o pk_attn
//
//═══════════════════════════════════════════════════════════════════════════════
#include <cuda.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <random>
#include <vector>
#include <cstring>

namespace cg = cooperative_groups;
#define CK(x) do{ cudaError_t _e=(x); if(_e){                                 \
    fprintf(stderr,"CUDA %s:%d  %s\n",__FILE__,__LINE__,cudaGetErrorString(_e));\
    exit(1);} }while(0)

//──────────────────────── tunables ────────────────────────
#ifndef HEAD_DIM
#define HEAD_DIM 128   // multiple of 4
#endif

#ifndef TILE_TOKENS
#define TILE_TOKENS 32
#endif

#ifndef WARPS_PER_CTA
#define WARPS_PER_CTA 4
#endif

#ifndef ROTARY_DIM
#define ROTARY_DIM 64
#endif

static_assert(HEAD_DIM % 4 == 0, "HEAD_DIM must be a multiple of 4");

//──────────────────────── cp.async helpers ────────────────────────
#if !defined(NO_CP_ASYNC) && (__CUDACC_VER_MAJOR__*10+__CUDACC_VER_MINOR__) >= 114
#define USE_CP_ASYNC 1
__device__ __forceinline__ uint32_t sm_ofs(const void* p){
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));      // 32-bit SMEM offset
}
__device__ __forceinline__ unsigned long long gm_addr(const void* p){
    return reinterpret_cast<unsigned long long>(p);                 // 64-bit global addr
}
#else
#define USE_CP_ASYNC 0
#endif

//──────────────────────── aligned half4 helper ────────────────────────
struct __align__(8) half4 { half x,y,z,w; };
__device__ __host__ __forceinline__ half4 make_half4(half a,half b,half c,half d){
    half4 h{a,b,c,d}; return h;
}

// rotate two complex numbers (x0+ix1 , x2+ix3) by θ
__device__ __forceinline__
void rotary_half4(half4 &q, half4 &k, float s, float c){
    float2 q0 = make_float2(__half2float(q.x), __half2float(q.y));
    float2 q1 = make_float2(__half2float(q.z), __half2float(q.w));
    float2 k0 = make_float2(__half2float(k.x), __half2float(k.y));
    float2 k1 = make_float2(__half2float(k.z), __half2float(k.w));

    q0 = make_float2(q0.x*c - q0.y*s, q0.x*s + q0.y*c);
    q1 = make_float2(q1.x*c - q1.y*s, q1.x*s + q1.y*c);
    k0 = make_float2(k0.x*c - k0.y*s, k0.x*s + k0.y*c);
    k1 = make_float2(k1.x*c - k1.y*s, k1.x*s + k1.y*c);

    q = make_half4(__float2half(q0.x), __float2half(q0.y),
                   __float2half(q1.x), __float2half(q1.y));
    k = make_half4(__float2half(k0.x), __float2half(k0.y),
                   __float2half(k1.x), __float2half(k1.y));
}

//═══════════════════════════════════════════════════════════════════════════════
//                                 KERNEL
//═══════════════════════════════════════════════════════════════════════════════
template<int HD=HEAD_DIM,int TT=TILE_TOKENS,int RD=ROTARY_DIM>
__global__ void pk_attn(const half* __restrict__ Q,
                        const half* __restrict__ K,
                        const half* __restrict__ V,
                        half*       __restrict__ O,
                        int seq)
{
    extern __shared__ half sm[];
    half *q = sm;              // [TT][HD]
    half *k = q + TT*HD;
    half *v = k + TT*HD;

    int head = blockIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    const half *Qh = Q + (size_t)head*seq*HD;
    const half *Kh = K + (size_t)head*seq*HD;
    const half *Vh = V + (size_t)head*seq*HD;
          half *Oh = O + (size_t)head*seq*HD;

    cg::thread_block tb = cg::this_thread_block();
    const float scale = 1.f / sqrtf((float)HD);

    for (int base=0; base<seq; base+=TT){

        //── Stage 1: load Q/K/V (8-byte cp.async or fallback) ──
        int elems = TT*HD;
        for (int e = warp*32 + lane; e < elems; e += WARPS_PER_CTA*32){
            int tok = e / HD, dim = e % HD;

            const half *sQ = Qh + (base+tok)*HD + dim;
            const half *sK = Kh + (base+tok)*HD + dim;
            const half *sV = Vh + (base+tok)*HD + dim;
            half *dQ = q + tok*HD + dim;
            half *dK = k + tok*HD + dim;
            half *dV = v + tok*HD + dim;

#if USE_CP_ASYNC
            if ((dim & 3) == 0){                               // 8-byte aligned
                asm volatile(
                  "cp.async.ca.shared.global [%0], [%3], 8;\n"
                  "cp.async.ca.shared.global [%1], [%4], 8;\n"
                  "cp.async.ca.shared.global [%2], [%5], 8;\n" ::
                  "r"(sm_ofs(dQ)), "r"(sm_ofs(dK)), "r"(sm_ofs(dV)),
                  "l"(gm_addr(sQ)), "l"(gm_addr(sK)), "l"(gm_addr(sV)));
            }
#else
            *dQ = *sQ;  *dK = *sK;  *dV = *sV;
#endif
        }
#if USE_CP_ASYNC
        cg::sync(tb); asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n");
#else
        __syncthreads();
#endif

        //── Stage 2: rotary embedding (operate on aligned half4) ──
        for (int tok = warp; tok < TT && (base+tok)<seq; tok += WARPS_PER_CTA){
            float θ_base = (float)(base+tok) * 0.0001f;
#pragma unroll
            for (int dim = lane*4; dim < RD; dim += 128){      // stride 128 = 4*32
                half4 *q4 = reinterpret_cast<half4*>(q + tok*HD + dim);
                half4 *k4 = reinterpret_cast<half4*>(k + tok*HD + dim);
                float s = __sinf(θ_base + dim);
                float c = __cosf(θ_base + dim);
                half4 qv = *q4, kv = *k4;
                rotary_half4(qv, kv, s, c);
                *q4 = qv; *k4 = kv;
            }
        }
        cg::sync(tb);

        //── Stage 3: dot-product attention ──
        for (int qt = warp; qt < TT && (base+qt)<seq; qt += WARPS_PER_CTA){
            float acc = 0.f;
#pragma unroll
            for (int d = lane; d < HD; d += 32)
                acc += __half2float(q[qt*HD+d]) * __half2float(k[qt*HD+d]);
#pragma unroll
            for (int o = 16; o; o >>= 1)
                acc += __shfl_down_sync(0xffffffff, acc, o);

            float w = __expf(acc * scale);
            for (int d = lane; d < HD; d += 32)
                Oh[(base+qt)*HD + d] = __float2half(w * __half2float(v[qt*HD+d]));
        }
        cg::sync(tb);
    }
}

//═══════════════════════════════════════════════════════════════════════════════
//                                   HOST
//═══════════════════════════════════════════════════════════════════════════════
struct Opt { int seq=2048, heads=8, iters=100; };

int main(int argc,char**argv)
{
    Opt o;
    for (int i=1;i<argc;++i){
        if (!strcmp(argv[i],"--seq"))   o.seq   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--heads")) o.heads = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--iters")) o.iters = atoi(argv[++i]);
    }

    size_t elems  = (size_t)o.seq * o.heads * HEAD_DIM;
    size_t bytes  = elems * sizeof(half);

    half *dQ,*dK,*dV,*dO;
    CK(cudaMalloc(&dQ,bytes));
    CK(cudaMalloc(&dK,bytes));
    CK(cudaMalloc(&dV,bytes));
    CK(cudaMalloc(&dO,bytes));

    std::vector<half> h(elems);
    std::mt19937 rng(0); std::uniform_real_distribution<float> dist(-1,1);
    for (auto &x: h) x = __float2half(dist(rng));
    CK(cudaMemcpy(dQ,h.data(),bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK,h.data(),bytes,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV,h.data(),bytes,cudaMemcpyHostToDevice));

    dim3 grid(o.heads);
    dim3 block(WARPS_PER_CTA * 32);
    size_t smem = 3ULL * TILE_TOKENS * HEAD_DIM * sizeof(half);

    printf("PK-Attn  seq=%d  heads=%d  cp.async=%s  SMEM=%.1f KB\n",
           o.seq, o.heads, USE_CP_ASYNC?"on":"off", smem/1024.0);

    cudaEvent_t t0,t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    CK(cudaEventRecord(t0));
    for (int i=0;i<o.iters;++i)
        pk_attn<<<grid,block,smem>>>(dQ,dK,dV,dO,o.seq);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));

    float ms; CK(cudaEventElapsedTime(&ms,t0,t1));
    printf("Elapsed %.3f ms  (%.2f µs / token / head)\n",
           ms, 1000.f*ms/(o.iters*o.seq*o.heads));

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    return 0;
}
