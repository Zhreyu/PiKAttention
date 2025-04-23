/**********************************************************************
*  Ultra-Bro – persistent attention kernel with async double-buffer
*  • TILE_TOKENS = 32   • HEAD_DIM = 128 (multiple-of-4 required)
*********************************************************************/
#define TILE_TOKENS 32
#include "pk_attn_common.h"             // our shared helpers (cp.async, rotary)

/* ------------------------------------------------------------------ */
template<int HD=HEAD_DIM,int TT=TILE_TOKENS,int RD=ROTARY_DIM>
__global__ void ultrabro_kernel(const half* __restrict__ Q,
                                const half* __restrict__ K,
                                const half* __restrict__ V,
                                half*       __restrict__ O,
                                int seq_len)
{
    extern __shared__ half sm[];
    /* double buffer: buf[0] and buf[1] – each has Q, K, V slices        */
    half *buf0 = sm;
    half *buf1 = buf0 + 3*TT*HD;
    bool swap  = false;

    int head = blockIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    const half *Qh = Q + (size_t)head*seq_len*HD;
    const half *Kh = K + (size_t)head*seq_len*HD;
    const half *Vh = V + (size_t)head*seq_len*HD;
          half *Oh = O + (size_t)head*seq_len*HD;

    cg::thread_block tb = cg::this_thread_block();
    const float scale = 1.f / sqrtf(float(HD));

    auto async_load = [&](half* dst, int base_tok){
        /* copy TT*HD Q/K/V (three streams) with 8-byte cp.async */
        for(int e = warp*32+lane; e < TT*HD; e += WARPS_PER_CTA*32){
            int tok =  e      / HD;
            int dim = (e & (HD-1));                  // faster mod
            if ((dim & 3) == 0){                     // 8-B aligned
                asm volatile(
                 "cp.async.ca.shared.global [%0],[%3],8;\n"
                 "cp.async.ca.shared.global [%1],[%4],8;\n"
                 "cp.async.ca.shared.global [%2],[%5],8;\n" ::
                 "r"(sm_ofs(dst + 0*TT*HD + tok*HD + dim)),
                 "r"(sm_ofs(dst + 1*TT*HD + tok*HD + dim)),
                 "r"(sm_ofs(dst + 2*TT*HD + tok*HD + dim)),
                 "l"(gm_addr(Qh + (base_tok+tok)*HD + dim)),
                 "l"(gm_addr(Kh + (base_tok+tok)*HD + dim)),
                 "l"(gm_addr(Vh + (base_tok+tok)*HD + dim)));
            }
        }
    };

    /* ---------- prefetch first tile ---------- */
    async_load(buf0, 0);
    cg::sync(tb); asm volatile("cp.async.commit_group; cp.async.wait_group 0;\n");

    for (int base = 0; base < seq_len; base += TT){
        half* cur = swap ? buf1 : buf0;
        half* nxt = swap ? buf0 : buf1;
        swap = !swap;

        /* overlap: prefetch next while computing current --------------- */
        if (base + TT < seq_len){
            async_load(nxt, base + TT);
        }

        half *q = cur;
        half *k = q + TT*HD;
        half *v = k + TT*HD;

        /* ---------- rotary (half4) ---------- */
        for(int tok = warp; tok < TT && (base+tok)<seq_len; tok += WARPS_PER_CTA){
            float θ0 = float(base+tok)*0.0001f;
#pragma unroll
            for(int dim = lane*4; dim < RD; dim += 128){
                rotary_half4(*(half4*)(q+tok*HD+dim),
                             *(half4*)(k+tok*HD+dim),
                             __sinf(θ0+dim), __cosf(θ0+dim));
            }
        }
        cg::sync(tb);

        /* ---------- dot-product with half2 ---------- */
        for(int qt = warp; qt < TT && (base+qt)<seq_len; qt += WARPS_PER_CTA){
            float acc = 0.f;
#pragma unroll
            for(int d = lane*2; d < HD; d += 64){
                half2 a = *reinterpret_cast<half2*>(q + qt*HD + d);
                half2 b = *reinterpret_cast<half2*>(k + qt*HD + d);
                half2 p = __hmul2(a,b);          // two products
                acc += __half2float(p.x) + __half2float(p.y);
            }
            for(int s = 16; s; s >>= 1) acc += __shfl_down_sync(0xffffffff,acc,s);

            float w = __expf(acc * scale);
            for(int d = lane; d < HD; d += 32)
                Oh[(base+qt)*HD+d] = __float2half(w * __half2float(v[qt*HD+d]));
        }
        cg::sync(tb);

#if USE_CP_ASYNC
        /* complete async copy of next tile */
        if (base + TT < seq_len){
            cg::sync(tb);
            asm volatile("cp.async.commit_group; cp.async.wait_group 0;\n");
        }
#endif
    }
}

/* ===== host stub =================================================== */
extern "C" void pk_attn(const void* Q,const void* K,const void* V,
                        void* O,int seq)
{
    dim3 grid(8), block(WARPS_PER_CTA*32);
    size_t smem = 2ULL * 3 * TILE_TOKENS * HEAD_DIM * sizeof(half); // double buf
    ultrabro_kernel<<<grid,block,smem>>>(
        static_cast<const half*>(Q),
        static_cast<const half*>(K),
        static_cast<const half*>(V),
        static_cast<half*>(O), seq);
}
