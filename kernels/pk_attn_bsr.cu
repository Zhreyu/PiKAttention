/**********************************************************************
 *  pk_attn_bsr.cu  ── block-sparse PK-Attention (BSR schedule)
 *********************************************************************/

 #define TILE_TOKENS 16          // 16-token tiles (one row per warp)
 #define USE_TENSOR_CORE 1       // pull in WMMA helpers
 #include "pk_attn_common.h"     // cp.async, rotary, etc.
 
 #include <mma.h>
 using namespace nvcuda::wmma;
 
 #ifndef HEAD_DIM
 #define HEAD_DIM 128
 #endif
 #define TILE_TOK  16
 #define ROTARY_DIM 64
 #define WARPS_PER_CTA 4
 
 struct Sched { int2 kv; };
 
 template<int HD = HEAD_DIM>
 __global__ void pk_bsr_kernel(const half *__restrict__ Q,
                               const half *__restrict__ K,
                               const half *__restrict__ V,
                               half       *__restrict__ O,
                               const Sched* __restrict__ sched,
                               int  seqLen)
 {
     extern __shared__ half smem[];
     half *q = smem;                 // 16×HD
     half *k = q + TILE_TOK*HD;      // 16×HD
     half *v = k + TILE_TOK*HD;      // 16×HD
 
     int head = blockIdx.x;
     int lane = threadIdx.x & 31;
     int warp = threadIdx.x >> 5;
 
     const half *Qh = Q + (size_t)head*seqLen*HD;
     const half *Kh = K + (size_t)head*seqLen*HD;
     const half *Vh = V + (size_t)head*seqLen*HD;
           half *Oh = O + (size_t)head*seqLen*HD;
 
     namespace cg = cooperative_groups;
     cg::thread_block tb = cg::this_thread_block();
     const float inv_sqrt_hd = 1.f / sqrtf(float(HD));
 
     for (int qBase = blockIdx.y * TILE_TOK; qBase < seqLen; qBase += gridDim.y*TILE_TOK)
     {
         // Load Q tile with regular copies
         for (int e = lane + warp*32; e < TILE_TOK*HD; e += WARPS_PER_CTA*32) {
             int t = e / HD, d = e % HD;
             if ((d & 3) == 0) {
                 *(int4*)(q + t*HD + d) = *(const int4*)(Qh + (qBase + t)*HD + d);
             }
         }
         cg::sync(tb);
 
         if (ROTARY_DIM) {
             for (int t = warp; t < TILE_TOK; t += WARPS_PER_CTA) {
                 float theta = float(qBase+t) * 0.0001f;
                 for (int d = lane*4; d < ROTARY_DIM; d += 128)
                     rotary_half4(*(half4*)(q+t*HD+d),
                                  *(half4*)(q+t*HD+d),
                                  __sinf(theta+d), __cosf(theta+d));
             }
         }
         cg::sync(tb);
 
         int2 kvInfo = sched[qBase >> 4].kv;
         int kStart = kvInfo.x;
         int kTiles = kvInfo.y;
 
         float accum[TILE_TOK] = {0}; // Each element is a float
 
         for (int kt = 0; kt < kTiles; ++kt)
         {
             int kBase = (kStart + kt) * TILE_TOK;
 
             // Load K and V tiles with regular copies
             for (int e = lane + warp*32; e < TILE_TOK*HD; e += WARPS_PER_CTA*32) {
                 int t = e / HD, d = e % HD;
                 if ((d & 3) == 0) {
                     *(int4*)(k + t*HD + d) = *(const int4*)(Kh + (kBase + t)*HD + d);
                     *(int4*)(v + t*HD + d) = *(const int4*)(Vh + (kBase + t)*HD + d);
                 }
             }
             cg::sync(tb);
 
             for (int qt = warp; qt < TILE_TOK; qt += WARPS_PER_CTA) {
                 fragment<matrix_a, 16, 16, 16, half, row_major> fq;
                 fragment<matrix_b, 16, 16, 16, half, col_major> fk;
                 fragment<accumulator, 16, 16, 16, float> fr;
                 fill_fragment(fr, 0.f);
 
                 for (int d = 0; d < HD; d += 16) {
                     load_matrix_sync(fq, q + qt*HD + d, HD);
                     load_matrix_sync(fk, k + d, HD);
                     mma_sync(fr, fq, fk, fr);
                 }
 
                 float r = 0.f;
                 for (int x = 0; x < fr.num_elements; ++x) r += fr.x[x];
                 accum[qt] += r;
             }
             cg::sync(tb);
 
             for (int qt = warp; qt < TILE_TOK; qt += WARPS_PER_CTA) {
                 float w = __expf(accum[qt] * inv_sqrt_hd);
                 for (int d = lane; d < HD; d += 32) {
                     half vo = v[d + ((lane >> 5) ? (qt*HD) : ( (d/HD)*HD ))];
                     Oh[(qBase + qt)*HD + d] = __float2half_rn(w * __half2float(vo));
                 }
             }
             cg::sync(tb);
         }
     }
 }
 
 extern "C" void pk_attn(const void* Q, const void* K, const void* V,
                         void* O, const Sched* sched, int seq)
 {
     dim3 grid(8, (seq + TILE_TOK-1)/TILE_TOK);
     dim3 block(WARPS_PER_CTA*32);
     size_t smem = 3ULL * TILE_TOK * HEAD_DIM * sizeof(half);
 
     pk_bsr_kernel<<<grid, block, smem>>>(
         static_cast<const half*>(Q),
         static_cast<const half*>(K),
         static_cast<const half*>(V),
         static_cast<half*>(O),
         sched, seq);
 }