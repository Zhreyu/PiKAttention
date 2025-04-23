# PiKAttention  🐱

*A tiny collection of CUDA kernels that showcase persistent–kernel (PK) tricks for Transformer attention!*  

---

## ✨ Highlights

| Kernel | Idea in one line | .so produced |
|--------|-----------------|--------------|
| **Scalar** | naïve FP16 MAC loop – reference timing | `pk_attn_scalar.so` |
| **Opt** | micro‑unrolled & vectorised (float2/float4) | `pk_attn_opt.so` |
| **MMA** | WMMA Tensor‑Core mma.sync (16×16×16) | `pk_attn_mma.so` |
| **DP4A** | 8‑bit INT4 KV, DP4A accumulate to FP16 | `pk_attn_dp4a.so` |
| **PiKA‑BSR** | **P**ersistent‑**K**ernel, block‑**A**ttention, 16‑token tiles kept in shared mem + user‑supplied block‑sparse schedule | `pk_attn_bsr.so` |

### What “PiKAttention” means

> **PiKA** = **P**ersistent‑**K**ernel **A**ttention.  
> One persistent CTA keeps a <code>16 × D</code> query tile in shared memory, streams just the K/V tiles it needs, and writes out the softmax‑weighted values – cutting DRAM traffic to the bone.

I don’t claim state‑of‑the‑art – the goal is to **demo clean PK patterns** you can graft into your own models.

---

## 🚀 Quick start

```bash
# clone & build
make          # produces all *.so under default sm_86; override CUDA_ARCH if needed
!pip install -r requirements.txt
# reference benchmark (dense, 1 k tokens, 8 heads, 128‑d)
python bench.py               

# sparse run: 2 048 tokens, 256‑d heads, 60 % sparsity
python bench.py --seq 2048 --hdim 256 --sparsity 0.6

# full sweep with CSV + plots
python benchmark.py \
       --seq 1024 2048 \
       --heads 8 \
       --hdim 128 256 \
       --sparsity 0 0.5 0.9 \
       --loops 50 \
       --csv sweeps/pk_results.csv --plot
```

> 🗒️  **Flash‑Attention‑2** timings are auto‑included when the Python package is importable – handy sanity check.

---

## 📂 Repo layout

```
│  bench.py            # quick single‑point benchmark
│  benchmark.py        # multi‑dim sweep + plots/CSV
│  make_bsr.py         # make random block‑sparse schedules (.npy)
│  Makefile            # toggles HEAD_DIM via env var, outputs *.so
│
├─ kernels/
│   ├─ pk_attn_common.h  # cp.async helpers, rotary, etc.
│   ├─ pk_attn_scalar.cu
│   ├─ pk_attn_opt.cu
│   ├─ pk_attn_mma.cu
│   ├─ pk_attn_dp4a.cu
│   └─ pk_attn_bsr.cu    # the PiKA magic
└─ sweeps/               # CSV & logs land here
```

---

## 📈 Example numbers (RTX3060 ↯ sm86, FP16)

<!-- | seq | heads | dim | sparsity | Scalar | PiKA‑BSR | speed‑up |
|-----|-------|-----|----------|--------|----------|-----------|
|1 024| 8|128|0.50|0.028 µs|**0.004 µs**| ×7.0|
|4 096|16|256|0.60|0.012 µs|**0.001 µs**| ×12 | -->

Latencies are per *token·head*; lower is better.

---

## 🧩 Tinkering guide

* **Change head dim** – re‑export when building:
  ```bash
  HEAD_DIM=256 make clean all
  ```
* **Bring your own schedule** – `make_bsr.py` can emit `.npy` that `bench.py --sched my.npy` consumes.
* **INT4 path** – check `pk_attn_dp4a.cu`; INT4 quant + DP4A accumulate, handy for low‑mem.

---

## 🔭 Future ideas

* column‑parallel PiKA‑BSR for multi‑SM scaling (1 block per head right now)
* FP8 kernels once H100 becomes dime‑a‑dozen 🤓
* integrate Triton for auto‑tuning (but still keep the raw CUDA around for learning)
* Paged KV Cache for decoding
* Embed in a model (YES)
---

## 📜 License

MIT – do whatever, just keep a reference back.

