#!/usr/bin/env python3
"""
Benchmark Persistent-Kernel attention variants (+ optional sparse BSR)
---------------------------------------------------------------------

• Rebuilds the .so’s via `make`
• Times each kernel over `LOOPS` iterations
• Reports µs / (token·head)
• Optional sparsity schedule for BSR kernels
• Robust: rows print “n/a” if the .so or symbol is missing
"""
from __future__ import annotations
import argparse, ctypes, importlib, os, subprocess, sys, time
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.manual_seed(0)
assert torch.cuda.is_available(), "CUDA GPU required"

# ───────────────────────── CLI & defaults ───────────────────────────
P = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
P.add_argument("--seq",      type=int,   default=1024, help="sequence length (tokens)")
P.add_argument("--heads",    type=int,   default=8,    help="# attention heads")
P.add_argument("--hdim",     type=int,   default=128,  help="head dimension")
P.add_argument("--loops",    type=int,   default=50,   help="# timing loops")
P.add_argument("--sparsity", type=float, default=0.50, help="fraction of K-tiles masked out")
P.add_argument("--sched",    type=str,   default="",   help="optional .npy int32 schedule")
args = P.parse_args()
SEQ, HEADS, HDIM, LOOPS = args.seq, args.heads, args.hdim, args.loops

# ────────────────────────── (re)compile ─────────────────────────────
try:
    subprocess.check_call(["make"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except subprocess.CalledProcessError as e:
    sys.exit(f"[make] failed → {e}")

# ─────────────────── allocate benchmark tensors ─────────────────────
Q = torch.randn(1, HEADS, SEQ, HDIM, device="cuda", dtype=torch.float16)
K = Q.clone();  V = Q.clone();  O = torch.empty_like(Q)

# ─────────────────── helpers to load .so files ──────────────────────
def load_so(path: str) -> ctypes.CDLL | None:
    if not os.path.exists(path):
        return None
    return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)

def get_sym(lib: ctypes.CDLL, candidates: list[str]):
    """Return first symbol found (or None)."""
    for name in candidates:
        try:
            return getattr(lib, name)
        except AttributeError:
            continue
    return None

# paths → .so
SO_PATHS = {
    "Scalar": "./pk_attn_scalar.so",
    "Opt"   : "./pk_attn_opt.so",
    "MMA"   : "./pk_attn_mma.so",
    "DP4A"  : "./pk_attn_dp4a.so",
    "BSR"   : "./pk_attn_bsr.so",
}
LIBS = {name: load_so(p) for name, p in SO_PATHS.items()}

# ─────────────────── sparsity schedule (for BSR) ────────────────────
def build_random_schedule(seq: int, tile: int = 16, sparsity: float = 0.5) -> np.ndarray:
    """Return int32 buffer of (tile_idx , n_keep) pairs – toy example."""
    n_tiles = (seq + tile - 1) // tile
    keep   = np.random.rand(n_tiles) > sparsity
    tiles  = np.nonzero(keep)[0].astype(np.int32)
    n_keep = np.full_like(tiles, 1, dtype=np.int32)   # one tile per keep-entry
    return np.vstack([tiles, n_keep]).T.reshape(-1)

if args.sched:
    SCHED = np.load(args.sched).astype(np.int32)
else:
    SCHED = build_random_schedule(SEQ, sparsity=args.sparsity)

SCHED_TORCH = torch.from_numpy(SCHED).cuda(non_blocking=True)
SCHED_PTR   = ctypes.c_void_p(int(SCHED_TORCH.data_ptr()))

# ─────────────────── timing helpers ─────────────────────────────────
def bench_generic(sym, needs_sched: bool) -> float:
    """µs per (token·head)"""
    t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
    t0.record()
    for _ in range(LOOPS):
        if needs_sched:
            sym(Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), SEQ, SCHED_PTR)
        else:
            sym(Q.data_ptr(), K.data_ptr(), V.data_ptr(), O.data_ptr(), SEQ)
    t1.record(); torch.cuda.synchronize()
    return 1000.0 * t0.elapsed_time(t1) / (LOOPS * SEQ * HEADS)

def bench_flash() -> float | None:
    try:
        fa_mod = importlib.import_module("flash_attn.flash_attn_interface")
        q_bthd = Q.transpose(1, 2).contiguous()
        # warm-up
        fa_mod.flash_attn_func(q_bthd, q_bthd, q_bthd, dropout_p=0.0, causal=False)
        torch.cuda.synchronize()
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        t0.record()
        for _ in range(LOOPS):
            fa_mod.flash_attn_func(q_bthd, q_bthd, q_bthd, dropout_p=0.0, causal=False)
        t1.record(); torch.cuda.synchronize()
        return 1000.0 * t0.elapsed_time(t1) / (LOOPS * SEQ * HEADS)
    except Exception as e:
        print(f"[warn] Flash-Attention-2 unavailable → {e}")
        return None

# ───────────────────── run benchmark table ──────────────────────────
print(f"\nseq={SEQ}  heads={HEADS}  head_dim={HDIM}  loops={LOOPS}"
      f"\n sparsity={args.sparsity:.2f}  schedule_len={len(SCHED)}")
print(f"{'kernel':8} | time  (µs / tok-head)")
print("────────────┼─────────────────────")

for name, lib in LIBS.items():
    if lib is None:
        print(f"{name:8} |   n/a")
        continue

    # resolve symbol(s)
    if name == "BSR":
        sym = get_sym(lib, ["pk_attn_bsr_sched", "pk_attn"])   # prefer extended signature
        if sym is None:
            print(f"{name:8} |   n/a")
            continue
        needs_sched = sym.__name__ == "pk_attn_bsr_sched"
        # set argtypes
        if needs_sched:
            sym.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int, ctypes.c_void_p]
        else:
            sym.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]
    else:
        sym = get_sym(lib, ["pk_attn"])
        if sym is None:
            print(f"{name:8} |   n/a")
            continue
        sym.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]
        needs_sched = False

    print(f"{name:8} | {bench_generic(sym, needs_sched):7.3f}")

flash_t = bench_flash()
print(f"{'Flash2':8} | {flash_t:7.3f}" if flash_t else f"{'Flash2':8} |   n/a")
