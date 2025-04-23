#!/usr/bin/env python3
import numpy as np, argparse, math, os
p = argparse.ArgumentParser()
p.add_argument('--seq', type=int, required=True)
p.add_argument('--tile', type=int, default=16)
p.add_argument('--sparsity', type=float, required=True)
p.add_argument('--out', type=str, required=True)
args = p.parse_args()

n_tiles = math.ceil(args.seq / args.tile)
keep = max(1, round((1.0 - args.sparsity) * n_tiles))

sched = np.zeros((n_tiles, 2), dtype=np.int32)
for qt in range(n_tiles):
    # choose a contiguous run of K-tiles to keep
    start = np.random.randint(0, n_tiles - keep + 1)
    sched[qt,0] = start
    sched[qt,1] = keep

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
np.save(args.out, sched)
print("wrote", args.out, sched.shape)
