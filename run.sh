#!/usr/bin/env bash
# --------------------------------------------------------------------
# run_sweep.sh
# Grid-sweep wrapper for     bench.py     +     make_bsr.py
#
#   • iterates over {seq, head-dim, sparsity}
#   • auto-recompiles kernels once per head-dim
#   • generates a deterministic BSR schedule (.npy) for every {seq,sparsity}
#   • calls bench.py with those arguments
#   • captures all stdout in a log-file under ./sweeps/
# --------------------------------------------------------------------
set -euo pipefail                                # safer bash
shopt -s lastpipe                                # for read loops

# -------- sweep parameters ----------------------------------------------------
SEQS=(1024 2048)          # token counts
HEADS=8                   # number of heads (fixed here, change if you want)
HDIMS=(128 256)           # head dimensions
SPARSITIES=(0.0 0.5 0.9)  # fraction of *K* tiles to mask
LOOPS=50                  # timing loops inside bench.py
TILE=16                   # tile-size used by make_bsr.py

# -------- folders / log -------------------------------------------------------
mkdir -p sweeps
LOGFILE="sweeps/bench_$(date +%Y%m%d_%H%M%S).log"
touch   "${LOGFILE}"

echo "results will be appended to ${LOGFILE}"
echo "──────────────────────────────────────────────" | tee -a "${LOGFILE}"

# -------- helper: nice echo that also logs ------------------------------------
log() { echo -e "$*" | tee -a "${LOGFILE}" ; }

# ==========================================================================
for HDIM in "${HDIMS[@]}"; do
    # --- compile kernels once for this head-dim --------------------------
    log "\n==========  BUILD  (HEAD_DIM=${HDIM})  =========="
    HEAD_DIM=${HDIM} make -s -C "$(dirname "$0")" clean all

    for SEQ in "${SEQS[@]}"; do
        for SP in "${SPARSITIES[@]}"; do
            # -------- build / reuse BSR schedule -------------------------
            SCHED_FILE="sweeps/sched_seq${SEQ}_sp$(printf '%.2f' "${SP}" | tr . _)_"\
"d${HDIM}.npy"
            if [[ ! -f "${SCHED_FILE}" ]]; then
                python make_bsr.py              \
                       --seq       "${SEQ}"     \
                       --tile      "${TILE}"    \
                       --sparsity  "${SP}"      \
                       --out       "${SCHED_FILE}"
            fi

            # -------- run the benchmark ---------------------------------
            log "\n▶  seq=${SEQ}  heads=${HEADS}  hdim=${HDIM}  sparsity=${SP}"
            python bench.py                            \
                   --seq      "${SEQ}"                 \
                   --heads    "${HEADS}"               \
                   --hdim     "${HDIM}"                \
                   --loops    "${LOOPS}"               \
                   --sparsity "${SP}"                  \
                   --sched    "${SCHED_FILE}"          \
               2>&1 | tee -a "${LOGFILE}"
        done
    done
done

log "\nall sweeps finished ✔   (full log in ${LOGFILE})"
