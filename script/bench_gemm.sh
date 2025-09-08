#!/usr/bin/env bash
set -euo pipefail

# ===== 默认配置（环境变量可覆盖） =====
INPUT_DEFAULT="./test/mm_bench_memref.mlir"   
OUTDIR="${OUTDIR:-out}"
TILER="${TILER:-auto}"                      # auto | affine | linalg
REPEATS="${REPEATS:-5}"
PREHEAT="${PREHEAT:-1}"
LLVM_CONFIG="${LLVM_CONFIG:-llvm-config-18}"
MLIR_OPT="${MLIR_OPT:-mlir-opt-18}"
RUNNER="${RUNNER:-mlir-cpu-runner-18}"

# GFLOPS 统计使用（不改变 IR 本身）
M="${M:-64}" N="${N:-32}" K="${K:-128}" N_RUNS="${N_RUNS:-200}"

# ===== 解析参数 =====
INPUT="$INPUT_DEFAULT"
TILES=()
if [[ $# -gt 0 && -f "$1" ]]; then
  INPUT="$1"; shift
fi
if [[ $# -gt 0 ]]; then
  TILES=("$@")
else
  TILES=("8,8,32" "16,16,16" "32,32,8")
fi

mkdir -p "$OUTDIR"
SHLIB_DIR="$($LLVM_CONFIG --libdir)"
BASE_RUN_ARGS=(-e main -entry-point-result=void
  -shared-libs="$SHLIB_DIR/libmlir_runner_utils.so"
  -shared-libs="$SHLIB_DIR/libmlir_c_runner_utils.so")

have_pass() { "$MLIR_OPT" --pass-list 2>/dev/null | grep -iq "$1"; }

# auto：优先 linalg-tile，缺失则 affine
if [[ "$TILER" == "auto" ]]; then
  if have_pass "linalg-tile"; then TILER="linalg"; else TILER="affine"; fi
fi
# 强制 linalg 但不可用 → 降级
if [[ "$TILER" == "linalg" ]] && ! have_pass "linalg-tile"; then
  echo "[warn] linalg-tile not registered; falling back to affine." >&2
  TILER="affine"
fi

build_one() {
  local ts="$1" out="$2"
  if [[ "$TILER" == "linalg" ]]; then
   
    "$MLIR_OPT" "$INPUT" \
      -pass-pipeline="builtin.module(func.func(linalg-generalize-named-ops, linalg-tile{tile-sizes=${ts}}, canonicalize, cse, convert-linalg-to-affine-loops, canonicalize, cse, lower-affine, convert-scf-to-cf, convert-to-llvm))" \
      > "$out"
  else
    "$MLIR_OPT" "$INPUT" \
      -linalg-generalize-named-ops \
      -convert-linalg-to-affine-loops -canonicalize -cse \
      "-affine-loop-tile=tile-sizes=${ts}" \
      -lower-affine -convert-scf-to-cf \
      -convert-to-llvm \
      > "$out"
  fi
}

median_of() { awk '{print $1}' | sort -n | awk '{a[NR]=$1} END{if(NR){print a[int((NR+1)/2)]}}'; }

echo ">>> Using:"
echo "INPUT=$INPUT"
echo "TILER=$TILER"
echo "SHLIB_DIR=$SHLIB_DIR"
echo "Dims: M=$M, N=$N, K=$K, N_RUNS=$N_RUNS"
echo

for ts in "${TILES[@]}"; do
  tag="${ts//,/x}"
  out="$OUTDIR/mm_${tag}.ll"
  build_one "$ts" "$out"

  # 预热
  for ((i=0;i<PREHEAT;i++)); do "$RUNNER" "$out" "${BASE_RUN_ARGS[@]}" >/dev/null; done

  # 多次测量：保留 /usr/bin/time 的 stderr，丢被测程序 stdout
  times=()
  for ((r=1;r<=REPEATS;r++)); do
    sec=$( { /usr/bin/time -f "%e" "$RUNNER" "$out" "${BASE_RUN_ARGS[@]}" >/dev/null; } 2>&1 )
    times+=("$sec")
  done

  median=$(printf '%s\n' "${times[@]}" | median_of)
  mean=$(printf '%s\n' "${times[@]}" | awk '{s+=$1} END{if(NR) printf("%.6f", s/NR); else print "NA"}')

  flops=$(( 2*M*N*K*N_RUNS ))
  gflops=$(python3 - <<PY
flops=${flops}
t=${median:-0}
print(f"{(flops/(t*1e9)) if t else 0:.3f}")
PY
)
  echo "tile-sizes=${ts} | repeats=${REPEATS} | median=${median}s | mean=${mean}s | GFLOPS=${gflops}"
done
