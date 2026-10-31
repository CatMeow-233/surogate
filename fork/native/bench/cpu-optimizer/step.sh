#!/usr/bin/env bash
# Times the host side of a cpu_training step before and after fork ledgers T13-T15 (stepbench.cpp), against the
# trainer's own objects: cpu_adamw.cpp and fork/native/src/cpu_step_kernels.cpp compiled with their commands from
# compile_commands.json, so the kernels get their per-source flags (-fno-math-errno -ffp-contract=off). It also
# builds the kernels once per ISA without clones (x86-64 baseline, AVX2, x86-64-v4 where this host has it) and
# checks each against upstream's loop over every pair of BF16 values and by AdamW hash. Prints one line per run;
# fork/docs/performance.md ("cpu_training 主机端步") records the results and the host. No GPU is needed, only the
# CUDA headers the configured build uses.
#
# usage: fork/native/bench/cpu-optimizer/step.sh BUILD_DIR [OUT_DIR]
#   BUILD_DIR  a csrc build directory configured with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON (see run.sh)
#   OUT_DIR    objects and binaries (default: a new directory under ${TMPDIR:-/tmp})
# environment:
#   ROUNDS=<n>   repetitions of the timing matrix (default 3); compare medians across rounds
#   REPS=<n>     timed repetitions inside each run (default 5)
#   SMALL=<shape> SHAPE of the small-model rows (default llama:8:896:4864:32000, as run.sh's Llama shape)
#   LARGE=<shape> SHAPE of the 8B-layer rows (default llama:2:4096:12288:0: two Qwen3-8B blocks, 386M elements)
#   ISA=0        skip the per-ISA bit checks
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
BUILD=$(cd "${1:?usage: step.sh BUILD_DIR [OUT_DIR]}" && pwd)
[[ -f $BUILD/compile_commands.json ]] || {
    echo "step.sh: $BUILD has no compile_commands.json;" \
        "run: cmake -S csrc -B $BUILD -DCMAKE_EXPORT_COMPILE_COMMANDS=ON" >&2
    exit 1
}
OUT=${2:-$(mktemp -d "${TMPDIR:-/tmp}/cpu-step-bench.XXXXXX")}
mkdir -p "$OUT"

# The compile commands of the two sources, without their -o/-c operands, one argument per line.
args_of() {
    python3 - "$BUILD/compile_commands.json" "$1" <<'PY'
import json, shlex, sys
for entry in json.load(open(sys.argv[1])):
    if entry["file"].endswith(sys.argv[2]):
        args = entry.get("arguments") or shlex.split(entry["command"])
        break
else:
    sys.exit(sys.argv[2] + " is not in compile_commands.json (configure the build after the T14 change)")
out, skip = [], False
for a in args:
    if skip:
        skip = False
    elif a in ("-o", "-c"):
        skip = True
    else:
        out.append(a)
print("\n".join(out))
PY
}
mapfile -t ADAMW_CC < <(args_of runtime/optimizers/cpu_adamw.cpp)
mapfile -t KERNEL_CC < <(args_of fork/native/src/cpu_step_kernels.cpp)

echo "+ compiling cpu_adamw.cpp, cpu_step_kernels.cpp and stepbench.cpp" >&2
"${ADAMW_CC[@]}" -c "$ROOT/csrc/src/runtime/optimizers/cpu_adamw.cpp" -o "$OUT/cpu_adamw.o"
"${KERNEL_CC[@]}" -c "$ROOT/fork/native/src/cpu_step_kernels.cpp" -o "$OUT/kernels.o"
"${ADAMW_CC[@]}" -c "$HERE/stepbench.cpp" -o "$OUT/stepbench.o"
"${ADAMW_CC[0]}" "$OUT/stepbench.o" "$OUT/cpu_adamw.o" "$OUT/kernels.o" -o "$OUT/stepbench" -fopenmp -pthread

SMALL=${SMALL:-llama:8:896:4864:32000}
LARGE=${LARGE:-llama:2:4096:12288:0}
REPS=${REPS:-5}
B=$OUT/stepbench
for round in $(seq "${ROUNDS:-3}"); do
    echo "round=$round $("$B" memset "$SMALL" 0 "$REPS")"
    echo "round=$round $("$B" memset "$LARGE" 0 "$REPS")"
    for pin in 1 2 4; do
        echo "round=$round $("$B" accum "$SMALL" "$pin" "$REPS")"
        echo "round=$round $("$B" accum "$LARGE" "$pin" "$REPS")"
        echo "round=$round $(env -u SUROGATE_FORK_CPU_ADAMW_SIMD "$B" adamw "$SMALL" "$pin" "$REPS")"
        echo "round=$round $(SUROGATE_FORK_CPU_ADAMW_SIMD=1 "$B" adamw "$SMALL" "$pin" "$REPS")"
    done
    echo "round=$round $("$B" group "$SMALL" 0 "$REPS")"
done

[[ ${ISA:-1} == 0 ]] && exit 0
echo "isa=clones $("$B" exhaust)"
# One object per ISA, without clones; the whole object is built for that ISA. The x86-64-v4 build runs only where
# this host has AVX-512 F/BW/DQ/VL.
isas=("-march=x86-64" "-mavx2")
grep -qw avx512bw /proc/cpuinfo && grep -qw avx512dq /proc/cpuinfo && grep -qw avx512vl /proc/cpuinfo \
    && isas+=("-march=x86-64-v4")
for isa in "${isas[@]}"; do
    name=${isa#-m}
    name=${name#arch=}
    "${KERNEL_CC[@]}" "$isa" -DSUROGATE_FORK_SIMD_CLONES= -c "$ROOT/fork/native/src/cpu_step_kernels.cpp" \
        -o "$OUT/kernels-$name.o"
    "${ADAMW_CC[0]}" "$OUT/stepbench.o" "$OUT/cpu_adamw.o" "$OUT/kernels-$name.o" -o "$OUT/stepbench-$name" \
        -fopenmp -pthread
    echo "isa=$name $("$OUT/stepbench-$name" exhaust)"
    echo "isa=$name $(env -u SUROGATE_FORK_CPU_ADAMW_SIMD "$OUT/stepbench-$name" adamw "$SMALL" 0 2)"
    echo "isa=$name $(SUROGATE_FORK_CPU_ADAMW_SIMD=1 "$OUT/stepbench-$name" adamw "$SMALL" 0 2)"
done
