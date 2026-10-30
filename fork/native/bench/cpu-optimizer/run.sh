#!/usr/bin/env bash
# Times the cpu_training optimizer step (pinbench.cpp) with rank threads pinned the way NVML pins them, against
# cpu_adamw.cpp compiled with the trainer's own compile command: once as compiled (OpenMP, ledger T5) and once
# without -fopenmp (upstream's serial build). Prints one line per run; fork/docs/performance.md ("CPU 优化器并行")
# records the results and the host. No GPU is needed, only the CUDA headers the configured build uses.
#
# usage: fork/native/bench/cpu-optimizer/run.sh BUILD_DIR [OUT_DIR]
#   BUILD_DIR  a configured csrc build directory with compile_commands.json (CMAKE_EXPORT_COMPILE_COMMANDS=ON)
#   OUT_DIR    objects and binaries (default: a new directory under ${TMPDIR:-/tmp})
# environment:
#   CPU_ADAMW_SRC=<file>  compile this cpu_adamw.cpp instead of the checkout's, e.g. an older revision:
#                         CPU_ADAMW_SRC=<(git show <rev>:csrc/src/runtime/optimizers/cpu_adamw.cpp)
#   ROUNDS=<n>            repetitions of the whole matrix (default 3); compare medians across rounds
#   CONFIGS="<R PIN SOCKETS STEPS SHAPE>;..."  replaces the default matrix (pinbench.cpp explains the fields)
# The pinned configurations use CPUs 0-3. Other processes on the host distort the pinned runs most: a team
# thread that loses its CPU stalls every barrier of its region.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/../../../.." && pwd)
BUILD=$(cd "${1:?usage: run.sh BUILD_DIR [OUT_DIR]}" && pwd)
OUT=${2:-$(mktemp -d "${TMPDIR:-/tmp}/cpu-optimizer-bench.XXXXXX")}
mkdir -p "$OUT"
SRC=${CPU_ADAMW_SRC:-$ROOT/csrc/src/runtime/optimizers/cpu_adamw.cpp}

# The compile command of cpu_adamw.cpp, without its -o/-c operands, one argument per line.
python3 - "$BUILD/compile_commands.json" > "$OUT/args.txt" <<'PY'
import json, shlex, sys
for entry in json.load(open(sys.argv[1])):
    if entry["file"].endswith("runtime/optimizers/cpu_adamw.cpp"):
        args = entry.get("arguments") or shlex.split(entry["command"])
        break
else:
    sys.exit("cpu_adamw.cpp is not in compile_commands.json")
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
mapfile -t CC < "$OUT/args.txt"
SERIAL=()
for a in "${CC[@]}"; do [[ $a == -fopenmp* || $a == *_EXPORTS ]] || SERIAL+=("$a"); done

echo "+ compiling $SRC" >&2
# A copy, because CPU_ADAMW_SRC may be a pipe such as <(git show ...): readable once, without a .cpp suffix.
cat "$SRC" > "$OUT/cpu_adamw.cpp"
"${CC[@]}" -c "$OUT/cpu_adamw.cpp" -o "$OUT/cpu_adamw_omp.o"
"${SERIAL[@]}" -c "$OUT/cpu_adamw.cpp" -o "$OUT/cpu_adamw_serial.o"
"${SERIAL[@]}" -c "$HERE/pinbench.cpp" -o "$OUT/pinbench.o"
"${CC[0]}" "$OUT/pinbench.o" "$OUT/cpu_adamw_omp.o" -o "$OUT/pinbench-openmp" -fopenmp -pthread
"${CC[0]}" "$OUT/pinbench.o" "$OUT/cpu_adamw_serial.o" -o "$OUT/pinbench-serial" -pthread

IFS=';' read -r -a configs <<< "${CONFIGS:-1 2 1 7 toy:8388608;2 2 2 7 toy:8388608;2 2 1 7 toy:8388608;1 0 1 7 toy:8388608;1 2 1 5 llama:8:896:4864:32000;2 2 2 5 llama:8:896:4864:32000;2 2 1 5 llama:8:896:4864:32000;1 0 1 5 llama:8:896:4864:32000}"
for round in $(seq "${ROUNDS:-3}"); do
    for config in "${configs[@]}"; do
        for build in serial openmp; do
            # shellcheck disable=SC2086  # the config is a list of arguments
            echo "round=$round build=$build $("$OUT/pinbench-$build" $config)"
        done
    done
done
