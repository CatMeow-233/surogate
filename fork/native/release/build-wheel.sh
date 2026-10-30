#!/usr/bin/env bash
# Build the fork's native wheel (trainer + serving engine) inside the CUDA build image.
#
# The steps mirror the `wheel` job of upstream's .github/workflows/wheel.yml (as of upstream
# 282f9901): same image, system packages, Rust toolchain, version tagging, build targets,
# auditwheel excludes and artefact checks. Differences: one CUDA architecture set chosen by the
# caller (default sm_120a, the RTX 50 series) for both the trainer and the serving engine, a
# version taken from the native-v* tag, and no install.sh. When a sync changes wheel.yml, port
# the change here.
#
# Run from the repository root inside nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04 (as root):
#   RELEASE_TAG=native-v1.5.12.post1 bash fork/native/release/build-wheel.sh
# Environment:
#   RELEASE_TAG   native-v<PEP 440 version, canonical spelling, no local part> (a publish run of the
#                 workflow derives it from fork/native/release/VERSION); without it, as in build-only
#                 runs, the wheel is versioned 0.0.0.dev0
#   ARCHS         CUDA architectures for trainer and engine, ';'-separated (default 120a)
#   CCACHE_DIR    compiler cache (default /ccache)
#   HOST_UID      if set, the checkout is chowned back to it at the end (the build runs as root)
# Output: wheelhouse/*.whl (auditwheel-repaired), wheelhouse/SHA256SUMS, wheelhouse/build-info.json
set -euo pipefail

ARCHS=${ARCHS:-120a}
export CCACHE_DIR=${CCACHE_DIR:-/ccache}
cd "$(dirname "$0")/../../.."
# shellcheck source=../../tools/release_gate.sh
. fork/tools/release_gate.sh # release_version_ok: the check plan.sh applies to VERSION

if [ -n "${RELEASE_TAG:-}" ]; then
    VERSION=${RELEASE_TAG#native-v}
    if [[ "$RELEASE_TAG" != native-v* ]] || ! release_version_ok "$VERSION"; then
        echo "RELEASE_TAG must be native-v<PEP 440 version without a local part, canonical spelling>, got '$RELEASE_TAG'" >&2
        exit 2
    fi
else
    VERSION=0.0.0.dev0
fi

echo "::group::System packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends --allow-change-held-packages \
    git wget curl build-essential cmake ninja-build python3-dev ccache zstd libopenmpi-dev libdw-dev \
    libelf-dev libicu-dev patch libnccl2 libnccl-dev g++-13 pkg-config libcurl4-openssl-dev \
    libavformat-dev libavcodec-dev libavutil-dev libswscale-dev libswresample-dev libnuma-dev
echo "::endgroup::"

echo "::group::Rust and uv"
wget -qO /tmp/rustup.sh https://sh.rustup.rs
sh /tmp/rustup.sh -y --profile minimal --default-toolchain 1.90.0
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/0.12.23/install.sh | sh
uv --version
echo "::endgroup::"

git config --global --add safe.directory "$PWD"

# Same sizing rule as wheel.yml: the CUTLASS FP4 units take about 8 GB of cicc per job.
mem_gb=$(awk '/^MemTotal:/ {print int($2 / 1048576)}' /proc/meminfo)
swap_gb=$(awk '/^SwapTotal:/ {print int($2 / 1048576)}' /proc/meminfo)
jobs=$(( (mem_gb + swap_gb) / 8 )); cpus=$(nproc)
[ "$jobs" -gt "$cpus" ] && jobs=$cpus
[ "$jobs" -lt 1 ] && jobs=1
echo "${cpus} CPUs, ${mem_gb} GB memory, ${swap_gb} GB swap: ${jobs} build jobs; archs ${ARCHS}; version ${VERSION}"

# Static version and serve architectures go into this checkout's pyproject.toml, which the
# upstream script then rewrites with the +cu130 tag and the CUDA dependencies. A list-valued
# define reaches CMake as a ';'-joined list, which SKBUILD_CMAKE_DEFINE cannot carry.
uv run --no-project --with tomlkit python3 - "$VERSION" "$ARCHS" <<'PY'
import sys
import tomlkit

version, archs = sys.argv[1], sys.argv[2].split(";")
with open("pyproject.toml") as f:
    data = tomlkit.load(f)
data["project"]["version"] = version
data["tool"]["scikit-build"]["cmake"]["define"]["SUROGATE_SERVE_CUDA_ARCHS"] = archs
with open("pyproject.toml", "w") as f:
    tomlkit.dump(data, f)
PY
uv run --no-project --with tomlkit python3 .github/scripts/set_cuda_version_tag.py cu130

echo "::group::Build wheel"
mkdir -p "$CCACHE_DIR"
ccache --max-size=5G
ccache --zero-stats
rm -rf wheelhouse
CMAKE_GENERATOR=Ninja CMAKE_C_COMPILER=gcc-13 CMAKE_CXX_COMPILER=g++-13 \
CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_CUDA_COMPILER_LAUNCHER=ccache \
CCACHE_CUDA_PATHS=/usr/local/cuda NVCC_THREADS=1 CUDAARCHS="$ARCHS" UV_NO_SYNC=1 \
SKBUILD_CMAKE_DEFINE=SUROGATE_REQUIRE_SERVE=ON CMAKE_BUILD_PARALLEL_LEVEL=$jobs \
    uv build --wheel --out-dir wheelhouse
ccache --show-stats
echo "::endgroup::"

echo "::group::Repair wheel"
uv run --no-project --with auditwheel --with patchelf auditwheel repair wheelhouse/*.whl -w repaired/ \
    --exclude libcuda.so.1 --exclude libcudart.so.12 --exclude libcudart.so.13 --exclude libcudnn.so.9 \
    --exclude libcufile.so.0 --exclude libnccl.so.2 --exclude libcublas.so.12 --exclude libcublas.so.13 \
    --exclude libcublasLt.so.12 --exclude libcublasLt.so.13 --exclude libnvidia-ml.so.1 \
    --exclude libcupti.so.13 --exclude libnvrtc.so.13 --exclude libtorch_cpu.so --exclude libtorch.so \
    --exclude libc10.so
rm -f wheelhouse/*.whl
mv repaired/*.whl wheelhouse/
rmdir repaired
ls -lh wheelhouse/
echo "::endgroup::"

# The wheel must carry the engine, and both libraries cubins for every requested architecture.
check=$(mktemp -d)
python3 -m zipfile -e wheelhouse/*.whl "$check"
for f in surogate/serve/_bin/surogate-engine surogate/serve/_bin/surogate-engine-cli \
         surogate/serve/_bin/surogate-embed surogate/serve/_bin/surogate-stt \
         surogate/serve/_bin/surogate-stt-lm surogate/serve/_bin/surogate-tts \
         surogate/serve/_bin/surogate-tts-worker surogate/libsinfer.so \
         surogate/serve/_llama_cpp/bin/llama-quantize surogate/libsurogate-common.so; do
    [ -f "$check/$f" ] || { echo "the wheel is missing $f" >&2; exit 1; }
done
for lib in libsinfer.so libsurogate-common.so; do
    cubins=$(cuobjdump -lelf "$check/surogate/$lib" | grep -oE 'sm_[0-9]+a?' | sort | uniq -c)
    echo "$lib:"; echo "$cubins"
    for arch in ${ARCHS//;/ }; do
        grep -q "sm_${arch}" <<<"$cubins" || { echo "$lib has no sm_${arch} cubins" >&2; exit 1; }
    done
done
rm -rf "$check"

cat > wheelhouse/build-info.json <<EOF
{"version": "$VERSION", "release_tag": "${RELEASE_TAG:-}", "source_commit": "$(git rev-parse HEAD)",
 "cuda": "13.1", "archs": "$ARCHS", "image": "nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04"}
EOF
(cd wheelhouse && sha256sum ./*.whl build-info.json > SHA256SUMS && cat SHA256SUMS)

# The build edited pyproject.toml in place; restore it so the checkout stays clean.
git checkout -- pyproject.toml
[ -z "${HOST_UID:-}" ] || chown -R "$HOST_UID" .
