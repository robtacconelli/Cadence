#!/usr/bin/env bash
# SZ3 is used as an external comparison baseline (Section 5.2 of the paper).
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/ext" && cd "$ROOT/ext"
[ -d SZ3 ] || git clone --depth 1 https://github.com/szcompressor/SZ3.git
cd SZ3 && mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$ROOT/ext/sz3-install" -DBUILD_SHARED_LIBS=ON
make -j"$(nproc)" && make install
echo "SZ3 installed at $ROOT/ext/sz3-install"
echo "export LD_LIBRARY_PATH=$ROOT/ext/sz3-install/lib:\$LD_LIBRARY_PATH"
