#!/usr/bin/env bash
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Build XRT + XDNA driver with automatic dependency discovery.
# Scans CMakeLists.txt from source to find all find_package() deps,
# resolves them to apt packages, and installs before building.
# Usage: ./build_xdna.sh [driver_version]

set -euo pipefail

DRIVER_VERSION="${1:-854ff04}"
WORKDIR="${WORKDIR:-/ryzers}"

install_base() {
    apt-get update -qq
    apt-get install -y --no-install-recommends \
        git curl wget cmake build-essential pkg-config \
        pciutils dkms python3-dev \
        libboost-all-dev libssl-dev libprotobuf-dev rapidjson-dev \
        libdrm-dev libelf-dev uuid-dev libcurl4-openssl-dev \
        ocl-icd-opencl-dev libncurses-dev libffi-dev
}

clone_source() {
    [ -d "$WORKDIR/xdna-driver" ] || \
        git clone https://github.com/amd/xdna-driver "$WORKDIR/xdna-driver"
    cd "$WORKDIR/xdna-driver"
    git checkout "$DRIVER_VERSION"
    git submodule update --init --recursive
}

# Scan all CMakeLists.txt for find_package() calls → list of cmake package names
discover_cmake_deps() {
    grep -rh 'find_package\s*(' "$WORKDIR/xdna-driver" \
        --include='CMakeLists.txt' --include='*.cmake' 2>/dev/null \
        | grep -oP 'find_package\s*\(\s*\K[A-Za-z0-9_]+' \
        | sort -u
}

# Standard cmake→apt mapping (works across Ubuntu versions, not project-specific)
declare -A CMAKE_TO_APT=(
    [Boost]=libboost-all-dev  [boost_filesystem]=libboost-all-dev
    [OpenSSL]=libssl-dev  [Protobuf]=libprotobuf-dev
    [RapidJSON]=rapidjson-dev [CURL]=libcurl4-openssl-dev [Curses]=libncurses-dev
    [PkgConfig]=pkg-config    [Python3]=python3-dev  [PythonInterp]=python3
    [PythonLibs]=python3-dev  [GTest]=libgtest-dev   [Doxygen]=doxygen
    [OpenCL]=ocl-icd-opencl-dev [Threads]=""  [Git]=""  [UnixCommands]=""
    [uuid]=uuid-dev  [LibElf]=libelf-dev  [libffi]=libffi-dev
    [pybind11]=pybind11-dev   [cxxopts]=libcxxopts-dev [absl]=libabsl-dev
    [libdrm]=libdrm-dev       [libudev]=libudev-dev  [systemd]=libsystemd-dev
    [yaml-cpp]=libyaml-dev
)

resolve_apt_pkg() {
    local cmake_name="$1"
    local lower=$(echo "$cmake_name" | tr '[:upper:]' '[:lower:]')

    # 1. Check well-known mapping (instant)
    if [[ -v CMAKE_TO_APT[$cmake_name] ]]; then
        local mapped="${CMAKE_TO_APT[$cmake_name]}"
        [ -n "$mapped" ] && echo "$mapped"
        return $([ -n "$mapped" ] && echo 0 || echo 1)
    fi

    # 2. Try common naming patterns (fast, no network)
    for try in "lib${lower}-dev" "${lower}-dev"; do
        apt-cache show "$try" &>/dev/null && echo "$try" && return
    done

    # 3. apt-cache search fallback
    local search_pkg
    search_pkg=$(apt-cache search "^lib${lower}" 2>/dev/null \
        | grep -i dev | head -1 | awk '{print $1}')
    [ -n "$search_pkg" ] && echo "$search_pkg" && return

    return 1
}

install_cmake_deps() {
    echo "==> Scanning source for cmake dependencies..."
    local deps
    deps=$(discover_cmake_deps)
    echo "Found cmake packages: $deps"

    local to_install=()
    for pkg in $deps; do
        # Skip packages cmake provides natively
        case "$pkg" in
            Threads|Git|UnixCommands|PackageHandleStandardArgs) continue ;;
        esac

        local apt_pkg
        apt_pkg=$(resolve_apt_pkg "$pkg" 2>/dev/null) || {
            echo "  SKIP: $pkg (no apt match)"
            continue
        }
        echo "  $pkg → $apt_pkg"
        to_install+=("$apt_pkg")
    done

    if [ ${#to_install[@]} -gt 0 ]; then
        echo "==> Installing ${#to_install[@]} packages..."
        apt-get install -y --no-install-recommends "${to_install[@]}" || true
    fi
}

run_upstream_deps() {
    cd "$WORKDIR/xdna-driver/tools"
    apt-get update -qq
    ./amdxdna_deps.sh -docker || true
}

build_xrt() {
    echo "==> Building XRT..."
    cd "$WORKDIR/xdna-driver/xrt/build"
    ./build.sh -npu -opt -noctest
}

build_xdna_driver() {
    echo "==> Building XDNA driver..."
    cd "$WORKDIR/xdna-driver/build"
    ./build.sh -release
    ./build.sh -package
}

install_debs() {
    mkdir -p "$WORKDIR/debs"
    cp "$WORKDIR/xdna-driver/build/Release/xrt_plugin"*amd64-amdxdna.deb "$WORKDIR/debs/"
    cp "$WORKDIR/xdna-driver/xrt/build/Release/xrt"*-amd64-base.deb "$WORKDIR/debs/"
    dpkg -i "$WORKDIR/debs/"*.deb || true
}

echo "=== Building XDNA driver ($DRIVER_VERSION) ==="
install_base
clone_source
run_upstream_deps
install_cmake_deps
build_xrt
build_xdna_driver
install_debs
echo "=== XDNA build complete ==="
