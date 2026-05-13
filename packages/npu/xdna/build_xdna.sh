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
        pciutils dkms apt-file python3-dev
    apt-file update 2>/dev/null || true
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

# Map cmake package name → apt package
resolve_apt_pkg() {
    local cmake_name="$1"
    local lower=$(echo "$cmake_name" | tr '[:upper:]' '[:lower:]')

    # Try common -dev package naming first (most reliable)
    for try in "lib${lower}-dev" "${lower}-dev" "lib${lower}"; do
        apt-cache show "$try" &>/dev/null && echo "$try" && return
    done

    # Try cmake config file search (skip FindXxx - that's always cmake-data)
    local config_pkg
    config_pkg=$(apt-file search "${lower}-config.cmake" 2>/dev/null \
        | grep -v cmake-data | head -1 | cut -d: -f1)
    [ -n "$config_pkg" ] && echo "$config_pkg" && return
    config_pkg=$(apt-file search "${cmake_name}Config.cmake" 2>/dev/null \
        | grep -v cmake-data | head -1 | cut -d: -f1)
    [ -n "$config_pkg" ] && echo "$config_pkg" && return

    # Try apt-cache search as last resort
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
