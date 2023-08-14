#!/usr/bin/env bash
#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

# Build the builtins archive and libunwind.a for Android and add them to a copy
# of an existing Clang resource directory. Then generate clang/clang++ wrapper
# scripts that use the new resource directory.

set -ex

MONOREPO_ROOT=
OUTPUT_DIR=
CMAKE=
NINJA=
ARCH=
while [[ $# -gt 0 ]]; do
    case ${1} in
        --llvm-root)
            MONOREPO_ROOT="${2}"
            shift; shift
            ;;
        --output-dir)
            OUTPUT_DIR="${2}"
            shift; shift
            ;;
        --cmake)
            CMAKE="${2}"
            shift; shift
            ;;
        --ninja)
            NINJA="${2}"
            shift; shift
            ;;
        --arch)
            ARCH="${2}"
            shift; shift
            ;;
        *)
            echo "error: unrecognized argument: ${1}" >&2
            exit 1
            ;;
    esac
done

MONOREPO_ROOT="${MONOREPO_ROOT:-"$(git rev-parse --show-toplevel)"}"
OUTPUT_DIR="${OUTPUT_DIR:-${MONOREPO_ROOT}/build/android-toolchain}"
CMAKE="${CMAKE:-cmake}"
NINJA="${NINJA:-ninja}"
ARCH="${ARCH:-x86_64}"

case "${ARCH}" in
    arm)    RES_ARCH_SUBDIR=arm ;;
    arm64)  RES_ARCH_SUBDIR=aarch64 ;;
    x86)    RES_ARCH_SUBDIR=i386 ;;
    x86_64) RES_ARCH_SUBDIR=x86_64 ;;
    *) echo "error: unhandled arch"; exit 1 ;;
esac

rm -rf "${OUTPUT_DIR}"

# Default CC/CXX compilers. The LLVM_HEAD_VERSION variable is set both by
# the Docker image and by the buildkite yml file.
export CC=${CC:-clang-${LLVM_HEAD_VERSION}}
export CXX=${CXX:-clang++-${LLVM_HEAD_VERSION}}

# Duplicate the resource directory and replace CC/CXX with wrapper scripts
# using the new directory.
RESOURCE_DIR="${OUTPUT_DIR}/patched_res_dir"
mkdir -p "${OUTPUT_DIR}"
cp -r "$(${CC} -print-resource-dir)" "${RESOURCE_DIR}"
mkdir -p "${OUTPUT_DIR}/bin"

make_wrapper_script() {
    # The compilers from apt.llvm.org are built using
    # -DCLANG_DEFAULT_RTLIB=libgcc, which overrides the ordinary Android default
    # of -rtlib=compiler-rt --unwindlib=libunwind, so restore the normal default
    # in the wrapper scripts.
    local NAME="${1}"
    shift
    printf "#!/bin/bash\n${@} \
        --start-no-unused-arguments \
        -resource-dir \"${RESOURCE_DIR}\" \
        -rtlib=compiler-rt --unwindlib=libunwind \
        --end-no-unused-arguments \
        \"\${@}\"\n" >"${OUTPUT_DIR}/bin/${NAME}"
    chmod +x "${OUTPUT_DIR}/bin/${NAME}"
}

# N.B. ${OUTPUT_DIR} cannot have a space in it, because a space in CC/CXX
# always delimits arguments.
make_wrapper_script clang ${CC}
make_wrapper_script clang++ ${CXX}
export CC="${OUTPUT_DIR}/bin/clang"
export CXX="${OUTPUT_DIR}/bin/clang++"

# Build the builtins archive and install it into the patched resource
# directory.
${CMAKE} -GNinja -DCMAKE_MAKE_PROGRAM="${NINJA}" \
         -S "${MONOREPO_ROOT}/compiler-rt/lib/builtins" \
         -B "${OUTPUT_DIR}/builtins" \
         -C "${MONOREPO_ROOT}/runtimes/cmake/android/Arch-${ARCH}.cmake" \
         -C "${MONOREPO_ROOT}/compiler-rt/cmake/caches/AndroidBuiltins.cmake" \
         -DCMAKE_INSTALL_PREFIX="${RESOURCE_DIR}" \
         -DCMAKE_SYSROOT=/opt/android/ndk/sysroot
${NINJA} -vC "${OUTPUT_DIR}/builtins" install-builtins

# Build libunwind.a and install it into the patched resource directory, which is
# also where the NDK installs it. Building libunwind in the same CMake
# invocation as libc++abi and libc++ almost works using these options:
#
#     -D LIBUNWIND_HIDE_SYMBOLS=ON
#     -D LIBUNWIND_ENABLE_SHARED=OFF
#     -D LIBCXXABI_USE_LLVM_UNWINDER=ON
#     -D LIBCXXABI_ENABLE_STATIC_UNWINDER=ON
#     -D LIBCXX_STATICALLY_LINK_ABI_IN_SHARED_LIBRARY=ON
#
# The biggest problem is that LIBUNWIND_HIDE_SYMBOLS doesn't hide the unwinder
# symbols because it only applies to libunwind.a, and CMake skips the archive
# and links the unwinder objects directly into libc++[_shared].so.
#
# This set of options also includes libunwind's objects in libc++abi.a, which
# isn't appropriate for Android. Android binaries built by the NDK link against
# the unwinder using the default --unwindlib=libunwind driver option. For the
# Android platform, libc.so exports the C++ EH unwinder API as of API 30, so
# binaries use libc.so instead of libunwind. At some point, the NDK might switch
# to libc.so as well.
${CMAKE} -GNinja -DCMAKE_MAKE_PROGRAM="${NINJA}" \
         -S "${MONOREPO_ROOT}/runtimes" \
         -B "${OUTPUT_DIR}/libunwind" \
         -C "${MONOREPO_ROOT}/runtimes/cmake/android/Arch-${ARCH}.cmake" \
         -C "${MONOREPO_ROOT}/libunwind/cmake/caches/Android.cmake" \
         -DCMAKE_INSTALL_PREFIX=${OUTPUT_DIR}/libunwind-install \
         -DCMAKE_SYSROOT=/opt/android/ndk/sysroot
${NINJA} -vC "${OUTPUT_DIR}/libunwind" install
mkdir -p "${RESOURCE_DIR}/lib/linux/${RES_ARCH_SUBDIR}"
cp "${OUTPUT_DIR}/libunwind-install/lib/libunwind.a" \
   "${RESOURCE_DIR}/lib/linux/${RES_ARCH_SUBDIR}"
