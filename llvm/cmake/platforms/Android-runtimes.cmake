# A wrapper around the NDK's CMake toolchain that uses the just-built
# compiler instead of the NDK's and tweaks flags to work with a
# partially built toolchain, for the runtimes build.

if(NOT NDK_PATH)
  message(FATAL_ERROR "NDK_PATH must point to NDK path")
endif()

# These are passed down from the runtimes build.
set(ORIGINAL_C_COMPILER "${CMAKE_C_COMPILER}")
set(ORIGINAL_CXX_COMPILER "${CMAKE_CXX_COMPILER}")

include("${NDK_PATH}/build/cmake/android.toolchain.cmake")

set(CMAKE_C_COMPILER "${ORIGINAL_C_COMPILER}")
set(CMAKE_CXX_COMPILER "${ORIGINAL_CXX_COMPILER}")

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES NDK_PATH)

# We won't have libunwind available since we're building it. CMake's ABI
# checks fail without this.
# FIXME: compiler-rt uses --unwindlib=none already; does that just need
# to be more comprehensive, or is the "Detecting compiler ABI info"
# check ingrained enough in CMake that only the toolchain file can
# influence it?
string(APPEND CMAKE_EXE_LINKER_FLAGS " --unwindlib=none")
string(APPEND CMAKE_MODULE_LINKER_FLAGS " --unwindlib=none")
string(APPEND CMAKE_SHARED_LINKER_FLAGS " --unwindlib=none")

# libatomic is provided by compiler-rt and our just-built compiler
# doesn't have the NDK's dummy libatomic available.
string(REPLACE "-latomic" "" CMAKE_C_STANDARD_LIBRARIES_INIT "${CMAKE_C_STANDARD_LIBRARIES_INIT}")
string(REPLACE "-latomic" "" CMAKE_CXX_STANDARD_LIBRARIES_INIT "${CMAKE_CXX_STANDARD_LIBRARIES_INIT}")

# Fortify doesn't play well with -ffreestanding, which is used heavily
# by sanitizers, because e.g. it prevents Clang's built-in limits.h from
# reaching out to the NDK's and supplying SSIZE_MAX.
# FIXME: should this be in compiler-rt instead?
string(REPLACE "-D_FORTIFY_SOURCE=2" "" CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS}")
string(REPLACE "-D_FORTIFY_SOURCE=2" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
string(REPLACE "-D_FORTIFY_SOURCE=2" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# Clang canonicalizes androideabi as just android, so we need to mimic
# that to get the correct per-target installation directory.
# FIXME: remove this once https://reviews.llvm.org/D140925 lands
string(REPLACE "androideabi" "android" CMAKE_ASM_COMPILER_TARGET "${CMAKE_ASM_COMPILER_TARGET}")
string(REPLACE "androideabi" "android" CMAKE_C_COMPILER_TARGET "${CMAKE_C_COMPILER_TARGET}")
string(REPLACE "androideabi" "android" CMAKE_CXX_COMPILER_TARGET "${CMAKE_CXX_COMPILER_TARGET}")
