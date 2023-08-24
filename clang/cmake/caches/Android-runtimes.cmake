if(NOT NDK_PATH)
  message(FATAL_ERROR "NDK_PATH must point to NDK path")
endif()

set(LLVM_ENABLE_PROJECTS
  clang
  lld
  CACHE STRING "")

set(LLVM_ENABLE_RUNTIMES
  compiler-rt
  libunwind
  libcxxabi
  libcxx
  CACHE STRING "")

function(android_runtime arch api_level target)
  if(arch STREQUAL armv7)
    set(abi armeabi-v7a)
  elseif(arch STREQUAL aarch64)
    set(abi arm64-v8a)
  elseif(arch STREQUAL i686)
    set(abi x86)
  elseif(arch STREQUAL x86_64)
    set(abi x86_64)
  else()
    message(FATAL_ERROR "Invalid arch ${arch}")
  endif()

  set(toolchain_file "${CMAKE_SOURCE_DIR}/cmake/platforms/Android-runtimes.cmake")
  foreach(prefix BUILTINS RUNTIMES)
    set(${prefix}_${target}_ANDROID_ABI ${abi} CACHE STRING "")
    set(${prefix}_${target}_ANDROID_PLATFORM "android-${api_level}" CACHE STRING "")
    set(${prefix}_${target}_NDK_PATH "${NDK_PATH}" CACHE PATH "")
    set(${prefix}_${target}_CMAKE_TOOLCHAIN_FILE "${toolchain_file}" CACHE FILEPATH "")
  endforeach()

  # This matches the NDK. It's problematic if you have emulated atomics
  # used across shared libraries, but that should be very rare.
  set(BUILTINS_${target}_COMPILER_RT_EXCLUDE_ATOMIC_BUILTIN OFF CACHE BOOL "")

  set(RUNTIMES_${target}_CMAKE_CACHES
    "${CMAKE_CURRENT_LIST_DIR}/Android-libcxx.cmake"
    CACHE STRING "")
endfunction()

set(targets)
foreach(arch armv7 aarch64 i686 x86_64)
  # 21 is the minimum supported and 29 adds ELF TLS.
  foreach(api_level 21 29)
    set(target "${arch}-none-linux-android${api_level}")
    android_runtime(${arch} ${api_level} ${target})
    list(APPEND targets ${target})
  endforeach()
endforeach()

set(LLVM_BUILTIN_TARGETS ${targets} CACHE STRING "")
set(LLVM_RUNTIME_TARGETS ${targets} CACHE STRING "")
