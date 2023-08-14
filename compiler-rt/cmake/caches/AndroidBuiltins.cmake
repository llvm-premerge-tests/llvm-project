set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")

# Android does not have a separate libatomic (static or shared). The atomic APIs
# are provided by the builtins library.
set(COMPILER_RT_EXCLUDE_ATOMIC_BUILTIN OFF CACHE BOOL "")
