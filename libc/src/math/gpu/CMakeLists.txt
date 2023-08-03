# Math functions not yet available in the libc project, or those not yet tuned
# for GPU workloads are provided as wrappers over vendor libraries. If we find
# them ahead of time we will import them statically. Otherwise, we will keep
# them as external references and expect them to be resolved by the user when
# they compile. In the future,we will use implementations from the 'libc'
# project and not provide these wrappers.
add_subdirectory(vendor)

# For the GPU we want to be able to optionally depend on the vendor libraries
# until we have a suitable replacement inside `libc`.
# TODO: We should have an option to enable or disable these on a per-function
# basis.
option(LIBC_GPU_VENDOR_MATH "Use vendor wrappers for GPU math" ON)

# As a temporary solution to the above problem, we can add another CMake option
# that allows the user to select between generic and built-in versions of the
# math functions.
option(LIBC_GPU_BUILTIN_MATH "Use built-in wrappers for GPU math" OFF)

# And these options are regarded as mutually exclusive.
if (${LIBC_GPU_VENDOR_MATH} AND ${LIBC_GPU_BUILTIN_MATH})
  message(WARNING "\
    Invalid CMake configuration. Please do not enable both \
    'LIBC_GPU_VENDOR_MATH' and 'LIBC_GPU_BUILTIN_MATH'.\
  ")
endif()

function(add_math_entrypoint_gpu_object name)
  get_fq_target_name("vendor.${name}" fq_vendor_specific_target_name)
  if(TARGET ${fq_vendor_specific_target_name} AND ${LIBC_GPU_VENDOR_MATH} 
    AND NOT ${LIBC_GPU_BUILTIN_MATH})
    return()
  endif()

  # If the user has not specified to use vendor math and has also not specified
  # to use built-in math, we check wether a generic implementation exists. If a
  # generic version exists, we do not add a built-in entry point.
  if(NOT ${LIBC_GPU_BUILTIN_MATH})
    get_filename_component(generic_file_path
      "${CMAKE_CURRENT_LIST_DIR}/../generic/${name}.cpp" ABSOLUTE)
    if (EXISTS ${generic_file_path})
      return()
    endif()
  endif()

  add_entrypoint_object(
    ${name}
    SRCS
      ${name}.cpp
    HDRS
      ../${name}.h
    COMPILE_OPTIONS
      -O2
  )
endfunction()

add_math_entrypoint_gpu_object(acos)
add_math_entrypoint_gpu_object(acosf)
add_math_entrypoint_gpu_object(acosh)
add_math_entrypoint_gpu_object(acoshf)
add_math_entrypoint_gpu_object(asin)
add_math_entrypoint_gpu_object(asinf)
add_math_entrypoint_gpu_object(asinh)
add_math_entrypoint_gpu_object(asinhf)
add_math_entrypoint_gpu_object(atan)
add_math_entrypoint_gpu_object(atanf)
add_math_entrypoint_gpu_object(atan2)
add_math_entrypoint_gpu_object(atan2f)
add_math_entrypoint_gpu_object(atanh)
add_math_entrypoint_gpu_object(atanhf)
add_math_entrypoint_gpu_object(ceil)
add_math_entrypoint_gpu_object(ceilf)
add_math_entrypoint_gpu_object(copysign)
add_math_entrypoint_gpu_object(copysignf)
add_math_entrypoint_gpu_object(cos)
add_math_entrypoint_gpu_object(cosf)
add_math_entrypoint_gpu_object(cosh)
add_math_entrypoint_gpu_object(coshf)
add_math_entrypoint_gpu_object(erf)
add_math_entrypoint_gpu_object(erff)
add_math_entrypoint_gpu_object(exp)
add_math_entrypoint_gpu_object(exp2)
add_math_entrypoint_gpu_object(exp2f)
add_math_entrypoint_gpu_object(expf)
add_math_entrypoint_gpu_object(expm1)
add_math_entrypoint_gpu_object(expm1f)
add_math_entrypoint_gpu_object(fabs)
add_math_entrypoint_gpu_object(fabsf)
add_math_entrypoint_gpu_object(floor)
add_math_entrypoint_gpu_object(floorf)
add_math_entrypoint_gpu_object(fma)
add_math_entrypoint_gpu_object(fmaf)
add_math_entrypoint_gpu_object(fmax)
add_math_entrypoint_gpu_object(fmaxf)
add_math_entrypoint_gpu_object(fmin)
add_math_entrypoint_gpu_object(fminf)
add_math_entrypoint_gpu_object(fmod)
add_math_entrypoint_gpu_object(fmodf)
add_math_entrypoint_gpu_object(frexp)
add_math_entrypoint_gpu_object(frexpf)
add_math_entrypoint_gpu_object(hypot)
add_math_entrypoint_gpu_object(hypotf)
add_math_entrypoint_gpu_object(ldexp)
add_math_entrypoint_gpu_object(ldexpf)
add_math_entrypoint_gpu_object(llrint)
add_math_entrypoint_gpu_object(llrintf)
add_math_entrypoint_gpu_object(llround)
add_math_entrypoint_gpu_object(llroundf)
add_math_entrypoint_gpu_object(log10)
add_math_entrypoint_gpu_object(log10f)
add_math_entrypoint_gpu_object(log1p)
add_math_entrypoint_gpu_object(log1pf)
add_math_entrypoint_gpu_object(log2)
add_math_entrypoint_gpu_object(log2f)
add_math_entrypoint_gpu_object(log)
add_math_entrypoint_gpu_object(logf)
add_math_entrypoint_gpu_object(lrint)
add_math_entrypoint_gpu_object(lrintf)
add_math_entrypoint_gpu_object(lround)
add_math_entrypoint_gpu_object(lroundf)
add_math_entrypoint_gpu_object(modf)
add_math_entrypoint_gpu_object(modff)
add_math_entrypoint_gpu_object(nearbyint)
add_math_entrypoint_gpu_object(nearbyintf)
add_math_entrypoint_gpu_object(pow)
add_math_entrypoint_gpu_object(powf)
add_math_entrypoint_gpu_object(remainder)
add_math_entrypoint_gpu_object(remainderf)
add_math_entrypoint_gpu_object(remquo)
add_math_entrypoint_gpu_object(remquof)
add_math_entrypoint_gpu_object(rint)
add_math_entrypoint_gpu_object(rintf)
add_math_entrypoint_gpu_object(round)
add_math_entrypoint_gpu_object(roundf)
add_math_entrypoint_gpu_object(scalbn)
add_math_entrypoint_gpu_object(scalbnf)
add_math_entrypoint_gpu_object(sin)
add_math_entrypoint_gpu_object(sinf)
add_math_entrypoint_gpu_object(sinh)
add_math_entrypoint_gpu_object(sinhf)
add_math_entrypoint_gpu_object(sqrt)
add_math_entrypoint_gpu_object(sqrtf)
add_math_entrypoint_gpu_object(tan)
add_math_entrypoint_gpu_object(tanf)
add_math_entrypoint_gpu_object(tanh)
add_math_entrypoint_gpu_object(tanhf)
add_math_entrypoint_gpu_object(tgamma)
add_math_entrypoint_gpu_object(tgammaf)
add_math_entrypoint_gpu_object(trunc)
add_math_entrypoint_gpu_object(truncf)
