// RUN: %clang_cc1 %s -triple x86_64-linux-unknown -emit-llvm -o - \
// RUN:   | FileCheck -check-prefix=HOST %s
// RUN: %clang_cc1 %s -fcuda-is-device \
// RUN:   -emit-llvm -o - -triple nvptx64 \
// RUN:   -aux-triple x86_64-unknown-linux-gnu | FileCheck \
// RUN:   -check-prefix=DEV %s

#include "Inputs/cuda.h"

// Check host/device-based overloding resolution in global variable initializer.
template<typename T, typename U>
T pow(T, U) { return 1.0; }

__device__ double pow(double, int) { return 2.0; }

// HOST-DAG: call {{.*}}double @_Z3powIdiET_S0_T0_(double noundef 1.000000e+00, i32 noundef 1)
double X = pow(1.0, 1);

template<typename T, typename U>
constexpr T cpow(T, U) { return 11.0; }

constexpr __device__ double cpow(double, int) { return 12.0; }

// HOST-DAG: @CX = global double 1.100000e+01
double CX = cpow(11.0, 1);

// DEV-DAG: @CY = addrspace(1) externally_initialized global double 1.200000e+01
__device__ double CY = cpow(12.0, 1);

struct A {
  template<typename T, typename U>
  T pow(T, U) { return 3.0; }

  __device__ double pow(double, int) { return 4.0; }
};

A a;

// HOST-DAG: call {{.*}}double @_ZN1A3powIdiEET_S1_T0_(ptr {{.*}}@a, double noundef 3.000000e+00, i32 noundef 1)
double AX = a.pow(3.0, 1);

struct CA {
  template<typename T, typename U>
  constexpr T cpow(T, U) const { return 13.0; }

  constexpr __device__ double cpow(double, int) const { return 14.0; }
};

const CA ca;

// HOST-DAG: @CAX = global double 1.300000e+01
double CAX = ca.cpow(13.0, 1);

// DEV-DAG: @CAY = addrspace(1) externally_initialized global double 1.400000e+01
__device__ double CAY = ca.cpow(14.0, 1);
