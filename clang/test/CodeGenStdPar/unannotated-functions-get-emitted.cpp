// RUN: %clang_cc1 -x hip -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck --check-prefix=NO-STDPAR-DEV %s

// RUN: %clang_cc1 --stdpar -emit-llvm -fcuda-is-device \
// RUN:   -o - %s | FileCheck --check-prefix=STDPAR-DEV %s

#define __device__ __attribute__((device))

// NO-STDPAR-DEV-NOT: define {{.*}} void @_Z3fooPff({{.*}})
// STDPAR-DEV: define {{.*}} void @_Z3fooPff({{.*}})
void foo(float *a, float b) {
  *a = b;
}

// NO-STDPAR-DEV: define {{.*}} void @_Z3barPff({{.*}})
// STDPAR-DEV: define {{.*}} void @_Z3barPff({{.*}})
__device__ void bar(float *a, float b) {
  *a = b;
}
