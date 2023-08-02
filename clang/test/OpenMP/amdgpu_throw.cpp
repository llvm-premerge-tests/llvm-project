// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify=with -Wopenmp-target-exception
// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify=without -Wno-openmp-target-exception
// RUN: %clang_cc1 -fopenmp -triple amdgcn-amd-amdhsa -fopenmp-is-target-device %s -emit-llvm -S -verify=noexceptions

// noexceptions-error@9 {{cannot use 'throw' with exceptions disabled}}

#pragma omp declare target
void foo(void) {
	throw 404; // with-warning {{target 'amdgcn-amd-amdhsa' does not support exception handling; 'throw' is assumed to be never reached}}
}
#pragma omp end declare target
// without-no-diagnostics
