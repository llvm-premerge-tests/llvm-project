// RUN: %clang_cc1 -fopenmp -triple nvptx64 -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify=with -Wopenmp-target-exception
// RUN: %clang_cc1 -fopenmp -triple nvptx64 -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify=without -Wno-openmp-target-exception
// RUN: %clang_cc1 -fopenmp -triple nvptx64 -fopenmp-is-target-device %s -emit-llvm -S -verify=noexceptions

// noexceptions-error@11 {{cannot use 'try' with exceptions disabled}}
// noexceptions-error@12 {{cannot use 'throw' with exceptions disabled}}

#pragma omp declare target
int foo(void) {
	int error = -1;
	try { // with-warning {{target 'nvptx64' does not support exception handling; 'catch' block is ignored}}
		throw 404; // with-warning {{target 'nvptx64' does not support exception handling; 'throw' is assumed to be never reached}}
	}
	catch (int e){ 
		error = e;
	}
	return error;
}
#pragma omp end declare target
// without-no-diagnostics
