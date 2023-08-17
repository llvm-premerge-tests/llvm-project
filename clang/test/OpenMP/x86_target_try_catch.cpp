// RUN: %clang_cc1 -fopenmp -triple x86_64-pc-linux-gnu -fopenmp-is-target-device -fcxx-exceptions -fexceptions %s -emit-llvm -S -verify -Wopenmp-target-exception -o - &> /dev/null
#pragma omp declare target
int foo(void) {
	int error = -1;
	try {
		error = 1;
	}
	catch (int e){ 
		error = e;
	}
	return error;
}
#pragma omp end declare target
// expected-no-diagnostics
