// This file is to test thread_limit clause on target prallel loop directive

// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -emit-llvm %s -o - | FileCheck --check-prefix=OMP51 %s

// expected-no-diagnostics

int thread_limit_target_parallel_loop() {

// Check that the offloading function is called after setting thread_limit in the task entry function
#pragma omp target parallel loop thread_limit(2)
    for(int i=0; i<2; i++) {}

// OMP51: define {{.*}}thread_limit_target_parallel_loop
// OMP51: call i32 [[OMP_TASK_ENTRY:@.+]](i32 {{.*}}%0, ptr %1)

// OMP51: define internal {{.*}}i32 [[OMP_TASK_ENTRY]](i32 {{.*}}%0, ptr noalias noundef %1)
// OMP51: call void @__kmpc_set_thread_limit(ptr @{{.+}}, i32 %{{.+}}, i32 2)
// OMP51: call void {{.*omp_offloading.*thread_limit_target_parallel_loop.*}}
  return 0;
}