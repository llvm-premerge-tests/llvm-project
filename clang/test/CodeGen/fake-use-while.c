// RUN: %clang_cc1 %s -S -emit-llvm -fextend-lifetimes -o %t.ll
//
// Check we don't assert when there is no more code after a while statement
// and the body of the while statement ends in a return, i.e. no insertion point
// is available.

// CHECK: define{{.*}}main
// CHECK: call{{.*}}llvm.fake.use

void foo() {
  {
    while (1) {
      int ret;
      if (1)
        return;
    }
  }
}
