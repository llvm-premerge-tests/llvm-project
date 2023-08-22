// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +zifencei \
// RUN:  -target-feature +m -target-feature +a               \
// RUN:  -target-feature +f -target-feature +d               \
// RUN:  -emit-llvm %s -o - | FileCheck %s                   \
// RUN:  --check-prefix=CHECK-IR

// CHECK-IR: void @test1() #0
__attribute__((target("arch=-a,+v,+zbb,+zicond1p0"))) void test1() {}

// CHECK-IR: void @test2() #1
__attribute__((target("arch=rv64gc_zbb"))) void test2 () {}

// CHECK-IR: void @test3() #2
__attribute__((target("cpu=rocket-rv64;tune=generic-rv64;arch=+v"))) void test3 () {}

// CHECK-IR: attributes #0 {{.*}}+experimental-zicond{{.*}}+v,+zbb{{.*}}+zve32f,+zve32x,+zve64d,+zve64f,+zve64x,+zvl128b,+zvl32b,+zvl64b,-a{{.*}}
// CHECK-IR: attributes #1 {{.*}}+c{{.*}}+zbb{{.*}}
// CHECK-IR: attributes #2 {{.*}} "target-cpu"="rocket-rv64" {{.*}}+v{{.*}} "tune-cpu"="generic-rv64" {{.*}}
