// RUN: %clang_cc1 --std=c++20 %s -emit-llvm -o - -triple x86_64-linux | FileCheck %s --check-prefixes=CHECK

// Tests declaration of packed data members.
template<typename... Ts> struct S1 {
    Ts... ts;
};

template<typename T, typename... Ts> struct S2 {
    T t[2];
    Ts... ts;
};

// CHECK: %struct.S1 = type { i32 }
S1<int> s1;
// CHECK-NEXT: %struct.S1.0 = type { i32, float, double }
S1<int, float, double> s2;
// Test template args as the last arg.
// CHECK-NEXT: %struct.S2 = type { [2 x i32], float, double }
S2<int, float, double> s3;
// Test nested template args.
// CHECK-NEXT: %struct.S1.1 = type { i32, float, %struct.S1.2 }
// CHECK-NEXT: %struct.S1.2 = type { double, double }
S1<int, float, S1<double, double>> s4;
// Test empty template arg.
// CHECK-NEXT: %struct.S1.3 = type { i8 }
S1<> s5;