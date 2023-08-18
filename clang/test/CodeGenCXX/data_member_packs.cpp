// RUN: %clang_cc1 --std=c++20 %s -emit-llvm -o - -triple x86_64-linux | FileCheck %s --check-prefixes=CHECK

// Tests declaration data member packs.
template<typename... Ts> struct S1 {
    Ts... ts;
};

template<typename T, typename... Ts> struct S2 {
    T t[2];
    Ts... ts;
};

template<typename T, typename... Ts> struct S3 {
    T t;
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
// Test duplicate types in template args.
// CHECK-NEXT: %struct.S1.4 = type { i32, i32 }
S1<int, int> s6;

S1<int, long, float, double> s7;

template<typename... Ts> auto sum(Ts... ts) {
  return (ts + ...);
}

template<typename T, typename... Ts> auto partial_sum(T t, Ts... ts) {
  return sum(ts...);
}

template<typename... Ts> auto partial_sum(S1<Ts...> s) {
  return partial_sum(s.ts...);
}

auto take_empty() {
  return 0;
}

template<typename... Ts> auto sum_pack(S1<Ts...> s) {
  return sum(s.ts...);
}
// Test template arg + expansion.
template<typename T, typename... Ts> auto sum_pack2(S1<T, Ts...> s) {
  return sum(s.ts...);
}
// Test empty expansion.
template<typename... Ts> auto take_empty(S3<Ts...> s) {
  return take_empty(s.ts...);
}
// Test nested template args and multiple expansions.
template<typename... Ts> struct S4 {
  template<typename... Ts2> auto sum_pack(S1<Ts..., Ts2...> s) {
    return sum(s.ts...);
  }
};

int main() {
  // Check calling take_empty()
  // CHECK: %call = call noundef i32 @_Z10take_emptyv()
  take_empty(S3<int>{});
  // Check instantiation of sum(int, float, double)
  // CHECK: double @_Z3sumIJifdEEDaDpT_(i32 noundef %ts, float noundef %ts1, double noundef %ts3)
  sum_pack(s2);
  // Check instantiation of sum(int, int)
  // CHECK: i32 @_Z3sumIJiiEEDaDpT_(i32 noundef %ts, i32 noundef %ts1)
  sum_pack2<int, int>(s6);
  // Check instantiation of sum(int, long, float, double)
  // CHECK: double @_Z3sumIJilfdEEDaDpT_(i32 noundef %ts, i64 noundef %ts1, float noundef %ts3, double noundef %ts5)
  S4<int, long>{}.sum_pack<float, double>(s7);
  // Check instantiation of sum(long, float, double)
  // CHECK: double @_Z3sumIJlfdEEDaDpT_(i64 noundef %ts, float noundef %ts1, double noundef %ts3)
  partial_sum(s7);
  return 0;
}

