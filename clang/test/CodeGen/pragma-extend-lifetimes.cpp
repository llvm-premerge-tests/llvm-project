// RUN: %clang %s --target=%itanium_abi_triple -S -O2 -emit-llvm -o - \
// RUN:   | FileCheck %s --check-prefix=NO-EXTENSION --implicit-check-not="call void (...) @llvm.fake.use"
// Check that we do not generate fake_use instructions without -fextend-this-ptr
// or -fextend-lifetimes, even with `#pragma clang extend_lifetimes enable`
// declared.

// RUN: %clang %s --target=%itanium_abi_triple -S -O2 -emit-llvm -fextend-this-ptr -o - \
// RUN:   | FileCheck %s --implicit-check-not="call void (...) @llvm.fake.use"
// Check that we generate a fake_use instruction for this 'this' ptr in foo()
// and baz(), but not for bar().

// RUN: %clang %s --target=%itanium_abi_triple -S -O2 -emit-llvm -fextend-lifetimes -o - \
// RUN:   | FileCheck %s --check-prefixes CHECK,ALL --implicit-check-not="call void (...) @llvm.fake.use"
// Check that we generate a fake_use instruction for all parameters (including
// the 'this' ptr) in foo() and baz(), but not for bar().

class A {
  void foo(int i);
  void bar(int i);
  void bat(int i);
  void baz(int i);
  void goo(int i);
  void moo(int i);
  void zoo(int i);
  void fie(int i);
};

// NO-EXTENSION: define{{.*}}foo
void A::foo(int i) {
  // CHECK-LABEL: define{{.*}}foo
  // ALL:         call void (...) @llvm.fake.use(i32 %
  // CHECK:       call void (...) @llvm.fake.use(ptr nonnull %
}

#pragma clang extend_lifetimes disable
void A::bar(int i) {
  // CHECK-LABEL: define{{.*}}bar
  // CHECK:       ret
}

void A::bat(int i) {
// Make sure that a disabling pragma spans multiple function definitions.
  // CHECK-LABEL: define{{.*}}bat
  // CHECK:       ret
}

#pragma clang extend_lifetimes enable
void A::baz(int i) {
  // CHECK-LABEL: define{{.*}}baz
  // ALL:         call void (...) @llvm.fake.use(i32 %
  // CHECK:       call void (...) @llvm.fake.use(ptr nonnull %
}

void A::goo(int i) {
// Make sure that a disabling pragma has no effect on the current function
// when it appears in the middle.
#pragma clang extend_lifetimes disable
  // CHECK-LABEL: define{{.*}}goo
  // ALL:         call void (...) @llvm.fake.use(i32 %
  // CHECK:       call void (...) @llvm.fake.use(ptr nonnull %
}

void A::moo(int i) {
// Make sure
// 1) that a disabling pragma that appeared in the middle of the previous
// function affects the current function.
// 2) that an enabling pragma that appears in the middle of the current
// function does not affect the current function.
#pragma clang extend_lifetimes enable
  // CHECK-LABEL: define{{.*}}moo
  // CHECK:       ret
}

void A::zoo(int i) {
// Finally, make sure that an enabling pragma that appeared in the middle of
// the previous function affects the current function.
  // CHECK-LABEL: define{{.*}}zoo
  // ALL:         call void (...) @llvm.fake.use(i32 %
  // CHECK:       call void (...) @llvm.fake.use(ptr nonnull %
}

void __attribute__((nodebug)) A::fie(int i) {
  int j = 0;
// Make sure that the nodebug attribute prevents extend-lifetimes from
// taking effect.
  // CHECK-LABEL: define{{.*}}fie
}
