// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -o - %s | FileCheck %s

// An empty struct is handled as a struct with a dummy i8, on all targets.
// Most targets treat an empty struct return value as essentially void - but
// some don't. (Currently, at least x86_64-windows-* and powerpc64le-* don't
// treat it as void.)
//
// When intializing a struct with such a no_unique_address member, make sure we
// don't write the dummy i8 into the struct where there's no space allocated for
// it.
//
// This can only be tested with targets that don't treat empty struct returns as
// void.

struct S {};
S f();
struct Z {
  int x;
  [[no_unique_address]] S y;
  Z();
};
Z::Z() : x(111), y(f()) {}

// CHECK: define {{.*}} @_ZN1ZC2Ev

// CHECK: %coerce = alloca %struct.S, align 1

// CHECK: %call = call i8 @_Z1fv()
// CHECK-NEXT: %coerce.dive = getelementptr inbounds %struct.S, ptr %coerce, i32 0, i32 0
// CHECK-NEXT: store i8 %call, ptr %coerce.dive, align 1
// CHECK-NEXT: ret void
