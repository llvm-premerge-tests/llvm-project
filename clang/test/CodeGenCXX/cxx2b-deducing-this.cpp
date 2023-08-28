// RUN: %clang_cc1 -std=c++2b %s -emit-llvm -triple x86_64-linux -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++2b %s -emit-llvm -triple x86_64-windows-msvc -o - | FileCheck %s

struct TrivialStruct {
    void explicit_object_function(this TrivialStruct) {}
};
void test() {
    TrivialStruct s;
    s.explicit_object_function();
}
// CHECK:      define {{.*}}test{{.*}}
// CHECK-NEXT: entry:
// CHECK:      {{.*}} = alloca %struct.TrivialStruct, align 1
// CHECK:      {{.*}} = alloca %struct.TrivialStruct, align 1
// CHECK:      call void {{.*}}explicit_object_function{{.*}}
// CHECK-NEXT: ret void
// CHECK-NEXT: }

// CHECK:      define {{.*}}explicit_object_function{{.*}}
// CHECK-NEXT: entry:
// CHECK:        {{.*}} = alloca %struct.TrivialStruct, align 1
// CHECK:        ret void
// CHECK-NEXT: }

struct S {
  void foo(this const S&);
};
struct T {
  S bar(this const T&);
};
void chain_test() {
  T t;
  t.bar().foo();
}

// CHECK: define {{.*}}chain_test{{.*}}
// CHECK-NEXT: entry:
// CHECK: {{.*}} = alloca %struct.T, align 1
// CHECK: {{.*}} = alloca %struct.S, align 1
// CHECK: %call = call i8 @"?bar@T@@SA?AUS@@_VAEBU1@@Z"{{.*}}
// CHECK: %coerce.dive = getelementptr inbounds %struct.S, {{.*}} %{{.*}}, i32 0, i32 0
// CHECK  store i8 %call, ptr %coerce.dive, align 1
// CHECK: call void @"?foo@S@@SAX_VAEBU1@@Z"
// CHECK: ret void
