// RUN: %clang_cc1 -std=c++2b %s -emit-llvm -triple x86_64-linux -o - | FileCheck %s

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


void test_lambda() {
    [](this auto This) -> int {
        return This();
    }();
}

//CHECK: define dso_local void @{{.*}}test_lambda{{.*}}() #0 {
//CHECK: entry:
//CHECK:  %agg.tmp = alloca %class.anon, align 1
//CHECK:  %ref.tmp = alloca %class.anon, align 1
//CHECK:  %call = call noundef i32 @"_ZZ11test_lambdavENH3$_0clIS_EEiT_"()
//CHECK:  ret void
//CHECK: }

//CHECK: define internal noundef i32 @"_ZZ11test_lambdavENH3$_0clIS_EEiT_"() #0 align 2 {
//CHECK: entry:
//CHECK:   %This = alloca %class.anon, align 1
//CHECK:   %agg.tmp = alloca %class.anon, align 1
//CHECK:   %call = call noundef i32 @"_ZZ11test_lambdavENH3$_0clIS_EEiT_"()
//CHECK:   ret i32 %call
//CHECK: }
