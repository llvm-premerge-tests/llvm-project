// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple %itanium_abi_triple -DDOUBLE_BRACKET_ATTRS=1 -debug-info-kind=limited -S -emit-llvm -o - %s | FileCheck %s

#if DOUBLE_BRACKET_ATTRS
#define __tag1 [[clang::btf_type_tag("tag1")]]
#define __tag2 [[clang::btf_type_tag("tag2")]]
#define __tag3 [[clang::btf_type_tag("tag3")]]
#define __tag4 [[clang::btf_type_tag("tag4")]]
#else
#define __tag1 __attribute__((btf_type_tag("tag1")))
#define __tag2 __attribute__((btf_type_tag("tag2")))
#define __tag3 __attribute__((btf_type_tag("tag3")))
#define __tag4 __attribute__((btf_type_tag("tag4")))
#endif

int __tag1 * __tag2 *foo(int __tag1 * __tag2 *arg) { return arg; }

// CHECK: distinct !DISubprogram(name: "foo", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[L01:[0-9]+]], {{.*}})
// CHECK: ![[L01]] = !DISubroutineType(types: ![[L02:[0-9]+]])
// CHECK: ![[L02]] = !{![[L03:[0-9]+]], ![[L03]]}
// CHECK: ![[L03]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L04:[0-9]+]], size: [[#]], annotations: ![[L05:[0-9]+]])
// CHECK: ![[L04]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L06:[0-9]+]], size: [[#]], annotations: ![[L07:[0-9]+]])
// CHECK: ![[L06]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed, annotations: ![[L08:[0-9]+]])
// CHECK: ![[L08]] = !{![[L09:[0-9]+]]}
// CHECK: ![[L09]] = !{!"btf:type_tag", !"tag1"}
// CHECK: ![[L07]] = !{![[L10:[0-9]+]], ![[L11:[0-9]+]]}
// CHECK: ![[L10]] = !{!"btf:type_tag", !"tag2"}
// CHECK: ![[L11]] = !{!"btf_type_tag", !"tag1"}
// CHECK: ![[L05]] = !{![[L12:[0-9]+]]}
// CHECK: ![[L12]] = !{!"btf_type_tag", !"tag2"}
// CHECK: !DILocalVariable(name: "arg", arg: 1, scope: ![[#]], file: ![[#]], line: [[#]], type: ![[L03]])
