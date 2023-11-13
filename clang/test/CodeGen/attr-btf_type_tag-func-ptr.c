// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -S -emit-llvm -o - %s | FileCheck %s

struct t {
 int (__attribute__((btf_type_tag("rcu"))) *f)();
 int a;
};
int foo(struct t *arg) {
  return arg->a;
}

// CHECK: !DIDerivedType(tag: DW_TAG_member, name: "f", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L1:[0-9]+]], size: [[#]])
// CHECK: ![[L1]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L2:[0-9]+]], size: [[#]], annotations: ![[L3:[0-9]+]])
// CHECK: ![[L2]] = !DISubroutineType(types: ![[#]], annotations: ![[L4:[0-9]+]])
// CHECK: ![[L4]] = !{![[L5:[0-9]+]]}
// CHECK: ![[L5]] = !{!"btf:type_tag", !"rcu"}
// CHECK: ![[L3]] = !{![[L6:[0-9]+]]}
// CHECK: ![[L6]] = !{!"btf_type_tag", !"rcu"}
