// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -S -emit-llvm -o - %s | FileCheck %s

#define __tag1 __attribute__((btf_type_tag("tag1")))
#define __tag2 __attribute__((btf_type_tag("tag2")))

typedef void __fn_t(int);
typedef __fn_t __tag1 __tag2 *__fn2_t;
struct t {
  int __tag1 * __tag2 *a;
  __fn2_t b;
  long c;
};
int *foo1(struct t *a1) {
  return (int *)a1->c;
}

// CHECK: ![[L01:[0-9]+]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed)
// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "t", file: ![[#]], line: [[#]], size: [[#]], elements: ![[L02:[0-9]+]])
// CHECK: ![[L02]] = !{![[L03:[0-9]+]], ![[L04:[0-9]+]], ![[L05:[0-9]+]]}
// CHECK: ![[L03]] = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L06:[0-9]+]], size: [[#]])
// CHECK: ![[L06]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L07:[0-9]+]], size: [[#]], annotations: ![[L08:[0-9]+]])
// CHECK: ![[L07]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L09:[0-9]+]], size: [[#]], annotations: ![[L10:[0-9]+]])
// CHECK: ![[L09]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed, annotations: ![[L11:[0-9]+]])
// CHECK: ![[L11]] = !{![[L12:[0-9]+]]}
// CHECK: ![[L12]] = !{!"btf:type_tag", !"tag1"}
// CHECK: ![[L10]] = !{![[L13:[0-9]+]], ![[L14:[0-9]+]]}
// CHECK: ![[L13]] = !{!"btf:type_tag", !"tag2"}
// CHECK: ![[L14]] = !{!"btf_type_tag", !"tag1"}
// CHECK: ![[L08]] = !{![[L15:[0-9]+]]}
// CHECK: ![[L15]] = !{!"btf_type_tag", !"tag2"}
// CHECK: ![[L04]] = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L16:[0-9]+]], size: [[#]], offset: [[#]])
// CHECK: ![[L16]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__fn2_t", file: ![[#]], line: [[#]], baseType: ![[L17:[0-9]+]])
// CHECK: ![[L17]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L18:[0-9]+]], size: [[#]], annotations: ![[L19:[0-9]+]])
// CHECK: ![[L18]] = !DIDerivedType(tag: DW_TAG_typedef, name: "__fn_t", file: ![[#]], line: [[#]], baseType: ![[L20:[0-9]+]], annotations: ![[L21:[0-9]+]])
// CHECK: ![[L20]] = !DISubroutineType(types: ![[L22:[0-9]+]])
// CHECK: ![[L22]] = !{null, ![[L01]]}
// CHECK: ![[L21]] = !{![[L12]], ![[L13]]}
// CHECK: ![[L19]] = !{![[L14]], ![[L15]]}
// CHECK: ![[L05]] = !DIDerivedType(tag: DW_TAG_member, name: "c", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L23:[0-9]+]], size: [[#]], offset: [[#]])
// CHECK: ![[L23]] = !DIBasicType(name: "long", size: [[#]], encoding: DW_ATE_signed)
