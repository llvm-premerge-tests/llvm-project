// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -S -emit-llvm -o - %s | FileCheck %s

struct map_value {
        int __attribute__((btf_type_tag("tag1"))) __attribute__((btf_type_tag("tag3"))) *a;
        int __attribute__((btf_type_tag("tag2"))) __attribute__((btf_type_tag("tag4"))) *b;
};

struct map_value *func(void);

int test(struct map_value *arg)
{
        return *arg->a;
}

// CHECK: distinct !DICompositeType(tag: DW_TAG_structure_type, name: "map_value", file: ![[#]], line: [[#]], size: [[#]], elements: ![[L01:[0-9]+]])
// CHECK: ![[L01]] = !{![[L02:[0-9]+]], ![[L03:[0-9]+]]}
// CHECK: ![[L02]] = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L04:[0-9]+]], size: [[#]])
// CHECK: ![[L04]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L05:[0-9]+]], size: [[#]], annotations: ![[L06:[0-9]+]])
// CHECK: ![[L05]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed, annotations: ![[L07:[0-9]+]])
// CHECK: ![[L07]] = !{![[L08:[0-9]+]], ![[L09:[0-9]+]]}
// CHECK: ![[L08]] = !{!"btf:type_tag", !"tag1"}
// CHECK: ![[L09]] = !{!"btf:type_tag", !"tag3"}
// CHECK: ![[L06]] = !{![[L10:[0-9]+]], ![[L11:[0-9]+]]}
// CHECK: ![[L10]] = !{!"btf_type_tag", !"tag1"}
// CHECK: ![[L11]] = !{!"btf_type_tag", !"tag3"}
// CHECK: ![[L03]] = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: ![[#]], file: ![[#]], line: [[#]], baseType: ![[L12:[0-9]+]], size: [[#]], offset: [[#]])
// CHECK: ![[L12]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L13:[0-9]+]], size: [[#]], annotations: ![[L14:[0-9]+]])
// CHECK: ![[L13]] = !DIBasicType(name: "int", size: [[#]], encoding: DW_ATE_signed, annotations: ![[L15:[0-9]+]])
// CHECK: ![[L15]] = !{![[L16:[0-9]+]], ![[L17:[0-9]+]]}
// CHECK: ![[L16]] = !{!"btf:type_tag", !"tag2"}
// CHECK: ![[L17]] = !{!"btf:type_tag", !"tag4"}
// CHECK: ![[L14]] = !{![[L18:[0-9]+]], ![[L19:[0-9]+]]}
// CHECK: ![[L18]] = !{!"btf_type_tag", !"tag2"}
// CHECK: ![[L19]] = !{!"btf_type_tag", !"tag4"}
