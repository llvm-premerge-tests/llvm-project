// RUN: mlir-opt -convert-func-to-llvm='use-opaque-pointers=1' -set-module-datalayout %s | FileCheck %s

// RUN-32: mlir-opt -convert-func-to-llvm='use-opaque-pointers=1' -set-module-datalayout='data-layout=p:32:32:32' %s \
// RUN-32: | FileCheck %s

// CHECK: module attributes {llvm.data_layout = ""}
// CHECK-32: module attributes {llvm.data_layout ="p:32:32:32"}
module {}
