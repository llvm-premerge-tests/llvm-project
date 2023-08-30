// RUN: mlir-opt -set-module-datalayout -convert-func-to-llvm='use-opaque-pointers=1' %s | FileCheck %s

// RUN-32: mlir-opt -set-module-datalayout='data-layout=p:32:32:32' -convert-func-to-llvm='use-opaque-pointers=1' %s \
// RUN-32: | FileCheck %s

// CHECK: module attributes {llvm.data_layout = ""}
// CHECK-32: module attributes {llvm.data_layout ="p:32:32:32"}
module {}
