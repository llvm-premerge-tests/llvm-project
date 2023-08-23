// RUN: %clang_cc1 -triple x86_64-unknown-unknown -O1 -emit-llvm -o - -disable-llvm-passes %s | FileCheck %s

// CHECK: @i = unnamed_addr constant i32 8

[[clang::unnamed_addr]] const int i = 8;
