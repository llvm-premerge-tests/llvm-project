// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-win32 -regcall4 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-mingw32 -regcall4 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-pc-windows-msvc-elf -regcall4 | FileCheck %s --check-prefix=ELF32
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-win32 -regcall4 | FileCheck %s --check-prefix=X64
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-mingw32 -regcall4 | FileCheck %s --check-prefix=X64
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-pc-windows-msvc-elf -regcall4 | FileCheck %s --check-prefix=ELF64

// CHECK: target datalayout = "e-m:x-{{.*}}"
// X64: target datalayout = "e-m:w-{{.*}}"
// ELF32: target datalayout = "e-m:e-{{.*}}"
// ELF64: target datalayout = "e-m:e-{{.*}}"

void __regcall v1(void) {}
// CHECK: define dso_local x86_regcallcc void @__regcall4__v1
// X64: define dso_local x86_regcallcc void @__regcall4__v1
// ELF32: define{{.*}} x86_regcallcc void @__regcall4__v1
// ELF64: define{{.*}} x86_regcallcc void @__regcall4__v1

void __regcall v2(char a) {}
// CHECK: define dso_local x86_regcallcc void @__regcall4__v2
// X64: define dso_local x86_regcallcc void @__regcall4__v2
// ELF32: define{{.*}} x86_regcallcc void @__regcall4__v2
// ELF64: define{{.*}} x86_regcallcc void @__regcall4__v2

void __regcall v3(short a) {}
// CHECK: define dso_local x86_regcallcc void @__regcall4__v3
// X64: define dso_local x86_regcallcc void @__regcall4__v3

void __regcall v4(int a) {}
// CHECK: define dso_local x86_regcallcc void @__regcall4__v4
// X64: define dso_local x86_regcallcc void @__regcall4__v4

void __regcall v5(long long a) {}
// CHECK: define dso_local x86_regcallcc void @__regcall4__v5
// X64: define dso_local x86_regcallcc void @__regcall4__v5

void __regcall v6(char a, char b) {}
// CHECK: define dso_local x86_regcallcc void @__regcall4__v6
// X64: define dso_local x86_regcallcc void @__regcall4__v6

void __regcall v7(long long a, char b, char c, short d) {}
// CHECK: define dso_local x86_regcallcc void @__regcall4__v7(i64 noundef %a, i8 inreg noundef signext %b, i8 inreg noundef signext %c, i16 inreg noundef signext %d)