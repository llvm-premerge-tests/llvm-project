// RUN: %clang_cc1 -triple arm-none-eabi -fdump-record-layouts-simple -ffreestanding -emit-llvm -o %t %s | FileCheck %s -check-prefixes=LAYOUT
// RUN: <%t FileCheck %s -check-prefixes=IR
// RUN: %clang_cc1 -triple aarch64 -fdump-record-layouts-simple -ffreestanding -emit-llvm -o %t %s | FileCheck %s -check-prefixes=LAYOUT
// RUN: <%t FileCheck %s -check-prefixes=IR

extern struct T {
  int b0 : 8;
  int b1 : 24;
  int b2 : 1;
} g;

int func(void) {
  return g.b1;
}

// IR: @g = external global %struct.T, align 4
// IR: %{{.*}} = load i64, ptr @g, align 4

// LAYOUT-LABEL: Record: RecordDecl {{.*}} struct T definition
// LAYOUT: Layout: <CGRecordLayout
// LAYOUT-NEXT: LLVMType:%struct.T = type { i40 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:8 Size:24 IsSigned:1 StorageSize:64 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:32 Size:1 IsSigned:1 StorageSize:64 StorageOffset:0
// LAYOUT-NEXT: ]>
