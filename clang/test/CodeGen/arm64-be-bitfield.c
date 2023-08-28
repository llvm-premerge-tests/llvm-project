// RUN:  %clang_cc1 -triple aarch64_be-linux-gnu -ffreestanding -emit-llvm -O0 -o %t -fdump-record-layouts-simple %s | FileCheck %s
// RUN: <%t FileCheck %s --check-prefix=IR

struct bt3 { signed b2:10; signed b3:10; } b16;

// Get the high 32-bits and then shift appropriately for big-endian.
signed callee_b0f(struct bt3 bp11) {
// IR: callee_b0f(i64 [[ARG:%.*]])
// IR: store i64 [[ARG]], ptr [[PTR:%.*]], align 8
// IR: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr align 8 [[PTR]], i64 4
  return bp11.b2;
}

// CHECK-LABEL: Record: RecordDecl {{.*}} struct bt3 definition
// CHECK: Layout: <CGRecordLayout
// CHECK-NEXT: LLVMType:%struct.bt3 = type { i24 }
// CHECK: BitFields:[
// CHECK-NEXT: <CGBitFieldInfo Offset:22 Size:10 IsSigned:1 StorageSize:32 StorageOffset:0
// CHECK-NEXT: <CGBitFieldInfo Offset:12 Size:10 IsSigned:1 StorageSize:32 StorageOffset:0
// CHECK-NEXT: ]>
