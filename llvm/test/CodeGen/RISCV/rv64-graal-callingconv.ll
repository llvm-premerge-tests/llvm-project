; RUN: llc -mtriple=riscv64-unknown-linux-gnu -o - %s | FileCheck %s

@var = global [30 x i64] zeroinitializer

define cc 106 void @keep_live() {
  %val = load volatile [30 x i64], ptr @var
  store volatile [30 x i64] %val, ptr @var
; CHECK-NOT: lw s7,
; CHECK-NOT: ld s7,
; CHECK-NOT: lw s11,
; CHECK-NOT: ld s11,
  ret void
}
