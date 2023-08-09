; RUN: llc -mtriple=x86_64-- -stop-after fastpretileconfig -global-isel -O0 -o - %s | FileCheck %s

; GlobalIsel doesn't use all virtual registers and there may be holes
; during consecutive iteration of them. Note that %3 is absent.
; https://github.com/llvm/llvm-project/issues/64452
define i64 @f(i64 %0, i64 %1) {
entry:
; CHECK:     liveins
; CHECK-NOT: %3:
; CHECK:     RET 0, implicit $rax
  %2 = lshr i64 %0, %1
  %3 = add i64 %2, 123456789
  ret i64 %3
}
