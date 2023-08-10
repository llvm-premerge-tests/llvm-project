; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: Attribute 'noreadafterunwind' applied to incompatible type!
; CHECK-NEXT: ptr @not_pointer
define void @not_pointer(i32 noreadafterunwind %arg) {
  ret void
}
