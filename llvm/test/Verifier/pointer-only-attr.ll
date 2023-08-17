; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: Attribute 'writable' applied to incompatible type!
; CHECK-NEXT: ptr @not_pointer
define void @not_pointer_writable(i32 writable %arg) {
  ret void
}
