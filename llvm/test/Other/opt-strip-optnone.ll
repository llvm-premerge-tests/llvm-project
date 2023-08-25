; RUN: opt %s -strip-optnone -S | FileCheck %s

define i64 @f1 (i64 %a) optnone noinline {
  ret i64 %a 
}

; CHECK: define i64 @f1(i64 %a) {
; CHECK:   ret i64 %a
; CHECK: }
