; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define i1 @umin(i1 %0, i1 %1) {
; CHECK-LABEL: @umin
; CHECK-NEXT: and i1
  %3 = call i1 @llvm.umin.i1(i1 %0, i1 %1)
  ret i1 %3
}

define i1 @smin(i1 %0, i1 %1) {
; CHECK-LABEL: @smin
; CHECK-NEXT: or i1
  %3 = call i1 @llvm.smin.i1(i1 %0, i1 %1)
  ret i1 %3
}

define i1 @umax(i1 %0, i1 %1) {
; CHECK-LABEL: @umax
; CHECK-NEXT: or i1
  %3 = call i1 @llvm.umax.i1(i1 %0, i1 %1)
  ret i1 %3
}

define i1 @smax(i1 %0, i1 %1) {
; CHECK-LABEL: @smax
; CHECK-NEXT: and i1
  %3 = call i1 @llvm.smax.i1(i1 %0, i1 %1)
  ret i1 %3
}

declare i1 @llvm.umin.i1(i1, i1)
declare i1 @llvm.smin.i1(i1, i1)
declare i1 @llvm.umax.i1(i1, i1)
declare i1 @llvm.smax.i1(i1, i1)
