;; This test verifies that -gc-empty-block removes empty blocks.
; RUN: llc < %s -mtriple=x86_64 -O0 -gc-empty-blocks | FileCheck %s

define void @foo(i1 zeroext %0) nounwind {
entry:
  %1 = select i1 %0, ptr blockaddress(@foo, %bb2), ptr blockaddress(@foo, %bb2) ; <ptr> [#uses=1]
  indirectbr ptr %1, [label %bb2, label %bb2]

; CHECK:         .text
; CHECK-LABEL: foo:
; CHECK:         movl    $.Ltmp0, %eax
; CHECK-NEXT:    jmpq    *%rax

bb1:                                                ; preds = %entry
  unreachable
; CHECK-NOT:    %bb1

bb2:                                                ; preds = %entry
  %2 = call i32 @bar()
  ret void
; CHECK-NEXT:  .Ltmp0:
; CHECK-NEXT:  .LBB0_1:
; CHECK-NEXT:    callq   bar
; CHECK-NEXT:    popq   %rax 
; CHECK-NEXT:    retq
}

declare i32 @bar()
