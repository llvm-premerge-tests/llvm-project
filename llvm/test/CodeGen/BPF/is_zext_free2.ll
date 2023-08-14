; RUN: llc -march=bpfel -mattr=+alu32 < %s | FileCheck %s

; Check that zero extension is considered free for load instructions.
; The test case is derived from a bigger C program using llvm-reduce
; and manual simplifications.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"

define i1 @foo(ptr %ptr) {
entry:
  %byte = load volatile i8, ptr %ptr, align 1
  br label %next

; Jump to the new basic block is important, because it creates a COPY
; instruction for %byte, which might be constructed differently
; depending on TLI.isZExtFree() results, see RegsForValue::getCopyToRegs().
next:
; The 'icmp eq i8' requires second argument to be zero extended.
  %cmp = icmp eq i8 12, %byte
; CHECK-NOT: {{[rw][0-9]}} &= 255
  ret i1 %cmp
}
