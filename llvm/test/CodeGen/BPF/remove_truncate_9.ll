; RUN: llc -mcpu=v3 -march=bpf < %s -verify-machineinstrs | FileCheck %s

; This test checks if BPFMIPeepholeTruncElim::eliminateTruncSeq()
; operates correctly when truncation is done for 32-bit sub-registers.
; Snippet was manually constructed from results of llvm-reduce.

define i1 @foo(ptr %p) {
entry:
  %short = load i16, ptr %p, align 2
  br label %next

; Jump to the new basic block is important, because it creates a COPY
; instruction for %short, which might be materialized might as noop or
; as AND_ri (zero extension) at the start of the basic block.
; The decision depends on TLI.isZExtFree() results, see
; RegsForValue::getCopyToRegs(). Check below verifies that COPY is
; materialized as noop.
next:
  %cond = icmp eq i16 %short, 0
  ret i1 %cond
}

; CHECK:         w[[#A:]] = *(u16 *)(r1 + 0)
; CHECK:         w0 = 1
; CHECK:         if w[[#A]] == 0 goto [[LBL:.*]]
; CHECK:         w0 = 0
; CHECK: [[LBL]]:
; CHECK:         exit
