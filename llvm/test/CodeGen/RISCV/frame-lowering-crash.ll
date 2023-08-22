; RUN: llc -mattr=+zcmp -verify-machineinstrs  \
; RUN: -mtriple=riscv32 -target-abi ilp32 < %s \
; RUN: | FileCheck %s -check-prefixes=RV32I
; RUN: llc -mattr=+zcmp -verify-machineinstrs  \
; RUN: -mtriple=riscv64 -target-abi lp64 < %s  \
; RUN: | FileCheck %s -check-prefixes=RV64I

; This input source exposed a crash in RISC-V RISCVFrameLowering when executing
; the Prologue/Epilogue Insertion & Frame Finalization pass. The root cause was:
; Not doing a bounds check before using a returned iterator.
; NOTE: -mattr=+zcmp is essential to the reproducibility of the issue.

declare hidden void @f1() local_unnamed_addr
define hidden void @f0() local_unnamed_addr {
; RV32I-LABEL: f0:                                     # @f0
; RV32I:        .cfi_startproc
; RV32I-NEXT: # %bb.0:                                # %entry
; RV32I-NEXT:         beqz    zero, .LBB0_2
; RV32I-NEXT: .LBB0_1:                                # %cleanup
; RV32I-NEXT:         ret
; RV32I-NEXT: .LBB0_2:                                # %if.end
; RV32I-NEXT:         beqz    zero, .LBB0_5
; RV32I-NEXT: # %bb.3:                                # %for.cond
; RV32I-NEXT:         bnez    zero, .LBB0_1
; RV32I-NEXT: .LBB0_4:                                # %for.body
; RV32I-NEXT:                                         # =>This Inner Loop Header: Depth=1
; RV32I-NEXT:         sw      zero, 0(zero)
; RV32I-NEXT:         beqz    zero, .LBB0_4
; RV32I-NEXT:         j       .LBB0_1
; RV32I-NEXT: .LBB0_5:                                # %if.then2
; RV32I-NEXT:         tail    f1
; RV32I-NEXT: .Lfunc_end0:
; RV32I-NEXT:         .size   f0, .Lfunc_end0-f0
; RV32I-NEXT:         .cfi_endproc

; RV64I-LABEL: f0:                                     # @f0
; RV64I:        .cfi_startproc
; RV64I-NEXT: # %bb.0:                                # %entry
; RV64I-NEXT:         beqz    zero, .LBB0_2
; RV64I-NEXT: .LBB0_1:                                # %cleanup
; RV64I-NEXT:         ret
; RV64I-NEXT: .LBB0_2:                                # %if.end
; RV64I-NEXT:         beqz    zero, .LBB0_5
; RV64I-NEXT: # %bb.3:                                # %for.cond
; RV64I-NEXT:         bnez    zero, .LBB0_1
; RV64I-NEXT: .LBB0_4:                                # %for.body
; RV64I-NEXT:                                         # =>This Inner Loop Header: Depth=1
; RV64I-NEXT:         sw      zero, 0(zero)
; RV64I-NEXT:         beqz    zero, .LBB0_4
; RV64I-NEXT:         j       .LBB0_1
; RV64I-NEXT: .LBB0_5:                                # %if.then2
; RV64I-NEXT:         tail    f1
; RV64I-NEXT: .Lfunc_end0:
; RV64I-NEXT:         .size   f0, .Lfunc_end0-f0
; RV64I-NEXT:         .cfi_endproc

entry:
  br i1 poison, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  br i1 poison, label %if.end3, label %if.then2

if.then2:                                         ; preds = %if.end
  tail call void @f1() #4
  br label %cleanup

if.end3:                                          ; preds = %if.end
  br label %for.cond

for.cond:                                         ; preds = %for.body, %if.end3
  %cmp6 = icmp slt i32 poison, poison
  br i1 %cmp6, label %for.body, label %for.cond8

for.body:                                         ; preds = %for.cond
  store i32 0, ptr null, align 4
  br label %for.cond

for.cond8:                                        ; preds = %for.cond
  %cmp10 = icmp slt i32 poison, poison
  br label %cleanup

cleanup:                                          ; preds = %for.cond8, %if.then2, %entry
  ret void
}
