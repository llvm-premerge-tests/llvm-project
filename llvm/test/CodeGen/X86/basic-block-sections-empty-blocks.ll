;; This test verifies that with -gc-empty-blocks SHT_LLVM_BB_ADDR_MAP will not include entries for empty blocks.
; RUN: llc < %s -mtriple=x86_64 -O0 -basic-block-sections=labels -gc-empty-blocks | FileCheck --check-prefix=CHECK %s

define void @foo(i1 zeroext %0) nounwind {
entry:
  %1 = select i1 %0, ptr blockaddress(@foo, %bb2), ptr blockaddress(@foo, %bb2) ; <ptr> [#uses=1]
  indirectbr ptr %1, [label %bb2, label %bb2]

bb1:                                                ; preds = %entry
  unreachable

bb2:                                                ; preds = %entry
  %2 = call i32 @bar()
  ret void
}

declare i32 @bar()

; CHECK: .section	.llvm_bb_addr_map,"o",@llvm_bb_addr_map,.text
; CHECK-NEXT: .byte	2                               # version
; CHECK-NEXT: .byte	0                               # feature
; CHECK-NEXT: .quad	.Lfunc_begin0                   # function address
; CHECK-NEXT: .byte	2                               # number of basic blocks
; CHECK-NEXT: .byte	0                               # BB id
; CHECK-NEXT: .uleb128 .Lfunc_begin0-.Lfunc_begin0
; CHECK-NEXT: .uleb128 .LBB_END0_0-.Lfunc_begin0
; CHECK-NEXT: .byte	16
; CHECK-NEXT: .byte	1                               # BB id
; CHECK-NEXT: .uleb128 .LBB0_1-.LBB_END0_0
; CHECK-NEXT: .uleb128 .LBB_END0_1-.LBB0_1
; CHECK-NEXT: .byte	1
