; RUN: llc -mtriple=aarch64 -asm-verbose=0              < %s | FileCheck -DAUT="hint #29" --check-prefixes=COMMON,LDR %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -mattr=v8.3a < %s | FileCheck -DAUT="autiasp"  --check-prefixes=COMMON,LDR %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -aarch64-check-authenticated-lr-by-load=0              < %s | FileCheck -DAUT="hint #29" -DXPAC="hint #7" --check-prefixes=COMMON,XPAC %s
; RUN: llc -mtriple=aarch64 -asm-verbose=0 -aarch64-check-authenticated-lr-by-load=0 -mattr=v8.3a < %s | FileCheck -DAUT="autiasp"  -DXPAC="xpaclri" --check-prefixes=COMMON,XPAC %s

define i32 @tailcall_direct() "sign-return-address"="non-leaf" {
; COMMON-LABEL: tailcall_direct:
; COMMON:         str x30, [sp, #-16]!
; COMMON:         ldr x30, [sp], #16
;
; LDR-NEXT:       [[AUT]]
; LDR-NEXT:       .cfi_negate_ra_state
; LDR-NEXT:       ldr x16, [x30]
; LDR-NEXT:       b callee
;
; XPAC-NEXT:      [[AUT]]
; XPAC-NEXT:      .cfi_negate_ra_state
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPAC]]
; XPAC-NEXT:      cmp x16, x30
; XPAC-NEXT:      b.ne .[[FAIL:LBB[_0-9]+]]
; XPAC-NEXT:      b callee
; XPAC-NEXT:    .[[FAIL]]:
; XPAC-NEXT:      brk #0xc471
  tail call void asm sideeffect "", "~{lr}"()
  %call = tail call i32 @callee()
  ret i32 %call
}

define i32 @tailcall_indirect(ptr %fptr) "sign-return-address"="non-leaf" {
; COMMON-LABEL: tailcall_indirect:
; COMMON:         str x30, [sp, #-16]!
; COMMON:         ldr x30, [sp], #16
;
; LDR-NEXT:       [[AUT]]
; LDR-NEXT:       .cfi_negate_ra_state
; LDR-NEXT:       ldr x16, [x30]
; LDR-NEXT:       br x0
;
; XPAC-NEXT:      [[AUT]]
; XPAC-NEXT:      .cfi_negate_ra_state
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPAC]]
; XPAC-NEXT:      cmp x16, x30
; XPAC-NEXT:      b.ne .[[FAIL:LBB[_0-9]+]]
; XPAC-NEXT:      br x0
; XPAC-NEXT:    .[[FAIL]]:
; XPAC-NEXT:      brk #0xc471
  tail call void asm sideeffect "", "~{lr}"()
  %call = tail call i32 %fptr()
  ret i32 %call
}

define i32 @tailcall_direct_noframe() "sign-return-address"="non-leaf" {
; COMMON-LABEL: tailcall_direct_noframe:
; COMMON-NEXT:    .cfi_startproc
; COMMON-NEXT:    b callee
  %call = tail call i32 @callee()
  ret i32 %call
}

define i32 @tailcall_indirect_noframe(ptr %fptr) "sign-return-address"="non-leaf" {
; COMMON-LABEL: tailcall_indirect_noframe:
; COMMON-NEXT:    .cfi_startproc
; COMMON-NEXT:    br x0
  %call = tail call i32 %fptr()
  ret i32 %call
}

define i32 @tailcall_direct_noframe_sign_all() "sign-return-address"="all" {
; COMMON-LABEL: tailcall_direct_noframe_sign_all:
; COMMON-NOT:     str{{.*}}x30
; COMMON-NOT:     ldr{{.*}}x30
;
; LDR:            [[AUT]]
; LDR-NEXT:       .cfi_negate_ra_state
; LDR-NEXT:       ldr x16, [x30]
; LDR-NEXT:       b callee
;
; XPAC:           [[AUT]]
; XPAC-NEXT:      .cfi_negate_ra_state
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPAC]]
; XPAC-NEXT:      cmp x16, x30
; XPAC-NEXT:      b.ne .[[FAIL:LBB[_0-9]+]]
; XPAC-NEXT:      b callee
; XPAC-NEXT:    .[[FAIL]]:
; XPAC-NEXT:      brk #0xc471
  %call = tail call i32 @callee()
  ret i32 %call
}

define i32 @tailcall_indirect_noframe_sign_all(ptr %fptr) "sign-return-address"="all" {
; COMMON-LABEL: tailcall_indirect_noframe_sign_all:
; COMMON-NOT:     str{{.*}}x30
; COMMON-NOT:     ldr{{.*}}x30
;
; LDR:            [[AUT]]
; LDR-NEXT:       .cfi_negate_ra_state
; LDR-NEXT:       ldr x16, [x30]
; LDR-NEXT:       br x0
;
; XPAC:           [[AUT]]
; XPAC-NEXT:      .cfi_negate_ra_state
; XPAC-NEXT:      mov x16, x30
; XPAC-NEXT:      [[XPAC]]
; XPAC-NEXT:      cmp x16, x30
; XPAC-NEXT:      b.ne .[[FAIL:LBB[_0-9]+]]
; XPAC-NEXT:      br x0
; XPAC-NEXT:    .[[FAIL]]:
; XPAC-NEXT:      brk #0xc471
  %call = tail call i32 %fptr()
  ret i32 %call
}

; Do not emit any LR checks when Shadow Call Stack is enabled
define i32 @tailcall_scs(ptr %fptr) "sign-return-address"="all" shadowcallstack "target-features"="+reserve-x18" {
; COMMON-LABEL: tailcall_scs:
; COMMON:         str x30, [sp, #-16]!
; COMMON:         ldr x30, [sp], #16
;
; COMMON-NOT:     ldr {{.*}}, [x30]
; COMMON-NOT:     xpac
; COMMON-NOT:     hint #7
; COMMON-NOT:     brk
;
; Match the end of function:
; COMMON:         .size tailcall_scs,
  tail call void asm sideeffect "", "~{lr}"()
  %call = tail call i32 %fptr()
  ret i32 %call
}

declare i32 @callee()
