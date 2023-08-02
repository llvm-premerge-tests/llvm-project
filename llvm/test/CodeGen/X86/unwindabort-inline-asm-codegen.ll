; RUN: llc -fast-isel=0 < %s | FileCheck %s
; RUN: llc -fast-isel=1 < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define dso_local void @test() personality ptr @__gxx_personality_v0 {
entry:
; CHECK-LABEL: test:
; CHECK:      .Ltmp0:
; CHECK-NEXT: #APP
; CHECK-NEXT: nop
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: .Ltmp1:

  call unwindabort void asm sideeffect unwind "nop", "~{dirflag},~{fpsr},~{flags}"()
  ret void
}

declare dso_local i32 @__gxx_personality_v0(...)

; Exception table generation around the inline assembly.

; CHECK-LABEL: GCC_except_table0:
; CHECK-NEXT: .Lexception0:
; CHECK-NEXT: 	.byte	255                             # @LPStart Encoding = omit
; CHECK-NEXT: 	.byte	255                             # @TType Encoding = omit
; CHECK-NEXT: 	.byte	1                               # Call site Encoding = uleb128
; CHECK-NEXT: 	.uleb128 .Lcst_end0-.Lcst_begin0
; CHECK-NEXT: .Lcst_begin0:
; CHECK-NEXT: .Lcst_end0:
