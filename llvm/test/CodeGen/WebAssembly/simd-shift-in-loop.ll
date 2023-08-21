; RUN: llc < %s -asm-verbose=false -verify-machineinstrs -disable-wasm-fallthrough-return-opt -wasm-disable-explicit-locals -wasm-keep-registers -mattr=+simd128 | FileCheck %s

; Test that SIMD shifts can be lowered correctly even when shift
; values are exported from outside blocks.

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: shl_loop:
; CHECK-NEXT: .functype shl_loop (i32, i32, i32) -> ()
; CHECK-NOT: i8x16.splat
; CHECK-NOT: i8x16.extract_lane_u
; CHECK: i8x16.shl
define void @shl_loop(ptr %a, i8 %shift, i32 %count) {
entry:
 %t1 = insertelement <16 x i8> undef, i8 %shift, i32 0
 %vshift = shufflevector <16 x i8> %t1, <16 x i8> undef, <16 x i32> zeroinitializer
 br label %body
body:
 %out = phi ptr [%a, %entry], [%b, %body]
 %i = phi i32 [0, %entry], [%next, %body]
 %v = load <16 x i8>, ptr %out, align 1
 %r = shl <16 x i8> %v, %vshift
 %b = getelementptr inbounds i8, ptr %out, i32 16
 store <16 x i8> %r, ptr %b
 %next = add i32 %i, 1
 %i.cmp = icmp eq i32 %next, %count
 br i1 %i.cmp, label %body, label %exit
exit:
 ret void
}
