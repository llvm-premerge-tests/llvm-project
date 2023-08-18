; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

; FIXED WIDTH

define i32 @ctz_v8i1(<8 x i1> %a) {
; CHECK-LABEL: .LCPI0_0:
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .byte 7
; CHECK-NEXT:   .byte 6
; CHECK-NEXT:   .byte 5
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 1
; CHECK-LABEL: ctz_v8i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp x8, .LCPI0_0
; CHECK-NEXT:    shl v0.8b, v0.8b, #7
; CHECK-NEXT:    mov w9, #8 // =0x8
; CHECK-NEXT:    cmlt v0.8b, v0.8b, #0
; CHECK-NEXT:    ldr d1, [x8, :lo12:.LCPI0_0]
; CHECK-NEXT:    and v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    smaxv b0, v0.8b
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.v8i1(<8 x i1> %a)
  ret i32 %res
}

define i32 @ctz_v16i1(<16 x i1> %a) {
; CHECK-LABEL: .LCPI1_0:
; CHECK-NEXT:   .byte 16
; CHECK-NEXT:   .byte 15
; CHECK-NEXT:   .byte 14
; CHECK-NEXT:   .byte 13
; CHECK-NEXT:   .byte 12
; CHECK-NEXT:   .byte 11
; CHECK-NEXT:   .byte 10
; CHECK-NEXT:   .byte 9
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .byte 7
; CHECK-NEXT:   .byte 6
; CHECK-NEXT:   .byte 5
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 1
; CHECK-LABEL: ctz_v16i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp x8, .LCPI1_0
; CHECK-NEXT:    shl v0.16b, v0.16b, #7
; CHECK-NEXT:    mov w9, #16 // =0x10
; CHECK-NEXT:    cmlt v0.16b, v0.16b, #0
; CHECK-NEXT:    ldr q1, [x8, :lo12:.LCPI1_0]
; CHECK-NEXT:    and v0.16b, v0.16b, v1.16b
; CHECK-NEXT:    smaxv b0, v0.16b
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.v16i1(<16 x i1> %a)
  ret i32 %res
}

define i32 @ctz_v4i32(<4 x i32> %a) {
; CHECK-LABEL: .LCPI2_0:
; CHECK-NEXT:   .hword 4
; CHECK-NEXT:   .hword 3
; CHECK-NEXT:   .hword 2
; CHECK-NEXT:   .hword 1
; CHECK-LABEL: ctz_v4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    adrp x8, .LCPI2_0
; CHECK-NEXT:    mov w9, #4 // =0x4
; CHECK-NEXT:    cmtst v0.4s, v0.4s, v0.4s
; CHECK-NEXT:    ldr d1, [x8, :lo12:.LCPI2_0]
; CHECK-NEXT:    xtn v0.4h, v0.4s
; CHECK-NEXT:    and v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    smaxv h0, v0.4h
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.v4i32(<4 x i32> %a)
  ret i32 %res
}

; SCALABLE, WITH VSCALE RANGE

define i32 @ctz_nxv8i1(<vscale x 8 x i1> %a) #0 {
; CHECK-LABEL: ctz_nxv8i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    index z0.h, #0, #-1
; CHECK-NEXT:    mov z1.h, p0/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    inch z0.h
; CHECK-NEXT:    ptrue p0.h
; CHECK-NEXT:    and z0.d, z0.d, z1.d
; CHECK-NEXT:    cnth x9
; CHECK-NEXT:    sxtb z0.h, p0/m, z0.h
; CHECK-NEXT:    smaxv h0, p0, z0.h
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv8i1(<vscale x 8 x i1> %a)
  ret i32 %res
}

;define i32 @ctz_nxv32i1(<vscale x 32 x i1> %a) #0 {
;  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv32i1(<vscale x 32 x i1> %a)
;  ret i32 %res
;}

define i32 @ctz_nxv4i32(<vscale x 4 x i32> %a) {
; CHECK-LABEL: ctz_nxv4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    cntw x9
; CHECK-NEXT:    cmpne p1.s, p0/z, z0.s, #0
; CHECK-NEXT:    index z0.s, #0, #-1
; CHECK-NEXT:    mov z1.s, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    incw z0.s
; CHECK-NEXT:    and z0.d, z0.d, z1.d
; CHECK-NEXT:    smaxv s0, p0, z0.s
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv4i32(<vscale x 4 x i32> %a)
  ret i32 %res
}

; SCALABLE, NO VSCALE RANGE

define i32 @ctz_nxv8i1_no_range(<vscale x 8 x i1> %a) {
; CHECK-LABEL: ctz_nxv8i1_no_range:
; CHECK:       // %bb.0:
; CHECK-NEXT:    cntw x8
; CHECK-NEXT:    index z0.s, #0, #-1
; CHECK-NEXT:    neg x8, x8
; CHECK-NEXT:    punpklo p1.h, p0.b
; CHECK-NEXT:    incw z0.s, all, mul #2
; CHECK-NEXT:    punpkhi p0.h, p0.b
; CHECK-NEXT:    mov z1.s, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    cnth x9
; CHECK-NEXT:    mov z2.s, w8
; CHECK-NEXT:    and z1.d, z0.d, z1.d
; CHECK-NEXT:    add z0.s, z0.s, z2.s
; CHECK-NEXT:    mov z2.s, p0/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    and z0.d, z0.d, z2.d
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    smax z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT:    smaxv s0, p0, z0.s
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv8i1(<vscale x 8 x i1> %a)
  ret i32 %res
}

; MATCH WITH BRKB + CNTP

define i32 @ctz_nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: ctz_nxv16i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.b
; CHECK-NEXT:    brkb p0.b, p0/z, p1.b
; CHECK-NEXT:    cntp x0, p0, p0.b
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv16i1(<vscale x 16 x i1> %a)
  ret i32 %res
}

define i32 @ctz_and_nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: ctz_and_nxv16i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    brkb p0.b, p0/z, p1.b
; CHECK-NEXT:    cntp x0, p0, p0.b
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %and = and <vscale x 16 x i1> %pg, %a
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv16i1(<vscale x 16 x i1> %and)
  ret i32 %res
}

declare i32 @llvm.experimental.cttz.elts.i32.v8i1(<8 x i1>)
declare i32 @llvm.experimental.cttz.elts.i32.v16i1(<16 x i1>)
declare i32 @llvm.experimental.cttz.elts.i32.v4i32(<4 x i32>)

declare i32 @llvm.experimental.cttz.elts.i32.nxv32i1(<vscale x 32 x i1>)
declare i32 @llvm.experimental.cttz.elts.i32.nxv8i1(<vscale x 8 x i1>)
declare i32 @llvm.experimental.cttz.elts.i32.nxv16i1(<vscale x 16 x i1>)
declare i32 @llvm.experimental.cttz.elts.i32.nxv4i32(<vscale x 4 x i32>)

attributes #0 = { vscale_range(1,16) }
