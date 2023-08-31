; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

; FIXED WIDTH

define i8 @ctz_v8i1(<8 x i1> %a) {
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
; CHECK-NEXT:    shl v0.8b, v0.8b, #7
; CHECK-NEXT:    adrp x8, .LCPI0_0
; CHECK-NEXT:    mov w9, #8 // =0x8
; CHECK-NEXT:    ldr d1, [x8, :lo12:.LCPI0_0]
; CHECK-NEXT:    cmlt v0.8b, v0.8b, #0
; CHECK-NEXT:    and v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    umaxv b0, v0.8b
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i8 @llvm.experimental.cttz.elts.i8.v8i1(<8 x i1> %a, i32 0)
  ret i8 %res
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
; CHECK-NEXT:    shl v0.16b, v0.16b, #7
; CHECK-NEXT:    adrp x8, .LCPI1_0
; CHECK-NEXT:    mov w9, #16 // =0x10
; CHECK-NEXT:    ldr q1, [x8, :lo12:.LCPI1_0]
; CHECK-NEXT:    cmlt v0.16b, v0.16b, #0
; CHECK-NEXT:    and v0.16b, v0.16b, v1.16b
; CHECK-NEXT:    umaxv b0, v0.16b
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and w0, w8, #0xff
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.v16i1(<16 x i1> %a, i32 0)
  ret i32 %res
}

define i16 @ctz_v4i32(<4 x i32> %a) {
; CHECK-LABEL: .LCPI2_0:
; CHECK-NEXT:   .hword 4
; CHECK-NEXT:   .hword 3
; CHECK-NEXT:   .hword 2
; CHECK-NEXT:   .hword 1
; CHECK-LABEL: ctz_v4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    cmtst v0.4s, v0.4s, v0.4s
; CHECK-NEXT:    adrp x8, .LCPI2_0
; CHECK-NEXT:    mov w9, #4 // =0x4
; CHECK-NEXT:    ldr d1, [x8, :lo12:.LCPI2_0]
; CHECK-NEXT:    xtn v0.4h, v0.4s
; CHECK-NEXT:    and v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    umaxv h0, v0.4h
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and w0, w8, #0xff
; CHECK-NEXT:    ret
  %res = call i16 @llvm.experimental.cttz.elts.i16.v4i32(<4 x i32> %a, i32 0)
  ret i16 %res
}

; ZERO IS POISON

define i8 @ctz_v8i1_poison(<8 x i1> %a) {
; CHECK-LABEL: .LCPI3_0:
; CHECK-NEXT:   .byte 8
; CHECK-NEXT:   .byte 7
; CHECK-NEXT:   .byte 6
; CHECK-NEXT:   .byte 5
; CHECK-NEXT:   .byte 4
; CHECK-NEXT:   .byte 3
; CHECK-NEXT:   .byte 2
; CHECK-NEXT:   .byte 1
; CHECK-LABEL: ctz_v8i1_poison:
; CHECK:       // %bb.0:
; CHECK-NEXT:    shl v0.8b, v0.8b, #7
; CHECK-NEXT:    adrp x8, .LCPI3_0
; CHECK-NEXT:    mov w9, #8 // =0x8
; CHECK-NEXT:    ldr d1, [x8, :lo12:.LCPI3_0]
; CHECK-NEXT:    cmlt v0.8b, v0.8b, #0
; CHECK-NEXT:    and v0.8b, v0.8b, v1.8b
; CHECK-NEXT:    umaxv b0, v0.8b
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and w9, w8, #0xff
; CHECK-NEXT:    cmp w9, #8
; CHECK-NEXT:    csel w0, w8, w8, eq
; CHECK-NEXT:    ret
  %res = call i8 @llvm.experimental.cttz.elts.i8.v8i1(<8 x i1> %a, i32 1)
  ret i8 %res
}

; SCALABLE, WITH VSCALE RANGE

define i64 @ctz_nxv8i1(<vscale x 8 x i1> %a) #0 {
; CHECK-LABEL: ctz_nxv8i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    index z0.h, #0, #-1
; CHECK-NEXT:    mov z1.h, p0/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    ptrue p0.h
; CHECK-NEXT:    cnth x9
; CHECK-NEXT:    inch z0.h
; CHECK-NEXT:    and z0.d, z0.d, z1.d
; CHECK-NEXT:    and z0.h, z0.h, #0xff
; CHECK-NEXT:    umaxv h0, p0, z0.h
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and x0, x8, #0xff
; CHECK-NEXT:    ret
  %res = call i64 @llvm.experimental.cttz.elts.i64.nxv8i1(<vscale x 8 x i1> %a, i32 0)
  ret i64 %res
}

define i32 @ctz_nxv32i1(<vscale x 32 x i1> %a) #0 {
; CHECK-LABEL: ctz_nxv32i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    index z0.h, #0, #-1
; CHECK-NEXT:    cnth x8
; CHECK-NEXT:    punpklo p2.h, p0.b
; CHECK-NEXT:    neg x8, x8
; CHECK-NEXT:    punpklo p3.h, p1.b
; CHECK-NEXT:    rdvl x9, #2
; CHECK-NEXT:    punpkhi p0.h, p0.b
; CHECK-NEXT:    mov z1.h, w8
; CHECK-NEXT:    rdvl x8, #-1
; CHECK-NEXT:    punpkhi p1.h, p1.b
; CHECK-NEXT:    mov z2.h, w8
; CHECK-NEXT:    inch z0.h, all, mul #4
; CHECK-NEXT:    mov z3.h, p2/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    ptrue p2.h
; CHECK-NEXT:    mov z5.h, p3/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    add z1.h, z0.h, z1.h
; CHECK-NEXT:    add z4.h, z0.h, z2.h
; CHECK-NEXT:    mov z6.h, p0/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z7.h, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    and z0.d, z0.d, z3.d
; CHECK-NEXT:    add z2.h, z1.h, z2.h
; CHECK-NEXT:    and z3.d, z4.d, z5.d
; CHECK-NEXT:    and z1.d, z1.d, z6.d
; CHECK-NEXT:    and z2.d, z2.d, z7.d
; CHECK-NEXT:    umax z0.h, p2/m, z0.h, z3.h
; CHECK-NEXT:    umax z1.h, p2/m, z1.h, z2.h
; CHECK-NEXT:    umax z0.h, p2/m, z0.h, z1.h
; CHECK-NEXT:    umaxv h0, p2, z0.h
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and w0, w8, #0xffff
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv32i1(<vscale x 32 x i1> %a, i32 0)
  ret i32 %res
}

define i32 @ctz_nxv4i32(<vscale x 4 x i32> %a) #0 {
; CHECK-LABEL: ctz_nxv4i32:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    index z1.s, #0, #-1
; CHECK-NEXT:    cntw x9
; CHECK-NEXT:    incw z1.s
; CHECK-NEXT:    cmpne p1.s, p0/z, z0.s, #0
; CHECK-NEXT:    mov z0.s, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    and z0.d, z1.d, z0.d
; CHECK-NEXT:    and z0.s, z0.s, #0xff
; CHECK-NEXT:    umaxv s0, p0, z0.s
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w8, w9, w8
; CHECK-NEXT:    and w0, w8, #0xff
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv4i32(<vscale x 4 x i32> %a, i32 0)
  ret i32 %res
}

; SCALABLE, NO VSCALE RANGE

define i32 @ctz_nxv8i1_no_range(<vscale x 8 x i1> %a) {
; CHECK-LABEL: ctz_nxv8i1_no_range:
; CHECK:       // %bb.0:
; CHECK-NEXT:    index z0.s, #0, #-1
; CHECK-NEXT:    punpklo p1.h, p0.b
; CHECK-NEXT:    cntw x8
; CHECK-NEXT:    punpkhi p0.h, p0.b
; CHECK-NEXT:    neg x8, x8
; CHECK-NEXT:    cnth x9
; CHECK-NEXT:    mov z1.s, w8
; CHECK-NEXT:    incw z0.s, all, mul #2
; CHECK-NEXT:    mov z2.s, p1/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    mov z3.s, p0/z, #-1 // =0xffffffffffffffff
; CHECK-NEXT:    ptrue p0.s
; CHECK-NEXT:    add z1.s, z0.s, z1.s
; CHECK-NEXT:    and z0.d, z0.d, z2.d
; CHECK-NEXT:    and z1.d, z1.d, z3.d
; CHECK-NEXT:    umax z0.s, p0/m, z0.s, z1.s
; CHECK-NEXT:    umaxv s0, p0, z0.s
; CHECK-NEXT:    fmov w8, s0
; CHECK-NEXT:    sub w0, w9, w8
; CHECK-NEXT:    ret
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv8i1(<vscale x 8 x i1> %a, i32 0)
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
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv16i1(<vscale x 16 x i1> %a, i32 0)
  ret i32 %res
}

define i32 @ctz_and_nxv16i1(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b) {
; CHECK-LABEL: ctz_and_nxv16i1:
; CHECK:       // %bb.0:
; CHECK-NEXT:    ptrue p1.b
; CHECK-NEXT:    cmpne p0.b, p0/z, z0.b, z1.b
; CHECK-NEXT:    brkb p0.b, p1/z, p0.b
; CHECK-NEXT:    cntp x0, p0, p0.b
; CHECK-NEXT:    // kill: def $w0 killed $w0 killed $x0
; CHECK-NEXT:    ret
  %cmp = icmp ne <vscale x 16 x i8> %a, %b
  %select = select <vscale x 16 x i1> %pg, <vscale x 16 x i1> %cmp, <vscale x 16 x i1> zeroinitializer
  %and = and <vscale x 16 x i1> %pg, %select
  %res = call i32 @llvm.experimental.cttz.elts.i32.nxv16i1(<vscale x 16 x i1> %and, i32 0)
  ret i32 %res
}

declare i8 @llvm.experimental.cttz.elts.i8.v8i1(<8 x i1>, i32)
declare i32 @llvm.experimental.cttz.elts.i32.v16i1(<16 x i1>, i32)
declare i16 @llvm.experimental.cttz.elts.i16.v4i32(<4 x i32>, i32)

declare i32 @llvm.experimental.cttz.elts.i32.nxv8i1(<vscale x 8 x i1>, i32)
declare i64 @llvm.experimental.cttz.elts.i64.nxv8i1(<vscale x 8 x i1>, i32)
declare i32 @llvm.experimental.cttz.elts.i32.nxv16i1(<vscale x 16 x i1>, i32)
declare i32 @llvm.experimental.cttz.elts.i32.nxv32i1(<vscale x 32 x i1>, i32)
declare i32 @llvm.experimental.cttz.elts.i32.nxv4i32(<vscale x 4 x i32>, i32)

attributes #0 = { vscale_range(1,16) }
