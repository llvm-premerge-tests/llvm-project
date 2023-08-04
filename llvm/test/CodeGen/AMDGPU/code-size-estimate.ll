; RUN: llc -march=amdgcn -mcpu=gfx900 -show-mc-encoding < %s | FileCheck -check-prefixes=CHECK,GFX9 %s
; RUN: llc -march=amdgcn -mcpu=gfx1030 -show-mc-encoding < %s | FileCheck -check-prefixes=CHECK,GFX10 %s

declare float @llvm.fabs.f32(float)

define float @v_mul_f32_vop2(float %x, float %y) {
; GFX9-LABEL: v_mul_f32_vop2:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_mul_f32_e32 v0, v0, v1 ; encoding: [0x00,0x03,0x00,0x0a]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop2:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e32 v0, v0, v1 ; encoding: [0x00,0x03,0x00,0x10]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
  %mul = fmul float %x, %y
  ret float %mul
}
; CHECK: codeLenInByte = 12

define float @v_mul_f32_vop2_inline_imm(float %x) {
; GFX9-LABEL: v_mul_f32_vop2_inline_imm:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_mul_f32_e32 v0, 4.0, v0 ; encoding: [0xf6,0x00,0x00,0x0a]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop2_inline_imm:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e32 v0, 4.0, v0 ; encoding: [0xf6,0x00,0x00,0x10]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
  %mul = fmul float %x, 4.0
  ret float %mul
}
; CHECK: codeLenInByte = 12

define float @v_mul_f32_vop2_literal(float %x) {
; GFX9-LABEL: v_mul_f32_vop2_literal:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_mul_f32_e32 v0, 0x42f60000, v0 ; encoding: [0xff,0x00,0x00,0x0a,0x00,0x00,0xf6,0x42]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop2_literal:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e32 v0, 0x42f60000, v0 ; encoding: [0xff,0x00,0x00,0x10,0x00,0x00,0xf6,0x42]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
  %mul = fmul float %x, 123.0
  ret float %mul
}
; CHECK: codeLenInByte = 16

define float @v_mul_f32_vop3_src_mods(float %x, float %y) {
; GFX9-LABEL: v_mul_f32_vop3_src_mods:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_mul_f32_e64 v0, |v0|, v1 ; encoding: [0x00,0x01,0x05,0xd1,0x00,0x03,0x02,0x00]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop3_src_mods:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e64 v0, |v0|, v1 ; encoding: [0x00,0x01,0x08,0xd5,0x00,0x03,0x02,0x00]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %mul = fmul float %fabs.x, %y
  ret float %mul
}
; CHECK: codeLenInByte = 16

define float @v_mul_f32_vop3_src_mods_inline_imm(float %x, float %y) {
; GFX9-LABEL: v_mul_f32_vop3_src_mods_inline_imm:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_mul_f32_e64 v0, |v0|, 4.0 ; encoding: [0x00,0x01,0x05,0xd1,0x00,0xed,0x01,0x00]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop3_src_mods_inline_imm:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e64 v0, |v0|, 4.0 ; encoding: [0x00,0x01,0x08,0xd5,0x00,0xed,0x01,0x00]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %mul = fmul float %fabs.x, 4.0
  ret float %mul
}

; CHECK: codeLenInByte = 16

define float @v_mul_f32_vop3_src_mods_literal(float %x, float %y) {
; GFX9-LABEL: v_mul_f32_vop3_src_mods_literal:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    s_mov_b32 s4, 0x42f60000 ; encoding: [0xff,0x00,0x84,0xbe,0x00,0x00,0xf6,0x42]
; GFX9-NEXT:    v_mul_f32_e64 v0, |v0|, s4 ; encoding: [0x00,0x01,0x05,0xd1,0x00,0x09,0x00,0x00]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop3_src_mods_literal:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_mul_f32_e64 v0, 0x42f60000, |v0| ; encoding: [0x00,0x02,0x08,0xd5,0xff,0x00,0x02,0x00,0x00,0x00,0xf6,0x42]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
  %fabs.x = call float @llvm.fabs.f32(float %x)
  %mul = fmul float %fabs.x, 123.0
  ret float %mul
}

; GFX9: codeLenInByte = 24
; GFX10: codeLenInByte = 20

define float @v_mul_f32_vop2_frame_index(float %x) {
; GFX9-LABEL: v_mul_f32_vop2_frame_index:
; GFX9:       ; %bb.0:
; GFX9-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX9-NEXT:    v_lshrrev_b32_e64 v1, 6, s32 ; encoding: [0x01,0x00,0x10,0xd1,0x86,0x40,0x00,0x00]
; GFX9-NEXT:    v_mul_f32_e32 v0, v1, v0 ; encoding: [0x01,0x01,0x00,0x0a]
; GFX9-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x1d,0x80,0xbe]
;
; GFX10-LABEL: v_mul_f32_vop2_frame_index:
; GFX10:       ; %bb.0:
; GFX10-NEXT:    s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0) ; encoding: [0x00,0x00,0x8c,0xbf]
; GFX10-NEXT:    v_lshrrev_b32_e64 v1, 5, s32 ; encoding: [0x01,0x00,0x16,0xd5,0x85,0x40,0x00,0x00]
; GFX10-NEXT:    v_mul_f32_e32 v0, v1, v0 ; encoding: [0x01,0x01,0x00,0x10]
; GFX10-NEXT:    s_setpc_b64 s[30:31] ; encoding: [0x1e,0x20,0x80,0xbe]
  %alloca = alloca i32, addrspace(5)
  %ptrtoint = ptrtoint ptr addrspace(5) %alloca to i32
  %cast = bitcast i32 %ptrtoint to float
  %mul = fmul float %x, %cast
  ret float %mul
}

; CHECK: codeLenInByte = 20
