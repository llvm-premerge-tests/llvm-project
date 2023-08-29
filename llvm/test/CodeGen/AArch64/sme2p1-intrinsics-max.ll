; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme2p1 -mattr=+b16b16 -verify-machineinstrs < %s | FileCheck %s

; BFMAX (Single, x2)

define { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @multi_vec_max_single_x2_bf16(<vscale x 8 x bfloat> %zdn1, <vscale x 8 x bfloat> %zdn2, <vscale x 8 x bfloat> %zm) {
; CHECK-LABEL: multi_vec_max_single_x2_bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $z1 killed $z1 killed $z0_z1 def $z0_z1
; CHECK-NEXT:    // kill: def $z0 killed $z0 killed $z0_z1 def $z0_z1
; CHECK-NEXT:    bfmax { z0.h, z1.h }, { z0.h, z1.h }, z2.h
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.bfmax.single.x2.nxv8bf16(<vscale x 8 x bfloat> %zdn1, <vscale x 8 x bfloat> %zdn2, <vscale x 8 x bfloat> %zm)
  ret { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } %res
}

; BFMAX (Single, x4)

define { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> }
@multi_vec_max_single_x4_bf16(<vscale x 8 x bfloat> %zdn1, <vscale x 8 x bfloat> %zdn2, <vscale x 8 x bfloat> %zdn3, <vscale x 8 x bfloat> %zdn4, <vscale x 8 x bfloat> %zm) {
; CHECK-LABEL: multi_vec_max_single_x4_bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $z3 killed $z3 killed $z0_z1_z2_z3 def $z0_z1_z2_z3
; CHECK-NEXT:    // kill: def $z2 killed $z2 killed $z0_z1_z2_z3 def $z0_z1_z2_z3
; CHECK-NEXT:    // kill: def $z1 killed $z1 killed $z0_z1_z2_z3 def $z0_z1_z2_z3
; CHECK-NEXT:    // kill: def $z0 killed $z0 killed $z0_z1_z2_z3 def $z0_z1_z2_z3
; CHECK-NEXT:    bfmax { z0.h - z3.h }, { z0.h - z3.h }, z4.h
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> }
              @llvm.aarch64.sve.bfmax.single.x4.nxv8bf16(<vscale x 8 x bfloat> %zdn1, <vscale x 8 x bfloat> %zdn2, <vscale x 8 x bfloat> %zdn3, <vscale x 8 x bfloat> %zdn4, <vscale x 8 x bfloat> %zm)
  ret { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } %res
}

; BFMAX (Multi, x2)

define { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @multi_vec_max_multi_x2_bf16(<vscale x 8 x bfloat> %zdn1, <vscale x 8 x bfloat> %zdn2, <vscale x 8 x bfloat> %zm1, <vscale x 8 x bfloat> %zm2) {
; CHECK-LABEL: multi_vec_max_multi_x2_bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $z1 killed $z1 killed $z0_z1 def $z0_z1
; CHECK-NEXT:    // kill: def $z3 killed $z3 killed $z2_z3 def $z2_z3
; CHECK-NEXT:    // kill: def $z0 killed $z0 killed $z0_z1 def $z0_z1
; CHECK-NEXT:    // kill: def $z2 killed $z2 killed $z2_z3 def $z2_z3
; CHECK-NEXT:    bfmax { z0.h, z1.h }, { z0.h, z1.h }, { z2.h, z3.h }
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.bfmax.x2.nxv8bf16(<vscale x 8 x bfloat> %zdn1, <vscale x 8 x bfloat> %zdn2, <vscale x 8 x bfloat> %zm1, <vscale x 8 x bfloat> %zm2)
  ret { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } %res
}

; BFMAX (Multi, x4)

define { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> }
@multi_vec_max_multi_x4_bf16(<vscale x 8 x bfloat> %zdn1, <vscale x 8 x bfloat> %zdn2, <vscale x 8 x bfloat> %zdn3, <vscale x 8 x bfloat> %zdn4, <vscale x 8 x bfloat> %zm1, <vscale x 8 x bfloat> %zm2, <vscale x 8 x bfloat> %zm3, <vscale x 8 x bfloat> %zm4) {
; CHECK-LABEL: multi_vec_max_multi_x4_bf16:
; CHECK:       // %bb.0:
; CHECK-NEXT:    // kill: def $z3 killed $z3 killed $z0_z1_z2_z3 def $z0_z1_z2_z3
; CHECK-NEXT:    // kill: def $z7 killed $z7 killed $z4_z5_z6_z7 def $z4_z5_z6_z7
; CHECK-NEXT:    // kill: def $z2 killed $z2 killed $z0_z1_z2_z3 def $z0_z1_z2_z3
; CHECK-NEXT:    // kill: def $z6 killed $z6 killed $z4_z5_z6_z7 def $z4_z5_z6_z7
; CHECK-NEXT:    // kill: def $z1 killed $z1 killed $z0_z1_z2_z3 def $z0_z1_z2_z3
; CHECK-NEXT:    // kill: def $z5 killed $z5 killed $z4_z5_z6_z7 def $z4_z5_z6_z7
; CHECK-NEXT:    // kill: def $z0 killed $z0 killed $z0_z1_z2_z3 def $z0_z1_z2_z3
; CHECK-NEXT:    // kill: def $z4 killed $z4 killed $z4_z5_z6_z7 def $z4_z5_z6_z7
; CHECK-NEXT:    bfmax { z0.h - z3.h }, { z0.h - z3.h }, { z4.h - z7.h }
; CHECK-NEXT:    ret
  %res = call { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> }
              @llvm.aarch64.sve.bfmax.x4.nxv8bf16(<vscale x 8 x bfloat> %zdn1, <vscale x 8 x bfloat> %zdn2, <vscale x 8 x bfloat> %zdn3, <vscale x 8 x bfloat> %zdn4,
                                                        <vscale x 8 x bfloat> %zm1,  <vscale x 8 x bfloat> %zm2,  <vscale x 8 x bfloat> %zm3,  <vscale x 8 x bfloat> %zm4)
  ret { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } %res
}

declare { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.bfmax.single.x2.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)

declare { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> }
 @llvm.aarch64.sve.bfmax.single.x4.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)

declare { <vscale x 8 x bfloat>, <vscale x 8 x bfloat> } @llvm.aarch64.sve.bfmax.x2.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)

declare { <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat> }
 @llvm.aarch64.sve.bfmax.x4.nxv8bf16(<vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>, <vscale x 8 x bfloat>)
