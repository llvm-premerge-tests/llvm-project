; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme < %s | FileCheck %s

; Streaming mode functions can use the full SME instruction set and the subset
; of SVE and NEON instructions that are legal in streaming mode.
;

; CHECK: sclamp z0.b, z0.b, z0.b
define <vscale x 16 x i8> @streaming_compatible_sme(<vscale x 16 x i8> %x) {
  %1 = call <vscale x 16 x i8> asm "sclamp $0.b, $0.b, $0.b", "=w,0"(<vscale x 16 x i8> %x)
  ret <vscale x 16 x i8> %1
}

; CHECK: add z0.b, z0.b, z0.b
define <vscale x 16 x i8> @streaming_compatible_sve(<vscale x 16 x i8> %x) {
  %1 = call <vscale x 16 x i8> asm "add $0.b, $0.b, $0.b", "=w,0"(<vscale x 16 x i8> %x)
  ret <vscale x 16 x i8> %1
}

; CHECK: fmulx s0, s0, s0
define float @streaming_compatible_neon(float %x) {
  %1 = call float asm "fmulx ${0:s}, ${0:s}, ${0:s}", "=w,0"(float %x)
  ret float %1
}
