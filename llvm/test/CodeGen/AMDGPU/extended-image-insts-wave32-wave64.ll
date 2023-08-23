; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,WAVE32 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -mattr="+wavefrontsize32" -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,WAVE32 %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx1010 -mattr="+wavefrontsize64" -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,WAVE64 %s

; GCN-LABEL: has_extended_image_insts
; WAVE32: s_and_b32 vcc_lo, exec_lo, {{.*}}
; WAVE64: s_and_b64 vcc, exec, {{.*}}
; WAVE32: amdhsa_wavefront_size32 1
; WAVE64: amdhsa_wavefront_size32 0

define amdgpu_kernel void @has_extended_image_insts(float %arg10) #0 {
.entry:
  %tmp100 = fcmp ogt float %arg10, 0.25
  br i1 %tmp100, label %if, label %endif
if:
  %tmp101 = fadd float %arg10, 0.125
  br label %endif
endif:
  %tmp102 = phi float [ %arg10, %.entry ], [ %tmp101, %if ]
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tmp102, float %tmp102, float %tmp102, float %tmp102, i1 true, i1 true)
  ret void
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1)

attributes #0 = { nounwind "target-features"="+extended-image-insts" }
