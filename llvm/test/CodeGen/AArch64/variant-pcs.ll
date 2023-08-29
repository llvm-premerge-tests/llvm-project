; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve,+sme -o - %s | FileCheck %s --check-prefix=CHECK-ASM --strict-whitespace
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve,+sme -filetype=obj -o - %s \
; RUN:   | llvm-readobj --symbols - | FileCheck %s --check-prefix=CHECK-OBJ

define i32 @base_pcs() {
; CHECK-ASM-LABEL: base_pcs:
; CHECK-ASM-NOT: .variant_pcs
; CHECK-OBJ-LABEL: Name: base_pcs
; CHECK-OBJ: Other: 0
  ret i32 42
}

define aarch64_vector_pcs <4 x i32> @neon_vector_pcs_1(<4 x i32> %arg) {
; CHECK-ASM: .variant_pcs	neon_vector_pcs_1
; CHECK-ASM-NEXT: neon_vector_pcs_1:
; CHECK-OBJ-LABEL: Name: neon_vector_pcs_1
; CHECK-OBJ: Other [ (0x80)
  ret <4 x i32> %arg
}

define <vscale x 4 x i32> @sve_vector_pcs_1() {
; CHECK-ASM: .variant_pcs	sve_vector_pcs_1
; CHECK-ASM-NEXT: sve_vector_pcs_1:
; CHECK-OBJ-LABEL: Name: sve_vector_pcs_1
; CHECK-OBJ: Other [ (0x80)
  ret <vscale x 4 x i32> undef
}

define <vscale x 4 x i1> @sve_vector_pcs_2() {
; CHECK-ASM: .variant_pcs	sve_vector_pcs_2
; CHECK-ASM-NEXT: sve_vector_pcs_2:
; CHECK-OBJ-LABEL: Name: sve_vector_pcs_2
; CHECK-OBJ: Other [ (0x80)
  ret <vscale x 4 x i1> undef
}

define void @sve_vector_pcs_3(<vscale x 4 x i32> %arg) {
; CHECK-ASM: .variant_pcs	sve_vector_pcs_3
; CHECK-ASM-NEXT: sve_vector_pcs_3:
; CHECK-OBJ-LABEL: Name: sve_vector_pcs_3
; CHECK-OBJ: Other [ (0x80)
  ret void
}

define void @sve_vector_pcs_4(<vscale x 4 x i1> %arg) {
; CHECK-ASM: .variant_pcs	sve_vector_pcs_4
; CHECK-ASM-NEXT: sve_vector_pcs_4:
; CHECK-OBJ-LABEL: Name: sve_vector_pcs_4
; CHECK-OBJ: Other [ (0x80)
  ret void
}

define void @variant_pcs_sme_streaming() "aarch64_pstate_sm_enabled" {
; CHECK-ASM: .variant_pcs	variant_pcs_sme_streaming
; CHECK-ASM-NEXT: variant_pcs_sme_streaming:
; CHECK-OBJ-LABEL: Name: variant_pcs_sme_streaming
; CHECK-OBJ: Other [ (0x80)
  ret void
}

define void @variant_pcs_sme_streaming_compatible() "aarch64_pstate_sm_compatible" {
; CHECK-ASM: .variant_pcs	variant_pcs_sme_streaming_compatible
; CHECK-ASM-NEXT: variant_pcs_sme_streaming_compatible:
; CHECK-OBJ-LABEL: Name: variant_pcs_sme_streaming_compatible
; CHECK-OBJ: Other [ (0x80)
  ret void
}

define void @variant_pcs_sme_shared_za() "aarch64_pstate_za_shared" {
; CHECK-ASM: .variant_pcs	variant_pcs_sme_shared_za
; CHECK-ASM-NEXT: variant_pcs_sme_shared_za:
; CHECK-OBJ-LABEL: Name: variant_pcs_sme_shared_za
; CHECK-OBJ: Other [ (0x80)
  ret void
}

define void @variant_pcs_sme_new_za() "aarch64_pstate_za_new" {
; CHECK-ASM-NOT: .variant_pcs	variant_pcs_sme_new_za
  ret void
}

define void @variant_pcs_sme_preserves_za() "aarch64_pstate_za_shared" "aarch64_pstate_za_preserved" {
; CHECK-ASM: .variant_pcs	variant_pcs_sme_preserves_za
; CHECK-ASM-NEXT: variant_pcs_sme_preserves_za:
; CHECK-OBJ-LABEL: Name: variant_pcs_sme_preserves_za
; CHECK-OBJ: Other [ (0x80)
  ret void
}
