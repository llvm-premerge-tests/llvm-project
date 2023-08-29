; RUN: llc -mtriple=aarch64-linux-gnu  -mattr=+sme < %s | FileCheck %s

declare void @normal_callee();
declare void @streaming_callee() "aarch64_pstate_sm_enabled";
declare void @streaming_compatible_callee() "aarch64_pstate_sm_compatible";


; Caller is non-streaming mode

define void @non_streaming_caller_to_streaming_callee() nounwind {
; CHECK-LABEL: non_streaming_caller_to_streaming_callee:
; CHECK:       // %bb.0: // %entry
; CHECK:         smstart sm
; CHECK:         bl streaming_callee
; CHECK:         smstop sm
entry:
  tail call void @streaming_callee() "aarch64_pstate_sm_enabled"
  ret void
}

define void @non_streaming_caller_to_streaming_compatible_callee() nounwind {
; CHECK-LABEL: non_streaming_caller_to_streaming_compatible_callee:
; CHECK:       // %bb.0: // %entry
; CHECK-NOT:     {{smstart|smstop}}
; CHECK:         b streaming_compatible_callee
; CHECK-NOT:     {{smstart|smstop}}
entry:
  tail call void @streaming_compatible_callee() "aarch64_pstate_sm_compatible"
  ret void
}

; Caller is streaming mode

define void @streaming_caller_to_streaming_callee() nounwind "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: streaming_caller_to_streaming_callee:
; CHECK:       // %bb.0: // %entry
; CHECK-NOT:     {{smstart|smstop}}
; CHECK-NEXT:    b streaming_callee
; CHECK-NOT:     {{smstart|smstop}}
entry:
  tail call void @streaming_callee() "aarch64_pstate_sm_enabled"
  ret void
}

define void @streaming_caller_to_non_streaming_callee() nounwind "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: streaming_caller_to_non_streaming_callee:
; CHECK:       // %bb.0: // %entry
; CHECK:         smstop sm
; CHECK:         bl normal_callee
; CHECK:         smstart sm
entry:
 tail call void @normal_callee()
 ret void
}

define void @streaming_caller_to_streaming_compatible_callee() nounwind "aarch64_pstate_sm_enabled" {
; CHECK-LABEL: streaming_caller_to_streaming_compatible_callee:
; CHECK:       // %bb.0: // %entry
; CHECK-NOT:     {{smstart|smstop}}
; CHECK:         b streaming_compatible_callee
; CHECK-NOT:     {{smstart|smstop}}
entry:
  tail call void @streaming_compatible_callee() "aarch64_pstate_sm_compatible"
  ret void
}

; Caller is streaming compatible mode

define void @streaming_compatible_caller_to_streaming_callee() nounwind "aarch64_pstate_sm_compatible"{
; CHECK-LABEL: streaming_compatible_caller_to_streaming_callee:
; CHECK:       // %bb.1: // %entry
; CHECK:         smstart sm
; CHECK:         bl streaming_callee
; CHECK:         smstop sm
entry:
  tail call void @streaming_callee() "aarch64_pstate_sm_enabled"
  ret void
}

define void @streaming_compatible_caller_to_non_streaming_callee() nounwind "aarch64_pstate_sm_compatible"{
; CHECK-LABEL: streaming_compatible_caller_to_non_streaming_callee:
; CHECK:       // %bb.1: // %entry
; CHECK:         smstop sm
; CHECK:         bl normal_callee
; CHECK:         smstart sm
entry:
 tail call void @normal_callee()
 ret void
}

define void @streaming_compatible_caller_to_streaming_compatible_callee() nounwind "aarch64_pstate_sm_compatible" {
; CHECK-LABEL: streaming_compatible_caller_to_streaming_compatible_callee:
; CHECK:       // %bb.0: // %entry
; CHECK-NOT:     {{smstart|smstop}}
; CHECK:         b streaming_compatible_callee
; CHECK-NOT:     {{smstart|smstop}}
entry:
  tail call void @streaming_compatible_callee() "aarch64_pstate_sm_compatible"
  ret void
}

declare void @za_new_callee() "aarch64_pstate_za_new";
declare void @za_shared_callee() "aarch64_pstate_za_shared";
declare void @za_preserved_callee() "aarch64_pstate_za_preserved";

; Caller with ZA state new

define void @za_new_caller_to_za_new_callee() nounwind "aarch64_pstate_za_new" {
; CHECK-LABEL: za_new_caller_to_za_new_callee:
; CHECK:       // %bb.0: // %prelude
; CHECK:         bl __arm_tpidr2_save
; CHECK:         smstart za
; CHECK:         bl za_new_callee
; CHECK:         smstart za
; CHECK:         bl __arm_tpidr2_restore
; CHECK:         smstop za
entry:
  tail call void @za_new_callee() "aarch64_pstate_za_new";
  ret void;
}


define void @za_new_caller_to_za_shared_callee() nounwind "aarch64_pstate_za_new" {
; CHECK-LABEL: za_new_caller_to_za_shared_callee:
; CHECK:       // %bb.0: // %prelude
; CHECK:         bl __arm_tpidr2_save
; CHECK:         smstart za
; CHECK:         bl za_shared_callee
; CHECK:         smstop za
entry:
  tail call void @za_shared_callee() "aarch64_pstate_za_shared";
 ret void;
}

define void @za_new_caller_to_za_preserved_callee() nounwind "aarch64_pstate_za_new" {
; CHECK-LABEL: za_new_caller_to_za_preserved_callee:
; CHECK:       // %bb.0: // %prelude
; CHECK:         bl __arm_tpidr2_save
; CHECK:         smstart za
; CHECK:         bl za_preserved_callee
; CHECK:         smstop za
entry:
  tail call void @za_preserved_callee() "aarch64_pstate_za_preserved";
  ret void;
}


; Caller with ZA state shared

define void @za_shared_caller_to_za_new_callee() nounwind "aarch64_pstate_za_shared" {
; CHECK-LABEL: za_shared_caller_to_za_new_callee:
; CHECK:       // %bb.0: // %entry
; CHECK:         bl za_new_callee
; CHECK-NEXT:    smstart za
; CHECK:         bl __arm_tpidr2_restore
entry:
  tail call void @za_new_callee() "aarch64_pstate_za_new";
  ret void;
}

define void @za_shared_caller_to_za_shared_callee() nounwind "aarch64_pstate_za_shared" {
; CHECK-LABEL: za_shared_caller_to_za_shared_callee:
; CHECK:       // %bb.0: // %entry
; CHECK-NOT:     {{smstart|smstop}}
; CHECK:         b za_shared_callee
; CHECK-NOT:     {{smstart|smstop}}
entry:
  tail call void @za_shared_callee() "aarch64_pstate_za_shared";
  ret void;
}

define void @za_shared_caller_to_za_preserved_callee() nounwind "aarch64_pstate_za_shared" {
; CHECK-LABEL: za_shared_caller_to_za_preserved_callee:
; CHECK:       // %bb.0: // %entry
; CHECK-NOT:     {{smstart|smstop}}
; CHECK:         b za_preserved_callee
; CHECK-NOT:     {{smstart|smstop}}
entry:
  tail call void @za_preserved_callee() "aarch64_pstate_za_preserved";
  ret void;
}

; Caller with ZA state preserved

define void @za_preserved_caller_to_za_preserved_callee() nounwind "aarch64_pstate_za_preserved" {
; CHECK-LABEL: za_preserved_caller_to_za_preserved_callee:
; CHECK:       // %bb.0: // %entry
; CHECK-NOT:     {{smstart|smstop}}
; CHECK-NEXT:    b za_preserved_callee
; CHECK-NOT:     {{smstart|smstop}}
entry:
  tail call void @za_preserved_callee() "aarch64_pstate_za_preserved";
  ret void;
}
