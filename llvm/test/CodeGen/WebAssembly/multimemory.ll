; RUN: llc < %s -mcpu=mvp -mattr=+multimemory | FileCheck %s

; Test that multimemory is properly emitted into the target features section

target triple = "wasm32-unknown-unknown"

define void @foo() {
  ret void
}

; CHECK-LABEL: .custom_section.target_features
; CHECK-NEXT: .int8 1
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 11
; CHECK-NEXT: .ascii "multimemory"
