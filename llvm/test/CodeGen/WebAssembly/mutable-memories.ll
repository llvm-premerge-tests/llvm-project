; RUN: llc < %s -mcpu=mvp -mattr=+multi-memories | FileCheck %s

; Test that multiple memories is properly emitted into the target features section

target triple = "wasm32-unknown-unknown"

define void @foo() {
  ret void
}

; CHECK-LABEL: .custom_section.target_features
; CHECK-NEXT: .int8 1
; CHECK-NEXT: .int8 43
; CHECK-NEXT: .int8 14
; CHECK-NEXT: .ascii "multi-memories"
