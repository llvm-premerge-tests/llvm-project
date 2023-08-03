; RUN: llvm-c-test --test-bpf-unsupported-emit < %s | FileCheck %s
;
; This test contains a function that cannot be correctly compiled to BPF assembly because it returns
; a value wider than a register.
;
; Before this test was introduced, the BPF backend would generate incorrect-but-viable code
; that would pass the verifier and violate user expectations at runtime.
;
; This test ensures that when forced to miscompile, the BPF backend emits code that would be
; rejected by the verifier.

; CHECK:      foo:
; CHECK-NOT:          r0 =
; CHECK:              exit

; Function Attrs: nounwind uwtable
define i64 @foo() #0 {
entry:
  %0 = tail call { i64, i32 } @bar() #3
  %1 = extractvalue { i64, i32 } %0, 0
  ret i64 %1
}

declare { i64, i32 } @bar() #1
