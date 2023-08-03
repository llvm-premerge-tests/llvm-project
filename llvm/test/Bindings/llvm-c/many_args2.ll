; RUN: llvm-c-test --test-bpf-unsupported-emit < %s | FileCheck %s
;
; Source code:
;   long foo(long a, long b, long c, long d, long e, long f) { return f; }
; Compilation flag:
;   clang -target bpf -S -emit-llvm test.c -o -
;
; This test contains a function that cannot be correctly compiled to BPF assembly because it exceeds
; the number of permissible arguments.
;
; Before this test was introduced, the BPF backend would generate incorrect-but-viable code
; that would pass the verifier and violate user expectations at runtime.
;
; This test ensures that when forced to miscompile, the BPF backend emits code that would be
; rejected by the verifier.

; CHECK:      foo:
; CHECK-NOT:          r0 =
; CHECK:              exit

; Function Attrs: noinline nounwind optnone
define dso_local i64 @foo(i64 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4, i64 noundef %5) #0 {
  %7 = alloca i64, align 8
  %8 = alloca i64, align 8
  %9 = alloca i64, align 8
  %10 = alloca i64, align 8
  %11 = alloca i64, align 8
  %12 = alloca i64, align 8
  store i64 %0, ptr %7, align 8
  store i64 %1, ptr %8, align 8
  store i64 %2, ptr %9, align 8
  store i64 %3, ptr %10, align 8
  store i64 %4, ptr %11, align 8
  store i64 %5, ptr %12, align 8
  %13 = load i64, ptr %12, align 8
  ret i64 %13
}
