; RUN: llvm-dis < %s.bc | FileCheck %s

; Test auto-upgrade of bitcode from load v1 to load v2. This test uses a
; non-standard value for !nonnull, which is not an empty metadata value. This
; distinguishes that all metadata references are correctly updated.

%array = type [10 x ptr]

; CHECK: define i32 @loadPtrPtr(ptr noundef %a)
; CHECK-NEXT: entry:
; CHECK-NEXT: %a.0.ptr = getelementptr inbounds [10 x ptr], ptr %a, i64 0, i64 0, !noundef [[NOUNDEF:![0-9]+]]
; CHECK-NEXT: %v.ptr = load ptr, ptr %a.0.ptr, align 8, !nonnull [[NONNULL:![0-9]+]], !noundef [[NOUNDEF]], !freeze_bits [[NOUNDEF]]
; CHECK-NEXT: %v = load i32, ptr %v.ptr, align 8, !freeze_bits [[NOUNDEF]]
; CHECK-NEXT: ret i32 %v
define i32 @loadPtrPtr(ptr noundef %a) {
  %a.0.ptr = getelementptr inbounds %array, ptr %a, i64 0, i64 0, !noundef !1
  %v.ptr = load ptr, ptr %a.0.ptr, align 8, !nonnull !0, !noundef !1
  %v = load i32, ptr %v.ptr, align 8
  ret i32 %v
}

; CHECK: [[NOUNDEF]] = !{}
; CHECK: [[NONNULL]] = !{i32 1}
!0 = !{i32 1}
!1 = !{}
