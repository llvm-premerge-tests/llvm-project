; RUN: llc < %s -stop-after=finalize-isel -O2 - | FileCheck %s
; Fake uses following tail calls should be pulled in front
; of the TCRETURN instruction. Fake uses using something defined by
; the tail call or after it should be suppressed.

; CHECK: body:
; CHECK: bb.0.{{.*}}:
; CHECK: %0:{{.*}}= COPY
; CHECK-NOT: FAKE_USE
; CHECK: FAKE_USE %0
; CHECK-NOT: FAKE_USE
; CHECK: TCRETURN

define void @bar(i32 %v) {
entry:
  %call = tail call i32 @_Z3fooi(i32 %v)
  %mul = mul nsw i32 %call, 3
  notail call void (...) @llvm.fake.use(i32 %mul)
  notail call void (...) @llvm.fake.use(i32 %call)
  notail call void (...) @llvm.fake.use(i32 %v)
  ret void
}

declare i32 @_Z3fooi(i32) local_unnamed_addr
declare void @llvm.fake.use(...)
