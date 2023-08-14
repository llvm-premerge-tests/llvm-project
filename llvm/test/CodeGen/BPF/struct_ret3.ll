; RUN: not llc -march=bpf < %s 2> %t1
; RUN: FileCheck %s < %t1
; CHECK: error: <unknown>:0:0: in function baz void (ptr): aggregate returns are not supported

%struct.B = type { [100 x i64] }

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define dso_local void @baz(ptr noalias nocapture sret(%struct.B) align 8 %agg.result) local_unnamed_addr #0 {
entry:
  ret void
}
