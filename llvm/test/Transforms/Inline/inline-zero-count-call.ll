; RUN: opt < %s -passes='require<profile-summary>,cgscc(inline)' -S | FileCheck %s

; This tests a call to a function with zero entry count while no profile data
; is available for the basic block of the call site.

define void  @noprofile_f(i32 %n) "function-inline-cost"="15" {
; CHECK-LABEL: define void @noprofile_f
; CHECK:       cond_false:
; CHECK-NEXT:    call void @zero_count_f()
; CHECK-NEXT:    ret void
;
  %cond = icmp sle i32 %n, 10
  br i1 %cond, label %cond_true, label %cond_false
cond_true:
  call void @extern_f()
  ret void
cond_false:
  call void @zero_count_f()
  ret void
}

define void @pprofile_f(i32 %n) !prof !16 {
; CHECK-LABEL: define void @pprofile_f
; CHECK:       cond_false:
; CHECK-NEXT:    call void @zero_count_f()
; CHECK-NEXT:    ret void
;
  %cond = icmp sle i32 %n, 10
  br i1 %cond, label %cond_true, label %cond_false
cond_true:
  call void @extern_f()
  ret void
cond_false:
  call void @zero_count_f()
  ret void
}

define void @zero_count_f() #0  "function-inline-cost"="75" !prof !15 {
  ret void
}

declare void @extern_f()
attributes #0 = { cold }

!15 = !{!"function_entry_count", i64 0}
!16 = !{!"function_entry_count", i64 1000000}

!llvm.module.flags = !{!1}
!1 = !{i32 1, !"ProfileSummary", !2}
!2 = !{!3, !4, !5, !6, !7, !8, !9, !10}
!3 = !{!"ProfileFormat", !"InstrProf"}
!4 = !{!"TotalCount", i64 1000000}
!5 = !{!"MaxCount", i64 1000000}
!6 = !{!"MaxInternalCount", i64 1}
!7 = !{!"MaxFunctionCount", i64 1000000}
!8 = !{!"NumCounts", i64 100}
!9 = !{!"NumFunctions", i64 100}
!10 = !{!"DetailedSummary", !11}
!11 = !{!12, !13, !14}
!12 = !{i32 10000, i64 100, i32 1}
!13 = !{i32 990000, i64 100, i32 1}
!14 = !{i32 999999, i64 1, i32 2}
