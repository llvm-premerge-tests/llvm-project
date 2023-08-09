; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -enable-split-machine-functions | FileCheck %s -check-prefix=MFS-DEFAULTS

define i32 @nosplit_jumptable(i32 %in) !prof !14 !section_prefix !15 {
;; Check that a cold block targeted by a jump table is not split.
; MFS-DEFAULTS-LABEL: nosplit_jumptable
; MFS-DEFAULTS:       .section        .text.split.nosplit_jumptable
; MFS-DEFAULTS-NEXT:  nosplit_jumptable.cold:
; MFS-DEFAULTS-SAME:    // %common.ret1
; MFS-DEFAULTS-NOT:   b       bar
; MFS-DEFAULTS-NOT:   b       baz
; MFS-DEFAULTS-NOT:   b       qux
; MFS-DEFAULTS-NOT:   b       bam

  switch i32 %in, label %common.ret [
    i32 0, label %hot1
    i32 1, label %hot2
    i32 2, label %cold1
    i32 3, label %cold2
  ], !prof !17

common.ret:                                       ; preds = %0
  ret i32 0

hot1:                                             ; preds = %0
  %1 = tail call i32 @bar()
  ret i32 %1

hot2:                                             ; preds = %0
  %2 = tail call i32 @baz()
  ret i32 %2

cold1:                                            ; preds = %0
  %3 = tail call i32 @bam()
  ret i32 %3

cold2:                                            ; preds = %0
  %4 = tail call i32 @qux()
  ret i32 %4
}

declare i32 @bar()
declare i32 @baz()
declare i32 @bam()
declare i32 @qux()

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ProfileSummary", !1}
!1 = !{!2, !3, !4, !5, !6, !7, !8, !9}
!2 = !{!"ProfileFormat", !"InstrProf"}
!3 = !{!"TotalCount", i64 10000}
!4 = !{!"MaxCount", i64 10}
!5 = !{!"MaxInternalCount", i64 1}
!6 = !{!"MaxFunctionCount", i64 1000}
!7 = !{!"NumCounts", i64 3}
!8 = !{!"NumFunctions", i64 5}
!9 = !{!"DetailedSummary", !10}
!10 = !{!11, !12, !13}
!11 = !{i32 10000, i64 100, i32 1}
!12 = !{i32 999900, i64 100, i32 1}
!13 = !{i32 999999, i64 1, i32 2}
!14 = !{!"function_entry_count", i64 9000}
!15 = !{!"function_section_prefix", !"hot"}
!16 = !{!"branch_weights", i32 7000, i32 0}
!17 = !{!"branch_weights", i32 0, i32 4000, i32 4000, i32 0, i32 1000}