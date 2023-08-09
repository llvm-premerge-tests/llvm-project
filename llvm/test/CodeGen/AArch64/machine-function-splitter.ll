; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -enable-split-machine-functions -aarch64-redzone | FileCheck %s -check-prefix=MFS-REDZONE

define i32 @nosplit_redzone(i1 zeroext %0, i32 %a, i32 %b) nounwind !prof !14 !section_prefix !15 {
;; Check that cold blocks in functions with red zones aren't split.
; MFS-REDZONE-LABEL: nosplit_redzone
; MFS-REDZONE-NOT:   nosplit_redzone.cold:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %x = alloca i32, align 4

  br i1 %0, label %2, label %3, !prof !16

2:                                                ; preds = %1
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  br label %4

3:                                                ; preds = %1
  store i32 %a, ptr %b.addr, align 4
  store i32 %b, ptr %a.addr, align 4
  br label %4

4:                                                ; preds = %3, %2
  %tmp = load i32, ptr %a.addr, align 4
  %tmp1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %tmp, %tmp1
  store i32 %add, ptr %x, align 4
  %tmp2 = load i32, ptr %x, align 4
  ret i32 %tmp2
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
