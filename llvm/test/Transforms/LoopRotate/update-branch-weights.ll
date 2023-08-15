; RUN: opt < %s -passes='print<block-freq>' -disable-output 2>&1 | FileCheck %s --check-prefixes=BFI_BEFORE
; RUN: opt < %s -passes='loop(loop-rotate),print<block-freq>' -disable-output 2>&1 | FileCheck %s --check-prefixes=BFI_AFTER
; RUN: opt < %s -passes='loop(loop-rotate)' -S | FileCheck %s --check-prefixes=IR

@g = global i32 0

; We should get the same "count =" results for "outer_loop_body" and
; "inner_loop_body" before and after the transformation.

; BFI_BEFORE-LABEL: block-frequency-info: func0
; BFI_BEFORE: - entry: float = 1.0, int = 8, count = 1
; BFI_BEFORE: - outer_loop_header: float = 1001.0, int = 8008, count = 1001
; BFI_BEFORE: - outer_loop_body: float = 1000.0, int = 8000, count = 1000
; BFI_BEFORE: - inner_loop_header: float = 4000.0, int = 32000, count = 4000
; BFI_BEFORE: - inner_loop_body: float = 3000.0, int = 24000, count = 3000
; BFI_BEFORE: - inner_loop_exit: float = 1000.0, int = 8000, count = 1000
; BFI_BEFORE: - outer_loop_exit: float = 1.0, int = 8, count = 1

; BFI_AFTER-LABEL: block-frequency-info: func0
; BFI_AFTER: - entry: float = 1.0, int = 8, count = 1
; BFI_AFTER: - outer_loop_body: float = 1000.0, int = 7999, count = 1000
; BFI_AFTER: - inner_loop_body: float = 3000.0, int = 23999, count = 3000
; BFI_AFTER: - inner_loop_exit: float = 1000.0, int = 7999, count = 1000
; BFI_AFTER: - outer_loop_exit: float = 1.0, int = 8, count = 1

; IR: inner_loop_body:
; IR:   br i1 %cmp1, label %inner_loop_body, label %inner_loop_exit, !prof [[PROF0:![0-9]+]]
; IR: inner_loop_exit:
; IR:   br i1 %cmp0, label %outer_loop_body, label %outer_loop_exit, !prof [[PROF1:![0-9]+]]
;
; A function with known loop-bounds where after loop-rotation we end with an
; unconditional branch in the pre-header.
define void @func0() !prof !0 {
entry:
  br label %outer_loop_header

outer_loop_header:
  %i0 = phi i32 [0, %entry], [%i0_inc, %inner_loop_exit]
  %cmp0 = icmp slt i32 %i0, 1000
  br i1 %cmp0, label %outer_loop_body, label %outer_loop_exit, !prof !1

outer_loop_body:
  store volatile i32 %i0, ptr @g, align 4
  br label %inner_loop_header

inner_loop_header:
  %i1 = phi i32 [0, %outer_loop_body], [%i1_inc, %inner_loop_body]
  %cmp1 = icmp slt i32 %i1, 3
  br i1 %cmp1, label %inner_loop_body, label %inner_loop_exit, !prof !2

inner_loop_body:
  store volatile i32 %i1, ptr @g, align 4
  %i1_inc = add i32 %i1, 1
  br label %inner_loop_header

inner_loop_exit:
  %i0_inc = add i32 %i0, 1
  br label %outer_loop_header

outer_loop_exit:
  ret void
}

; BFI_BEFORE-LABEL: block-frequency-info: func1
; BFI_BEFORE: - entry: float = 1.0, int = 8, count = 1024
; BFI_BEFORE: - loop_header: float = 21.0, int = 168, count = 21504
; BFI_BEFORE: - loop_body: float = 20.0, int = 160, count = 20480
; BFI_BEFORE: - loop_exit: float = 1.0, int = 8, count = 1024

; BFI_AFTER-LABEL: block-frequency-info: func1
; BFI_AFTER: - entry: float = 1.0, int = 8, count = 1024
; BFI_AFTER: - loop_body.lr.ph: float = 1.0, int = 8, count = 1024
; BFI_AFTER: - loop_body: float = 20.0, int = 160, count = 20480
; BFI_AFTER: - loop_header.loop_exit_crit_edge: float = 1.0, int = 8, count = 1024
; BFI_AFTER: - loop_exit: float = 1.0, int = 8, count = 1024

; IR: entry:
; IR:   br i1 %cmp1, label %loop_body.lr.ph, label %loop_exit, !prof [[PROF2:![0-9]+]]

; IR: loop_body:
; IR:   br i1 %cmp, label %loop_body, label %loop_header.loop_exit_crit_edge, !prof [[PROF3:![0-9]+]]

; A function with unknown loop-bounds so loop-rotation so we end up with a
; condition jump in pre-header and loop body. Variant where branch_weight
; shows body is executed more often than header.
define void @func1(i32 %n) !prof !3 {
entry:
  br label %loop_header

loop_header:
  %i = phi i32 [0, %entry], [%i_inc, %loop_body]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop_body, label %loop_exit, !prof !4

loop_body:
  store volatile i32 %i, ptr @g, align 4
  %i_inc = add i32 %i, 1
  br label %loop_header

loop_exit:
  ret void
}

; BFI_BEFORE-LABEL: block-frequency-info: func2
; BFI_BEFORE: - entry: float = 1.0, int = 256, count = 1024
; BFI_BEFORE: - loop_header: float = 1.0312, int = 264, count = 1056
; BFI_BEFORE: - loop_body: float = 0.03125, int = 8, count = 32
; BFI_BEFORE: - loop_exit: float = 1.0, int = 256, count = 1024

; BFI_AFTER-LABEL: block-frequency-info: func2
; - entry: float = 1.0, int = 256, count = 1024
; - loop_body.lr.ph: float = 0.03125, int = 8, count = 32
; - loop_body: float = 0.03125, int = 8, count = 32
; - loop_header.loop_exit_crit_edge: float = 0.03125, int = 8, count = 32
; - loop_exit: float = 1.0, int = 256, count = 1024

; IR: entry:
; IR:   br i1 %cmp1, label %loop_exit, label %loop_body.lr.ph, !prof [[PROF4:![0-9]+]]

; IR: loop_body:
; IR:   br i1 %cmp, label %loop_header.loop_exit_crit_edge, label %loop_body, !prof [[PROF5:![0-9]+]]

; A function with unknown loop-bounds so loop-rotation so we end up with a
; condition jump in pre-header and loop body. Variant where branch_weight
; shows loop-exit count higher than backedge counts.
define void @func2(i32 %n) !prof !3 {
entry:
  br label %loop_header

loop_header:
  %i = phi i32 [0, %entry], [%i_inc, %loop_body]
  %cmp = icmp slt i32 %i, %n
  br i1 %cmp, label %loop_exit, label %loop_body, !prof !5

loop_body:
  store volatile i32 %i, ptr @g, align 4
  %i_inc = add i32 %i, 1
  br label %loop_header

loop_exit:
  ret void
}

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"branch_weights", i32 1000, i32 1}
!2 = !{!"branch_weights", i32 3000, i32 1000}
!3 = !{!"function_entry_count", i64 1024}
!4 = !{!"branch_weights", i32 20480, i32 1024}
!5 = !{!"branch_weights", i32 10240, i32 320}

; IR: [[PROF0]] = !{!"branch_weights", i32 2000, i32 1000}
; IR: [[PROF1]] = !{!"branch_weights", i32 999, i32 1}
; IR: [[PROF2]] = !{!"branch_weights", i32 1024, i32 0}
; IR: [[PROF3]] = !{!"branch_weights", i32 19456, i32 1024}
; IR: [[PROF4]] = !{!"branch_weights", i32 9920, i32 320}
; IR: [[PROF5]] = !{!"branch_weights", i32 320, i32 0}
