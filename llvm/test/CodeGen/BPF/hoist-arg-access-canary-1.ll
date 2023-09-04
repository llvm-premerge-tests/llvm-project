; RUN: opt -O2 -mtriple=bpf-pc-linux -mcpu=v4 -o - %s \
; RUN:   | llc -mcpu=v4 - | FileCheck %s
;
; Source:
;    struct __sk_buff {
;      int _;
;      int priority;
;      int mark;
;      int tc_index;
;    };
;    
;    int store_sink_example(struct __sk_buff *ctx) {
;      switch (ctx->priority) {
;      case 10:
;        ctx->mark = 3;
;        break;
;      case 20:
;        ctx->priority = 4;
;        break;
;      }
;      return 0;
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.__sk_buff = type { i32, i32, i32, i32 }

@__ctx = global %struct.__sk_buff zeroinitializer, align 4

; Function Attrs: nounwind
define dso_local i32 @store_sink_example(ptr noundef %ctx) #0 {
; CHECK: {{.*}} = *(u32 *)(r1 + 4)
; CHECK: *(u32 *)(r1 + 8) = 3
; CHECK: *(u32 *)(r1 + 4) = 4
entry:
  %priority = getelementptr inbounds %struct.__sk_buff, ptr %ctx, i32 0, i32 1
  %0 = load i32, ptr %priority, align 4, !tbaa !2
  switch i32 %0, label %sw.epilog [
    i32 10, label %sw.bb
    i32 20, label %sw.bb1
  ]

sw.bb:                                            ; preds = %entry
  %mark = getelementptr inbounds %struct.__sk_buff, ptr %ctx, i32 0, i32 2
  store i32 3, ptr %mark, align 4, !tbaa !7
  br label %sw.epilog

sw.bb1:                                           ; preds = %entry
  %priority2 = getelementptr inbounds %struct.__sk_buff, ptr %ctx, i32 0, i32 1
  store i32 4, ptr %priority2, align 4, !tbaa !2
  br label %sw.epilog

sw.epilog:                                        ; preds = %sw.bb1, %sw.bb, %entry
  ret i32 0
}

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 18.0.0 (/home/eddy/work/llvm-project/clang 53c495f835926142c10c80d7d0505f59b1e46e49)"}
!2 = !{!3, !4, i64 4}
!3 = !{!"__sk_buff", !4, i64 0, !4, i64 4, !4, i64 8, !4, i64 12}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!3, !4, i64 8}
