; RUN: opt -O2 -mtriple=bpf-pc-linux -mcpu=v4 -o - %s \
; RUN:   | llc -mcpu=v4 - | FileCheck %s
;
; Source:
;    struct bpf_sockopt {
;      int family;
;      int level;
;      int optlen;
;    };
;    
;    extern void consume(int);
;    
;    void load_sink_example(struct bpf_sockopt *ctx)
;    {
;      if (ctx->level == 42)
;        consume(ctx->family);
;      else
;        consume(ctx->optlen);
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -triple bpf -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.bpf_sockopt = type { i32, i32, i32 }

; Function Attrs: nounwind
define dso_local void @load_sink_example(ptr noundef %ctx) #0 {
; CHECK: {{.*}} = *(u32 *)(r1 + 4)
; CHECK: w1 = *(u32 *)(r1 + 0)
; CHECK: w1 = *(u32 *)(r1 + 8)
entry:
  %level = getelementptr inbounds %struct.bpf_sockopt, ptr %ctx, i32 0, i32 1
  %0 = load i32, ptr %level, align 4, !tbaa !2
  %cmp = icmp eq i32 %0, 42
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %family = getelementptr inbounds %struct.bpf_sockopt, ptr %ctx, i32 0, i32 0
  %1 = load i32, ptr %family, align 4, !tbaa !7
  call void @consume(i32 noundef %1)
  br label %if.end

if.else:                                          ; preds = %entry
  %optlen = getelementptr inbounds %struct.bpf_sockopt, ptr %ctx, i32 0, i32 2
  %2 = load i32, ptr %optlen, align 4, !tbaa !8
  call void @consume(i32 noundef %2)
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

declare void @consume(i32 noundef) #1

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 18.0.0 (/home/eddy/work/llvm-project/clang 53c495f835926142c10c80d7d0505f59b1e46e49)"}
!2 = !{!3, !4, i64 4}
!3 = !{!"bpf_sockopt", !4, i64 0, !4, i64 4, !4, i64 8}
!4 = !{!"int", !5, i64 0}
!5 = !{!"omnipotent char", !6, i64 0}
!6 = !{!"Simple C/C++ TBAA"}
!7 = !{!3, !4, i64 0}
!8 = !{!3, !4, i64 8}
