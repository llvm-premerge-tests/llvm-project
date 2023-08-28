; RUN: opt -passes=bpf-context-marker -mtriple=bpf-pc-linux -S -o - %s 2>&1 | FileCheck %s
;
; If load offset is not a constant bpf-context-marker should report a
; warning and remove context.marker.bpf call.
;
; Source:
;    #define __ctx __attribute__((btf_decl_tag("ctx")))
;    
;    struct foo {
;      int a[7];
;    } __ctx;
;    
;    extern void consume(int);
;    
;    void bar(struct foo *p, unsigned long i) {
;      consume(p->a[i]);
;    }
;
; Compilation flag:
;   clang -cc1 -O2 -debug-info-kind=line-tables-only -triple bpf \
;         -S -emit-llvm -disable-llvm-passes -o - \
;       | opt -passes=function(sroa) -S -o -

%struct.foo = type { [7 x i32] }

; CHECK:      warning: some-file.c:10:11: in function bar void (ptr, i64):
; CHECK-SAME: Non-constant offset in access to a field of a type marked with
; CHECK-SAME: btf_decl_tag("ctx") might be rejected by BPF verifier

; Function Attrs: nounwind
define dso_local void @bar(ptr noundef %p, i64 noundef %i) #0 !dbg !5 {
entry:
  %0 = call ptr @llvm.context.marker.bpf(ptr %p), !dbg !9
  %a = getelementptr inbounds %struct.foo, ptr %0, i32 0, i32 0, !dbg !9
  %arrayidx = getelementptr inbounds [7 x i32], ptr %a, i64 0, i64 %i, !dbg !10
  %1 = load i32, ptr %arrayidx, align 4, !dbg !10, !tbaa !11
  call void @consume(i32 noundef %1), !dbg !15
  ret void, !dbg !16
}

; CHECK:      define dso_local void @bar(ptr noundef %[[p:.*]], i64 noundef %[[i:.*]])
; CHECK:        %[[a:.*]] = getelementptr inbounds %struct.foo, ptr %[[p]], i32 0, i32 0, !dbg
; CHECK-NEXT:   %[[arrayidx:.*]] = getelementptr inbounds [7 x i32], ptr %[[a]], i64 0, i64 %[[i]], !dbg
; CHECK-NEXT:   %[[v5:.*]] = load i32, ptr %[[arrayidx]], align 4, !dbg {{.*}}, !tbaa
; CHECK-NEXT:   call void @consume(i32 noundef %[[v5]]), !dbg

declare void @consume(i32 noundef) #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.context.marker.bpf(ptr nocapture readnone) #2

attributes #0 = { nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.ident = !{!4}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 17.0.0 (/home/eddy/work/llvm-project/clang 24793f24a0c419509b42ca4d86ec12f45f059b1f)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{!"clang version 17.0.0 (/home/eddy/work/llvm-project/clang 24793f24a0c419509b42ca4d86ec12f45f059b1f)"}
!5 = distinct !DISubprogram(name: "bar", scope: !6, file: !6, line: 9, type: !7, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!6 = !DIFile(filename: "some-file.c", directory: "/some/dir/")
!7 = !DISubroutineType(types: !8)
!8 = !{}
!9 = !DILocation(line: 10, column: 14, scope: !5)
!10 = !DILocation(line: 10, column: 11, scope: !5)
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !13, i64 0}
!13 = !{!"omnipotent char", !14, i64 0}
!14 = !{!"Simple C/C++ TBAA"}
!15 = !DILocation(line: 10, column: 3, scope: !5)
!16 = !DILocation(line: 11, column: 1, scope: !5)
