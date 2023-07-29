;; Check that SimplifyCFG does not attempt speculation until after PGO is
;; annotated in the IR, and then does not perform it when unprofitable.

;; No PGO: SimplifyCFG should speculate.
; RUN: opt < %s -passes='default<O2>' -S | FileCheck %s --check-prefix=YES
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -S | FileCheck %s --check-prefix=YES
; RUN: opt < %s -passes='lto-pre-link<O2>' -S | FileCheck %s --check-prefix=YES

;; Instrumentation PGO: SimplifyCFG should only attempt speculation after PGO
;; annotation, and in the below cases should not speculate given the branch
;; biases.
; RUN: llvm-profdata merge %S/Inputs/simplifycfg-speculate-blocks.proftext -o %t.profdata
; RUN: opt < %s -passes='default<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file=%t.profdata -S | FileCheck %s --check-prefixes=NO,INSTR
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file=%t.profdata -S | FileCheck %s --check-prefixes=NO,INSTR
; RUN: opt < %s -passes='lto-pre-link<O2>' -pgo-kind=pgo-instr-use-pipeline -profile-file=%t.profdata -S | FileCheck %s --check-prefixes=NO,INSTR --implicit-check-not=select

;; Sample PGO: SimplifyCFG should only attempt speculation after PGO
;; annotation, and in the below cases should not speculate given the branch
;; biases.
; RUN: opt < %s -passes='default<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/simplifycfg-speculate-blocks.sampleprof -S | FileCheck %s --check-prefixes=NO,SAMPLE
; RUN: opt < %s -passes='thinlto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/simplifycfg-speculate-blocks.sampleprof -S | FileCheck %s --check-prefixes=NO,SAMPLE
; RUN: opt < %s -passes='lto-pre-link<O2>' -pgo-kind=pgo-sample-use-pipeline -profile-file=%S/Inputs/simplifycfg-speculate-blocks.sampleprof -S | FileCheck %s --check-prefixes=NO,SAMPLE

%structx = type { [23 x i8], [0 x i8], i8 }
define ptr @fold_two_entry_phi(ptr noundef nonnull align 8 dereferenceable(24) %v0) #0 !dbg !12 {
; YES-LABEL: define ptr @fold_two_entry_phi
; NO-LABEL: define ptr @fold_two_entry_phi
  %1 = getelementptr inbounds %structx, ptr %v0, i64 0, i32 2
  %2 = load i8, ptr %1, align 1
  %3 = icmp slt i8 %2, 0
  br i1 %3, label %truelabel, label %falselabel, !dbg !15
; YES:    select
; INSTR: br i1 {{.*}} !prof ![[PROF:[0-9]+]]
; SAMPLE: br i1 {{.*}} !prof ![[PROF1:[0-9]+]]
truelabel:
  %4 = load ptr, ptr %v0, align 8, !dbg !16
  br label %retlabel, !dbg !16
falselabel:
  br label %retlabel, !dbg !17
retlabel:
  %5 = phi ptr [ %4, %truelabel ], [ %v0, %falselabel ], !dbg !18
  ret ptr %5, !dbg !18
}

define void @fold_branch_to_common_dest(i8 %v0, i8 %v1) #0 !dbg !19 {
; YES-LABEL: define void @fold_branch_to_common_dest
; NO-LABEL: define void @fold_branch_to_common_dest
pred:
  %c0 = icmp eq i8 %v0, 0
  br i1 %c0, label %dispatch, label %final_right, !dbg !20
; YES:    select
; INSTR: br i1 %c0, label %dispatch, label %final_right, !dbg {{.*}}, !prof ![[PROF]]
; SAMPLE: br i1 %c0, label %dispatch, label %final_right, !dbg {{.*}}, !prof ![[PROF2:[0-9]+]]
dispatch:
  %c1 = icmp eq i8 %v1, 0, !dbg !21
  br i1 %c1, label %final_left, label %final_right, !dbg !21
final_left:
  call void @sideeffect0(), !dbg !22
  ret void, !dbg !22
final_right:
  call void @sideeffect1(), !dbg !23
  ret void, !dbg !23
}

declare void @sideeffect0()
declare void @sideeffect1()

; INSTR: ![[PROF]] = !{!"branch_weights", i32 0, i32 5000}
; SAMPLE: ![[PROF1]] = !{!"branch_weights", i32 1, i32 5002}
; SAMPLE: ![[PROF2]] = !{!"branch_weights", i32 1, i32 5001}

attributes #0 = { "use-sample-profile" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 17.0.0 (git@github.com:llvm/llvm-project.git 180a8148f95d2d5ed96c670fc322292fa0df876c)", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "simplifycfg-speculate-blocks.cc", directory: ".", checksumkind: CSK_MD5, checksum: "89bb3bed72c5e9634aa5f0653fc7a01b")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!12 = distinct !DISubprogram(name: "fold_two_entry_phi", scope: !1, file: !1, line: 1, type: !13, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!13 = !DISubroutineType(types: !14)
!14 = !{}
!15 = !DILocation(line: 2, column: 11, scope: !12)
!16 = !DILocation(line: 3, column: 11, scope: !12)
!17 = !DILocation(line: 4, column: 11, scope: !12)
!18 = !DILocation(line: 5, column: 11, scope: !12)
!19 = distinct !DISubprogram(name: "fold_branch_to_common_dest", scope: !1, file: !1, line: 10, type: !13, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!20 = !DILocation(line: 11, column: 11, scope: !19)
!21 = !DILocation(line: 12, column: 11, scope: !19)
!22 = !DILocation(line: 13, column: 11, scope: !19)
!23 = !DILocation(line: 14, column: 11, scope: !19)
