; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sme -verify-machineinstrs < %s | FileCheck %s

; Test SMSTART and SMSTOP are marked as scheduling barriers and are not
; rescheduled around function calls.

declare void @streaming_callee() "aarch64_pstate_sm_enabled";

@arr = global [8 x i8] zeroinitializer, align 2

define void @clear_arr() {
; CHECK-LABEL: clear_arr:
; CHECK:    smstart sm
; CHECK-NEXT:    bl streaming_callee
; CHECK-NEXT:    smstop sm
  call void @streaming_callee()
  call void @llvm.memset.p0i8.i64(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @arr, i64 0, i64 0), i8 0, i64 8, i1 false)
  ret void;
}

declare void @llvm.memset.p0i8.i64(i8*, i8, i64, i1)
