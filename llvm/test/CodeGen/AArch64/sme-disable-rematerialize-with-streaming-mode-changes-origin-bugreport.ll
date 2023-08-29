; RUN: llc < %s | FileCheck %s

; This test is a bit brittle, but it's very hard to construct a test where this
; happens, and perhaps better than no test at all.
;
; We want to avoid the following sequence from happening where the register
; allocator rematerialises the cntd within the call sequence.
;   smstart sm
;   bl      foo
;   cntd    x8
;   smstop  sm

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

define void @avoid_remat_in_call_sequence(ptr %n, ptr %idx, i32 %0, ptr %list, i64 %idxprom35, ptr %1, ptr %2, ptr %vla25, <vscale x 2 x i1> %3, ptr %vla26) #0 {
; CHECK:        cntd
; CHECK:        smstart sm
; CHECK-NOT:    cntd
; CHECK:        bl      foo
; CHECK-NOT:    cntd
; CHECK:        smstop  sm
entry:
  br label %for.body

for.body:                                         ; preds = %for.body31, %entry
  %vla253 = alloca double, i64 0, align 8
  call void @foo()
  %.tr = call i32 @llvm.vscale.i32()
  %4 = shl i32 %.tr, 1
  br label %for.body31

for.body31:                                       ; preds = %for.body31, %for.body
  %5 = call <vscale x 2 x i1> @llvm.aarch64.sve.whilelt.nxv2i1.i32(i32 0, i32 %0)
  %idx2 = getelementptr [12000000 x i32], ptr %list, i64 0, i64 %idxprom35
  %6 = call <vscale x 2 x i32> @llvm.masked.load.nxv2i32.p0(ptr %idx2, i32 1, <vscale x 2 x i1> %5, <vscale x 2 x i32> zeroinitializer)
  %7 = sext <vscale x 2 x i32> %6 to <vscale x 2 x i64>
  %8 = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.index.nxv2f64(<vscale x 2 x i1> zeroinitializer, ptr %1, <vscale x 2 x i64> zeroinitializer)
  %9 = call <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.index.nxv2f64(<vscale x 2 x i1> zeroinitializer, ptr %2, <vscale x 2 x i64> %7)
  %10 = call <vscale x 2 x double> @llvm.masked.load.nxv2f64.p0(ptr %idx, i32 1, <vscale x 2 x i1> %3, <vscale x 2 x double> zeroinitializer)
  %11 = call <vscale x 2 x double> @llvm.masked.load.nxv2f64.p0(ptr %vla26, i32 1, <vscale x 2 x i1> %3, <vscale x 2 x double> zeroinitializer)
  %12 = call <vscale x 2 x double> @llvm.masked.load.nxv2f64.p0(ptr %n, i32 1, <vscale x 2 x i1> %3, <vscale x 2 x double> zeroinitializer)
  call void @llvm.aarch64.sve.st1.scatter.index.nxv2f64(<vscale x 2 x double> zeroinitializer, <vscale x 2 x i1> zeroinitializer, ptr %vla25, <vscale x 2 x i64> zeroinitializer)
  %cmp29 = icmp slt i32 %4, 0
  br i1 %cmp29, label %for.body31, label %for.body
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare <vscale x 2 x i1> @llvm.aarch64.sve.whilelt.nxv2i1.i32(i32, i32) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <vscale x 2 x double> @llvm.aarch64.sve.ld1.gather.index.nxv2f64(<vscale x 2 x i1>, ptr, <vscale x 2 x i64>) #2

declare void @foo() #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.aarch64.sve.st1.scatter.index.nxv2f64(<vscale x 2 x double>, <vscale x 2 x i1>, ptr, <vscale x 2 x i64>) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.vscale.i32() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <vscale x 2 x i32> @llvm.masked.load.nxv2i32.p0(ptr nocapture, i32 immarg, <vscale x 2 x i1>, <vscale x 2 x i32>) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare <vscale x 2 x double> @llvm.masked.load.nxv2f64.p0(ptr nocapture, i32 immarg, <vscale x 2 x i1>, <vscale x 2 x double>) #2

; uselistorder directives
uselistorder ptr @llvm.aarch64.sve.ld1.gather.index.nxv2f64, { 1, 0 }
uselistorder ptr @llvm.masked.load.nxv2f64.p0, { 2, 1, 0 }

attributes #0 = { "target-features"="+bf16,+crc,+dotprod,+fp-armv8,+fullfp16,+lse,+neon,+outline-atomics,+ras,+rcpc,+rdm,+sme,+sme2,+sve,+sve2,+sve2-bitperm,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8.5a,+v8a,+v9a,-fmv" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #3 = { "aarch64_pstate_sm_enabled" }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
