; RUN: opt -mtriple=aarch64-unknown-linux-gnu -mattr=+sve -passes=loop-vectorize -pass-remarks=loop-vectorize -disable-output < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; prefer vscale x 16 when vscale x 16 and vscale x 8 have same cost.
; CHECK: remark: {{.*}}:0:0: vectorized loop (vectorization width: vscale x 16, interleaved count: 2)

define void @pixel_avg(ptr noalias nocapture %dst, i32 %i_dst_stride, ptr noalias nocapture %src1, i32 %i_src1_stride, ptr noalias nocapture %src2, i32 %i_src2_stride, i32 %i_width, i32 %i_height) {
entry:
  %cmp29 = icmp sgt i32 %i_height, 0
  br i1 %cmp29, label %for.preheader.lr.ph, label %cleanup

for.preheader.lr.ph:                        ; preds = %entry
  %cmp227 = icmp sgt i32 %i_width, 0
  %idx.ext = sext i32 %i_dst_stride to i64
  %idx.ext12 = sext i32 %i_src1_stride to i64
  %idx.ext14 = sext i32 %i_src2_stride to i64
  %wide.trip.count = zext i32 %i_width to i64
  br i1 %cmp227, label %for.preheader, label %cleanup

for.preheader:                           ; preds = %for.preheader.lr.ph, %for.latch
  %y.033.us = phi i32 [ %inc17.us, %for.latch ], [ 0, %for.preheader.lr.ph ]
  %dst.addr.032.us = phi ptr [ %add.ptr.us, %for.latch ], [ %dst, %for.preheader.lr.ph ]
  %src1.addr.031.us = phi ptr [ %add.ptr13.us, %for.latch ], [ %src1, %for.preheader.lr.ph ]
  %src2.addr.030.us = phi ptr [ %add.ptr15.us, %for.latch ], [ %src2, %for.preheader.lr.ph ]
  br label %for.body

for.body:                                     ; preds = %for.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx.us = getelementptr inbounds i8, ptr %src1.addr.031.us, i64 %indvars.iv
  %0 = load i8, ptr %arrayidx.us, align 1
  %conv.us = zext i8 %0 to i16
  %arrayidx6.us = getelementptr inbounds i8, ptr %src2.addr.030.us, i64 %indvars.iv
  %1 = load i8, ptr %arrayidx6.us, align 1
  %conv7.us = zext i8 %1 to i16
  %add.us = add nuw nsw i16 %conv.us, 1
  %add8.us = add nuw nsw i16 %add.us, %conv7.us
  %shr.us = lshr i16 %add8.us, 1
  %conv9.us = trunc i16 %shr.us to i8
  %arrayidx11.us = getelementptr inbounds i8, ptr %dst.addr.032.us, i64 %indvars.iv
  store i8 %conv9.us, ptr %arrayidx11.us, align 1
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.latch, label %for.body

for.latch:         ; preds = %for.body
  %add.ptr.us = getelementptr inbounds i8, ptr %dst.addr.032.us, i64 %idx.ext
  %add.ptr13.us = getelementptr inbounds i8, ptr %src1.addr.031.us, i64 %idx.ext12
  %add.ptr15.us = getelementptr inbounds i8, ptr %src2.addr.030.us, i64 %idx.ext14
  %inc17.us = add nuw nsw i32 %y.033.us, 1
  %exitcond36.not = icmp eq i32 %inc17.us, %i_height
  br i1 %exitcond36.not, label %cleanup, label %for.preheader

cleanup:                                 ; preds = %for.latch, %for.preheader.lr.ph, %entry
  ret void
}
