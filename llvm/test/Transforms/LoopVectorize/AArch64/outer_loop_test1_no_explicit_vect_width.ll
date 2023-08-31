; RUN: opt -S -passes=loop-vectorize -enable-vplan-native-path -mtriple aarch64-gnu-linux < %s | FileCheck %s

; extern int arr[8][8];
; extern int arr2[8];
;
; void foo(int n)
; {
;   int i1, i2;
;
; #pragma clang loop vectorize(enable)
;   for (i1 = 0; i1 < 8; i1++) {
;     arr2[i1] = i1;
;     for (i2 = 0; i2 < 8; i2++)
;       arr[i2][i1] = i1 + n;
;   }
; }
;

; CHECK-LABEL: define void @foo_i32
; CHECK-SAME: (i32 [[N:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label %[[SCALAR_PH:.*]], label %vector.ph

; CHECK:       vector.ph:
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <4 x i32> poison, i32 [[N]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <4 x i32> [[BROADCAST_SPLATINSERT]], <4 x i32> poison, <4 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body

; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %[[FOR_INC:.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <4 x i64> [ <i64 0, i64 1, i64 2, i64 3>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %[[FOR_INC]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds [8 x i32], ptr @arr2, i64 0, i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = trunc <4 x i64> [[VEC_IND]] to <4 x i32>
; CHECK-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i32 0
; CHECK-NEXT:    store <4 x i32> [[TMP2]], ptr [[TMP3]], align 4
; CHECK-NEXT:    [[TMP4:%.*]] = trunc <4 x i64> [[VEC_IND]] to <4 x i32>
; CHECK-NEXT:    [[TMP5:%.*]] = add nsw <4 x i32> [[TMP4]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    br label %[[FOR_INNER:.*]]

; CHECK:       [[FOR_INNER]]:
; CHECK-NEXT:    [[SCALAR_PHI:%.*]] = phi i64 [ 0, %vector.body ], [ [[TMP8:%.*]], %[[FOR_INNER]] ]
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr inbounds [8 x [8 x i32]], ptr @arr, i64 0, i64 [[SCALAR_PHI]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr inbounds i32, ptr [[TMP6]], i32 0
; CHECK-NEXT:    store <4 x i32> [[TMP5]], ptr [[TMP7]], align 4
; CHECK-NEXT:    [[TMP8]] = add nuw nsw i64 [[SCALAR_PHI]], 1
; CHECK-NEXT:    [[TMP9:%.*]] = icmp eq i64 [[TMP8]], 8
; CHECK-NEXT:    br i1 [[TMP9]], label %[[FOR_INC]], label %[[FOR_INNER]]

; CHECK:       [[FOR_INC]]:
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 4
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <4 x i64> [[VEC_IND]], <i64 4, i64 4, i64 4, i64 4>
; CHECK-NEXT:    [[TMP10:%.*]] = icmp eq i64 [[INDEX_NEXT]], 8
; CHECK-NEXT:    br i1 [[TMP10]], label %[[MIDDLE_BLOCK:.*]], label %vector.body

@arr2 = external global [8 x i32], align 16
@arr = external global [8 x [8 x i32]], align 16

@arrX = external global [8 x i64], align 16
@arrY = external global [8 x [8 x i64]], align 16

; Function Attrs: norecurse nounwind uwtable
define void @foo_i32(i32 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc8, %entry
  %indvars.iv21 = phi i64 [ 0, %entry ], [ %indvars.iv.next22, %for.inc8 ]
  %arrayidx = getelementptr inbounds [8 x i32], ptr @arr2, i64 0, i64 %indvars.iv21
  %0 = trunc i64 %indvars.iv21 to i32
  store i32 %0, ptr %arrayidx, align 4
  %1 = trunc i64 %indvars.iv21 to i32
  %add = add nsw i32 %1, %n
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx7 = getelementptr inbounds [8 x [8 x i32]], ptr @arr, i64 0, i64 %indvars.iv, i64 %indvars.iv21
  store i32 %add, ptr %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.inc8, label %for.body3

for.inc8:                                         ; preds = %for.body3
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
  %exitcond23 = icmp eq i64 %indvars.iv.next22, 8
  br i1 %exitcond23, label %for.end10, label %for.body, !llvm.loop !1

for.end10:                                        ; preds = %for.inc8
  ret void
}

; CHECK-LABEL: define void @foo_i64
; CHECK-SAME: (i64 [[N:%.*]]) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 false, label [[SCALAR_PH:%.*]], label %vector.ph

; CHECK:       vector.ph:
; CHECK-NEXT:    [[BROADCAST_SPLATINSERT:%.*]] = insertelement <2 x i64> poison, i64 [[N]], i64 0
; CHECK-NEXT:    [[BROADCAST_SPLAT:%.*]] = shufflevector <2 x i64> [[BROADCAST_SPLATINSERT]], <2 x i64> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:    br label %vector.body

; CHECK:       vector.body:
; CHECK-NEXT:    [[INDEX:%.*]] = phi i64 [ 0, %vector.ph ], [ [[INDEX_NEXT:%.*]], %[[FOR_INC:.*]] ]
; CHECK-NEXT:    [[VEC_IND:%.*]] = phi <2 x i64> [ <i64 0, i64 1>, %vector.ph ], [ [[VEC_IND_NEXT:%.*]], %[[FOR_INC]] ]
; CHECK-NEXT:    [[TMP0:%.*]] = add i64 [[INDEX]], 0
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr inbounds [8 x i64], ptr @arrX, i64 0, i64 [[TMP0]]
; CHECK-NEXT:    [[TMP2:%.*]] = getelementptr inbounds i64, ptr [[TMP1]], i32 0
; CHECK-NEXT:    store <2 x i64> [[VEC_IND]], ptr [[TMP2]], align 4
; CHECK-NEXT:    [[TMP3:%.*]] = add nsw <2 x i64> [[VEC_IND]], [[BROADCAST_SPLAT]]
; CHECK-NEXT:    br label %[[FOR_INNER:.*]]

; CHECK:       [[FOR_INNER]]:
; CHECK-NEXT:    [[SCALAR_PHI:%.*]] = phi i64 [ 0, %vector.body ], [ [[TMP6:%.*]], %[[FOR_INNER]] ]
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr inbounds [8 x [8 x i64]], ptr @arrY, i64 0, i64 [[SCALAR_PHI]], i64 [[TMP0]]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr inbounds i64, ptr [[TMP4]], i32 0
; CHECK-NEXT:    store <2 x i64> [[TMP3]], ptr [[TMP5]], align 4
; CHECK-NEXT:    [[TMP6]] = add nuw nsw i64 [[SCALAR_PHI]], 1
; CHECK-NEXT:    [[TMP7:%.*]] = icmp eq i64 [[TMP6]], 8
; CHECK-NEXT:    br i1 [[TMP7]], label %[[FOR_INC]], label %[[FOR_INNER]]

; CHECK:       [[FOR_INC]]:
; CHECK-NEXT:    [[INDEX_NEXT]] = add nuw i64 [[INDEX]], 2
; CHECK-NEXT:    [[VEC_IND_NEXT]] = add <2 x i64> [[VEC_IND]], <i64 2, i64 2>
; CHECK-NEXT:    [[TMP8:%.*]] = icmp eq i64 [[INDEX_NEXT]], 8
; CHECK-NEXT:    br i1 [[TMP8]], label %[[MIDDLE_BLOCK:.*]], label %vector.body
; Function Attrs: norecurse nounwind uwtable
define void @foo_i64(i64 %n) {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc8, %entry
  %indvars.iv21 = phi i64 [ 0, %entry ], [ %indvars.iv.next22, %for.inc8 ]
  %arrayidx = getelementptr inbounds [8 x i64], ptr @arrX, i64 0, i64 %indvars.iv21
  store i64 %indvars.iv21, ptr %arrayidx, align 4
  %add = add nsw i64 %indvars.iv21, %n
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.body
  %indvars.iv = phi i64 [ 0, %for.body ], [ %indvars.iv.next, %for.body3 ]
  %arrayidx7 = getelementptr inbounds [8 x [8 x i64]], ptr @arrY, i64 0, i64 %indvars.iv, i64 %indvars.iv21
  store i64 %add, ptr %arrayidx7, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.inc8, label %for.body3

for.inc8:                                         ; preds = %for.body3
  %indvars.iv.next22 = add nuw nsw i64 %indvars.iv21, 1
  %exitcond23 = icmp eq i64 %indvars.iv.next22, 8
  br i1 %exitcond23, label %for.end10, label %for.body, !llvm.loop !1

for.end10:                                        ; preds = %for.inc8
  ret void
}


!1 = distinct !{!1, !2}
!2 = !{!"llvm.loop.vectorize.enable", i1 true}
