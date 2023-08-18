; RUN: opt -aarch64-lir -mtriple aarch64-unknown-linux-gnu -mattr=+sve -S < %s | FileCheck %s
; RUN: opt -aarch64-lir -simplifycfg -mtriple aarch64-unknown-linux-gnu -mattr=+sve -S < %s | FileCheck %s --check-prefix=LOOP-DEL

define i32 @compare_bytes_simple(ptr %a, ptr %b, i32 %len, i32 %n) {
; CHECK-LABEL: define i32 @compare_bytes_simple
; CHECK-SAME: (ptr [[A:%.*]], ptr [[B:%.*]], i32 [[LEN:%.*]], i32 [[N:%.*]]) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[LEN]], 1
; CHECK-NEXT:    br label [[MISMATCH_MIN_IT_CHECK:%.*]]
; CHECK:       mismatch_min_it_check:
; CHECK-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = zext i32 [[N]] to i64
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ule i32 [[TMP0]], [[N]]
; CHECK-NEXT:    br i1 [[TMP3]], label [[MISMATCH_MEM_CHECK:%.*]], label [[MISMATCH_LOOP_PRE:%.*]], !prof [[PROF0:![0-9]+]]
; CHECK:       mismatch_mem_check:
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP1]]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP1]]
; CHECK-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
; CHECK-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[TMP4]] to i64
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP2]]
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP2]]
; CHECK-NEXT:    [[TMP10:%.*]] = ptrtoint ptr [[TMP8]] to i64
; CHECK-NEXT:    [[TMP11:%.*]] = ptrtoint ptr [[TMP9]] to i64
; CHECK-NEXT:    [[TMP12:%.*]] = lshr i64 [[TMP7]], 12
; CHECK-NEXT:    [[TMP13:%.*]] = lshr i64 [[TMP10]], 12
; CHECK-NEXT:    [[TMP14:%.*]] = lshr i64 [[TMP6]], 12
; CHECK-NEXT:    [[TMP15:%.*]] = lshr i64 [[TMP11]], 12
; CHECK-NEXT:    [[TMP16:%.*]] = icmp ne i64 [[TMP12]], [[TMP13]]
; CHECK-NEXT:    [[TMP17:%.*]] = icmp ne i64 [[TMP14]], [[TMP15]]
; CHECK-NEXT:    [[TMP18:%.*]] = or i1 [[TMP16]], [[TMP17]]
; CHECK-NEXT:    br i1 [[TMP18]], label [[MISMATCH_LOOP_PRE]], label [[MISMATCH_SVE_LOOP_PREHEADER:%.*]], !prof [[PROF1:![0-9]+]]
; CHECK:       mismatch_sve_loop_preheader:
; CHECK-NEXT:    [[TMP19:%.*]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP1]], i64 [[TMP2]])
; CHECK-NEXT:    [[TMP20:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP21:%.*]] = mul nuw nsw i64 [[TMP20]], 16
; CHECK-NEXT:    br label [[MISMATCH_SVE_LOOP:%.*]]
; CHECK:       mismatch_sve_loop:
; CHECK-NEXT:    [[MISMATCH_SVE_LOOP_PRED:%.*]] = phi <vscale x 16 x i1> [ [[TMP19]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP30:%.*]], [[MISMATCH_SVE_LOOP_INC:%.*]] ]
; CHECK-NEXT:    [[MISMATCH_SVE_INDEX:%.*]] = phi i64 [ [[TMP1]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP29:%.*]], [[MISMATCH_SVE_LOOP_INC]] ]
; CHECK-NEXT:    [[TMP22:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[MISMATCH_SVE_INDEX]]
; CHECK-NEXT:    [[TMP23:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP22]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; CHECK-NEXT:    [[TMP24:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[MISMATCH_SVE_INDEX]]
; CHECK-NEXT:    [[TMP25:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP24]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; CHECK-NEXT:    [[TMP26:%.*]] = icmp ne <vscale x 16 x i8> [[TMP23]], [[TMP25]]
; CHECK-NEXT:    [[TMP27:%.*]] = select <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i1> [[TMP26]], <vscale x 16 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP28:%.*]] = call i1 @llvm.vector.reduce.or.nxv16i1(<vscale x 16 x i1> [[TMP27]])
; CHECK-NEXT:    br i1 [[TMP28]], label [[MISMATCH_SVE_LOOP_FOUND:%.*]], label [[MISMATCH_SVE_LOOP_INC]]
; CHECK:       mismatch_sve_loop_inc:
; CHECK-NEXT:    [[TMP29]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP21]]
; CHECK-NEXT:    [[TMP30]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP29]], i64 [[TMP2]])
; CHECK-NEXT:    [[TMP31:%.*]] = extractelement <vscale x 16 x i1> [[TMP30]], i64 0
; CHECK-NEXT:    br i1 [[TMP31]], label [[MISMATCH_SVE_LOOP]], label [[MISMATCH_END:%.*]]
; CHECK:       mismatch_sve_loop_found:
; CHECK-NEXT:    [[TMP32:%.*]] = and <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], [[TMP27]]
; CHECK-NEXT:    [[TMP33:%.*]] = call i32 @llvm.experimental.cttz.elts.nxv16i1(<vscale x 16 x i1> [[TMP32]])
; CHECK-NEXT:    [[TMP34:%.*]] = zext i32 [[TMP33]] to i64
; CHECK-NEXT:    [[TMP35:%.*]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP34]]
; CHECK-NEXT:    [[TMP36:%.*]] = trunc i64 [[TMP35]] to i32
; CHECK-NEXT:    br label [[MISMATCH_END]]
; CHECK:       mismatch_loop_pre:
; CHECK-NEXT:    [[MISMATCH_START_INDEX:%.*]] = phi i32 [ [[TMP0]], [[MISMATCH_MEM_CHECK]] ], [ [[TMP0]], [[MISMATCH_MIN_IT_CHECK]] ]
; CHECK-NEXT:    br label [[MISMATCH_LOOP:%.*]]
; CHECK:       mismatch_loop:
; CHECK-NEXT:    [[MISMATCH_INDEX:%.*]] = phi i32 [ [[MISMATCH_START_INDEX]], [[MISMATCH_LOOP_PRE]] ], [ [[TMP43:%.*]], [[MISMATCH_LOOP_INC:%.*]] ]
; CHECK-NEXT:    [[TMP37:%.*]] = zext i32 [[MISMATCH_INDEX]] to i64
; CHECK-NEXT:    [[TMP38:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[TMP37]]
; CHECK-NEXT:    [[TMP39:%.*]] = load i8, ptr [[TMP38]], align 1
; CHECK-NEXT:    [[TMP40:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[TMP37]]
; CHECK-NEXT:    [[TMP41:%.*]] = load i8, ptr [[TMP40]], align 1
; CHECK-NEXT:    [[TMP42:%.*]] = icmp eq i8 [[TMP39]], [[TMP41]]
; CHECK-NEXT:    br i1 [[TMP42]], label [[MISMATCH_LOOP_INC]], label [[MISMATCH_END]]
; CHECK:       mismatch_loop_inc:
; CHECK-NEXT:    [[TMP43]] = add i32 [[MISMATCH_INDEX]], 1
; CHECK-NEXT:    [[TMP44:%.*]] = icmp eq i32 [[MISMATCH_INDEX]], [[N]]
; CHECK-NEXT:    br i1 [[TMP44]], label [[MISMATCH_END]], label [[MISMATCH_LOOP]]
; CHECK:       mismatch_end:
; CHECK-NEXT:    [[MISMATCH_RESULT:%.*]] = phi i32 [ [[N]], [[MISMATCH_LOOP_INC]] ], [ [[MISMATCH_INDEX]], [[MISMATCH_LOOP]] ], [ [[N]], [[MISMATCH_SVE_LOOP_INC]] ], [ [[TMP36]], [[MISMATCH_SVE_LOOP_FOUND]] ]
; CHECK-NEXT:    br i1 true, label [[BYTE_COMPARE:%.*]], label [[WHILE_COND:%.*]]
; CHECK:       while.cond:
; CHECK-NEXT:    [[LEN_ADDR:%.*]] = phi i32 [ [[LEN]], [[MISMATCH_END]] ], [ [[MISMATCH_RESULT]], [[WHILE_BODY:%.*]] ]
; CHECK-NEXT:    [[INC:%.*]] = add i32 [[MISMATCH_RESULT]], 1
; CHECK-NEXT:    [[CMP_NOT:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], [[N]]
; CHECK-NEXT:    br i1 [[CMP_NOT]], label [[WHILE_END:%.*]], label [[WHILE_BODY]]
; CHECK:       while.body:
; CHECK-NEXT:    [[IDXPROM:%.*]] = zext i32 [[MISMATCH_RESULT]] to i64
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[IDXPROM]]
; CHECK-NEXT:    [[TMP45:%.*]] = load i8, ptr [[ARRAYIDX]], align 1
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[IDXPROM]]
; CHECK-NEXT:    [[TMP46:%.*]] = load i8, ptr [[ARRAYIDX2]], align 1
; CHECK-NEXT:    [[CMP_NOT2:%.*]] = icmp eq i8 [[TMP45]], [[TMP46]]
; CHECK-NEXT:    br i1 [[CMP_NOT2]], label [[WHILE_COND]], label [[WHILE_END]]
; CHECK:       byte.compare:
; CHECK-NEXT:    [[TMP47:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], [[N]]
; CHECK-NEXT:    br i1 [[TMP47]], label [[WHILE_END]], label [[WHILE_END]]
; CHECK:       while.end:
; CHECK-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[MISMATCH_RESULT]], [[WHILE_BODY]] ], [ [[MISMATCH_RESULT]], [[WHILE_COND]] ], [ [[MISMATCH_RESULT]], [[BYTE_COMPARE]] ], [ [[MISMATCH_RESULT]], [[BYTE_COMPARE]] ]
; CHECK-NEXT:    ret i32 [[INC_LCSSA]]
;
; LOOP-DEL-LABEL: define i32 @compare_bytes_simple
; LOOP-DEL-SAME: (ptr [[A:%.*]], ptr [[B:%.*]], i32 [[LEN:%.*]], i32 [[N:%.*]]) #[[ATTR0:[0-9]+]] {
; LOOP-DEL-NEXT:  entry:
; LOOP-DEL-NEXT:    [[TMP0:%.*]] = add i32 [[LEN]], 1
; LOOP-DEL-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; LOOP-DEL-NEXT:    [[TMP2:%.*]] = zext i32 [[N]] to i64
; LOOP-DEL-NEXT:    [[TMP3:%.*]] = icmp ule i32 [[TMP0]], [[N]]
; LOOP-DEL-NEXT:    br i1 [[TMP3]], label [[MISMATCH_MEM_CHECK:%.*]], label [[MISMATCH_LOOP_PRE:%.*]], !prof [[PROF0:![0-9]+]]
; LOOP-DEL:       mismatch_mem_check:
; LOOP-DEL-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP1]]
; LOOP-DEL-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP1]]
; LOOP-DEL-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
; LOOP-DEL-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[TMP4]] to i64
; LOOP-DEL-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP2]]
; LOOP-DEL-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP2]]
; LOOP-DEL-NEXT:    [[TMP10:%.*]] = ptrtoint ptr [[TMP8]] to i64
; LOOP-DEL-NEXT:    [[TMP11:%.*]] = ptrtoint ptr [[TMP9]] to i64
; LOOP-DEL-NEXT:    [[TMP12:%.*]] = lshr i64 [[TMP7]], 12
; LOOP-DEL-NEXT:    [[TMP13:%.*]] = lshr i64 [[TMP10]], 12
; LOOP-DEL-NEXT:    [[TMP14:%.*]] = lshr i64 [[TMP6]], 12
; LOOP-DEL-NEXT:    [[TMP15:%.*]] = lshr i64 [[TMP11]], 12
; LOOP-DEL-NEXT:    [[TMP16:%.*]] = icmp ne i64 [[TMP12]], [[TMP13]]
; LOOP-DEL-NEXT:    [[TMP17:%.*]] = icmp ne i64 [[TMP14]], [[TMP15]]
; LOOP-DEL-NEXT:    [[TMP18:%.*]] = or i1 [[TMP16]], [[TMP17]]
; LOOP-DEL-NEXT:    br i1 [[TMP18]], label [[MISMATCH_LOOP_PRE]], label [[MISMATCH_SVE_LOOP_PREHEADER:%.*]], !prof [[PROF1:![0-9]+]]
; LOOP-DEL:       mismatch_sve_loop_preheader:
; LOOP-DEL-NEXT:    [[TMP19:%.*]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP1]], i64 [[TMP2]])
; LOOP-DEL-NEXT:    [[TMP20:%.*]] = call i64 @llvm.vscale.i64()
; LOOP-DEL-NEXT:    [[TMP21:%.*]] = mul nuw nsw i64 [[TMP20]], 16
; LOOP-DEL-NEXT:    br label [[MISMATCH_SVE_LOOP:%.*]]
; LOOP-DEL:       mismatch_sve_loop:
; LOOP-DEL-NEXT:    [[MISMATCH_SVE_LOOP_PRED:%.*]] = phi <vscale x 16 x i1> [ [[TMP19]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP30:%.*]], [[MISMATCH_SVE_LOOP_INC:%.*]] ]
; LOOP-DEL-NEXT:    [[MISMATCH_SVE_INDEX:%.*]] = phi i64 [ [[TMP1]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP29:%.*]], [[MISMATCH_SVE_LOOP_INC]] ]
; LOOP-DEL-NEXT:    [[TMP22:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[MISMATCH_SVE_INDEX]]
; LOOP-DEL-NEXT:    [[TMP23:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP22]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; LOOP-DEL-NEXT:    [[TMP24:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[MISMATCH_SVE_INDEX]]
; LOOP-DEL-NEXT:    [[TMP25:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP24]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; LOOP-DEL-NEXT:    [[TMP26:%.*]] = icmp ne <vscale x 16 x i8> [[TMP23]], [[TMP25]]
; LOOP-DEL-NEXT:    [[TMP27:%.*]] = select <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i1> [[TMP26]], <vscale x 16 x i1> zeroinitializer
; LOOP-DEL-NEXT:    [[TMP28:%.*]] = call i1 @llvm.vector.reduce.or.nxv16i1(<vscale x 16 x i1> [[TMP27]])
; LOOP-DEL-NEXT:    br i1 [[TMP28]], label [[MISMATCH_SVE_LOOP_FOUND:%.*]], label [[MISMATCH_SVE_LOOP_INC]]
; LOOP-DEL:       mismatch_sve_loop_inc:
; LOOP-DEL-NEXT:    [[TMP29]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP21]]
; LOOP-DEL-NEXT:    [[TMP30]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP29]], i64 [[TMP2]])
; LOOP-DEL-NEXT:    [[TMP31:%.*]] = extractelement <vscale x 16 x i1> [[TMP30]], i64 0
; LOOP-DEL-NEXT:    br i1 [[TMP31]], label [[MISMATCH_SVE_LOOP]], label [[WHILE_END:%.*]]
; LOOP-DEL:       mismatch_sve_loop_found:
; LOOP-DEL-NEXT:    [[TMP32:%.*]] = and <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], [[TMP27]]
; LOOP-DEL-NEXT:    [[TMP33:%.*]] = call i32 @llvm.experimental.cttz.elts.nxv16i1(<vscale x 16 x i1> [[TMP32]])
; LOOP-DEL-NEXT:    [[TMP34:%.*]] = zext i32 [[TMP33]] to i64
; LOOP-DEL-NEXT:    [[TMP35:%.*]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP34]]
; LOOP-DEL-NEXT:    [[TMP36:%.*]] = trunc i64 [[TMP35]] to i32
; LOOP-DEL-NEXT:    br label [[WHILE_END]]
; LOOP-DEL:       mismatch_loop_pre:
; LOOP-DEL-NEXT:    [[MISMATCH_START_INDEX:%.*]] = phi i32 [ [[TMP0]], [[MISMATCH_MEM_CHECK]] ], [ [[TMP0]], [[ENTRY:%.*]] ]
; LOOP-DEL-NEXT:    br label [[MISMATCH_LOOP:%.*]]
; LOOP-DEL:       mismatch_loop:
; LOOP-DEL-NEXT:    [[MISMATCH_INDEX:%.*]] = phi i32 [ [[MISMATCH_START_INDEX]], [[MISMATCH_LOOP_PRE]] ], [ [[TMP43:%.*]], [[MISMATCH_LOOP_INC:%.*]] ]
; LOOP-DEL-NEXT:    [[TMP37:%.*]] = zext i32 [[MISMATCH_INDEX]] to i64
; LOOP-DEL-NEXT:    [[TMP38:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[TMP37]]
; LOOP-DEL-NEXT:    [[TMP39:%.*]] = load i8, ptr [[TMP38]], align 1
; LOOP-DEL-NEXT:    [[TMP40:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[TMP37]]
; LOOP-DEL-NEXT:    [[TMP41:%.*]] = load i8, ptr [[TMP40]], align 1
; LOOP-DEL-NEXT:    [[TMP42:%.*]] = icmp eq i8 [[TMP39]], [[TMP41]]
; LOOP-DEL-NEXT:    br i1 [[TMP42]], label [[MISMATCH_LOOP_INC]], label [[WHILE_END]]
; LOOP-DEL:       mismatch_loop_inc:
; LOOP-DEL-NEXT:    [[TMP43]] = add i32 [[MISMATCH_INDEX]], 1
; LOOP-DEL-NEXT:    [[TMP44:%.*]] = icmp eq i32 [[MISMATCH_INDEX]], [[N]]
; LOOP-DEL-NEXT:    br i1 [[TMP44]], label [[WHILE_END]], label [[MISMATCH_LOOP]]
; LOOP-DEL:       while.end:
; LOOP-DEL-NEXT:    [[MISMATCH_RESULT:%.*]] = phi i32 [ [[N]], [[MISMATCH_LOOP_INC]] ], [ [[MISMATCH_INDEX]], [[MISMATCH_LOOP]] ], [ [[N]], [[MISMATCH_SVE_LOOP_INC]] ], [ [[TMP36]], [[MISMATCH_SVE_LOOP_FOUND]] ]
; LOOP-DEL-NEXT:    ret i32 [[MISMATCH_RESULT]]
;
entry:
  br label %while.cond

while.cond:
  %len.addr = phi i32 [ %len, %entry ], [ %inc, %while.body ]
  %inc = add i32 %len.addr, 1
  %cmp.not = icmp eq i32 %inc, %n
  br i1 %cmp.not, label %while.end, label %while.body

while.body:
  %idxprom = zext i32 %inc to i64
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %idxprom
  %0 = load i8, ptr %arrayidx
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %idxprom
  %1 = load i8, ptr %arrayidx2
  %cmp.not2 = icmp eq i8 %0, %1
  br i1 %cmp.not2, label %while.cond, label %while.end

while.end:
  %inc.lcssa = phi i32 [ %inc, %while.body ], [ %inc, %while.cond ]
  ret i32 %inc.lcssa
}

define i32 @compare_bytes_umin(ptr %a, ptr %b, i32 %len, i32 %n, i32 %idx1, i32 %idx2) {
; CHECK-LABEL: define i32 @compare_bytes_umin
; CHECK-SAME: (ptr [[A:%.*]], ptr [[B:%.*]], i32 [[LEN:%.*]], i32 [[N:%.*]], i32 [[IDX1:%.*]], i32 [[IDX2:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[PH:%.*]]
; CHECK:       ph:
; CHECK-NEXT:    [[START:%.*]] = call i32 @llvm.umin.i32(i32 [[IDX1]], i32 [[IDX2]])
; CHECK-NEXT:    [[EXT:%.*]] = zext i32 [[START]] to i64
; CHECK-NEXT:    [[A0:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[EXT]]
; CHECK-NEXT:    [[TMP0:%.*]] = load i8, ptr [[A0]], align 1
; CHECK-NEXT:    [[A1:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[EXT]]
; CHECK-NEXT:    [[TMP1:%.*]] = load i8, ptr [[A1]], align 1
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[TMP0]], [[TMP1]]
; CHECK-NEXT:    br i1 [[CMP]], label [[WHILE_COND_PREHEADER:%.*]], label [[WHILE_END:%.*]]
; CHECK:       while.cond.preheader:
; CHECK-NEXT:    [[TMP2:%.*]] = add i32 [[START]], 1
; CHECK-NEXT:    br label [[MISMATCH_MIN_IT_CHECK:%.*]]
; CHECK:       mismatch_min_it_check:
; CHECK-NEXT:    [[TMP3:%.*]] = zext i32 [[TMP2]] to i64
; CHECK-NEXT:    [[TMP4:%.*]] = zext i32 [[N]] to i64
; CHECK-NEXT:    [[TMP5:%.*]] = icmp ule i32 [[TMP2]], [[N]]
; CHECK-NEXT:    br i1 [[TMP5]], label [[MISMATCH_MEM_CHECK:%.*]], label [[MISMATCH_LOOP_PRE:%.*]], !prof [[PROF0]]
; CHECK:       mismatch_mem_check:
; CHECK-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP3]]
; CHECK-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP3]]
; CHECK-NEXT:    [[TMP8:%.*]] = ptrtoint ptr [[TMP7]] to i64
; CHECK-NEXT:    [[TMP9:%.*]] = ptrtoint ptr [[TMP6]] to i64
; CHECK-NEXT:    [[TMP10:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP4]]
; CHECK-NEXT:    [[TMP11:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP4]]
; CHECK-NEXT:    [[TMP12:%.*]] = ptrtoint ptr [[TMP10]] to i64
; CHECK-NEXT:    [[TMP13:%.*]] = ptrtoint ptr [[TMP11]] to i64
; CHECK-NEXT:    [[TMP14:%.*]] = lshr i64 [[TMP9]], 12
; CHECK-NEXT:    [[TMP15:%.*]] = lshr i64 [[TMP12]], 12
; CHECK-NEXT:    [[TMP16:%.*]] = lshr i64 [[TMP8]], 12
; CHECK-NEXT:    [[TMP17:%.*]] = lshr i64 [[TMP13]], 12
; CHECK-NEXT:    [[TMP18:%.*]] = icmp ne i64 [[TMP14]], [[TMP15]]
; CHECK-NEXT:    [[TMP19:%.*]] = icmp ne i64 [[TMP16]], [[TMP17]]
; CHECK-NEXT:    [[TMP20:%.*]] = or i1 [[TMP18]], [[TMP19]]
; CHECK-NEXT:    br i1 [[TMP20]], label [[MISMATCH_LOOP_PRE]], label [[MISMATCH_SVE_LOOP_PREHEADER:%.*]], !prof [[PROF1]]
; CHECK:       mismatch_sve_loop_preheader:
; CHECK-NEXT:    [[TMP21:%.*]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP3]], i64 [[TMP4]])
; CHECK-NEXT:    [[TMP22:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP23:%.*]] = mul nuw nsw i64 [[TMP22]], 16
; CHECK-NEXT:    br label [[MISMATCH_SVE_LOOP:%.*]]
; CHECK:       mismatch_sve_loop:
; CHECK-NEXT:    [[MISMATCH_SVE_LOOP_PRED:%.*]] = phi <vscale x 16 x i1> [ [[TMP21]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP32:%.*]], [[MISMATCH_SVE_LOOP_INC:%.*]] ]
; CHECK-NEXT:    [[MISMATCH_SVE_INDEX:%.*]] = phi i64 [ [[TMP3]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP31:%.*]], [[MISMATCH_SVE_LOOP_INC]] ]
; CHECK-NEXT:    [[TMP24:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[MISMATCH_SVE_INDEX]]
; CHECK-NEXT:    [[TMP25:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP24]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; CHECK-NEXT:    [[TMP26:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[MISMATCH_SVE_INDEX]]
; CHECK-NEXT:    [[TMP27:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP26]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; CHECK-NEXT:    [[TMP28:%.*]] = icmp ne <vscale x 16 x i8> [[TMP25]], [[TMP27]]
; CHECK-NEXT:    [[TMP29:%.*]] = select <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i1> [[TMP28]], <vscale x 16 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP30:%.*]] = call i1 @llvm.vector.reduce.or.nxv16i1(<vscale x 16 x i1> [[TMP29]])
; CHECK-NEXT:    br i1 [[TMP30]], label [[MISMATCH_SVE_LOOP_FOUND:%.*]], label [[MISMATCH_SVE_LOOP_INC]]
; CHECK:       mismatch_sve_loop_inc:
; CHECK-NEXT:    [[TMP31]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP23]]
; CHECK-NEXT:    [[TMP32]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP31]], i64 [[TMP4]])
; CHECK-NEXT:    [[TMP33:%.*]] = extractelement <vscale x 16 x i1> [[TMP32]], i64 0
; CHECK-NEXT:    br i1 [[TMP33]], label [[MISMATCH_SVE_LOOP]], label [[MISMATCH_END:%.*]]
; CHECK:       mismatch_sve_loop_found:
; CHECK-NEXT:    [[TMP34:%.*]] = and <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], [[TMP29]]
; CHECK-NEXT:    [[TMP35:%.*]] = call i32 @llvm.experimental.cttz.elts.nxv16i1(<vscale x 16 x i1> [[TMP34]])
; CHECK-NEXT:    [[TMP36:%.*]] = zext i32 [[TMP35]] to i64
; CHECK-NEXT:    [[TMP37:%.*]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP36]]
; CHECK-NEXT:    [[TMP38:%.*]] = trunc i64 [[TMP37]] to i32
; CHECK-NEXT:    br label [[MISMATCH_END]]
; CHECK:       mismatch_loop_pre:
; CHECK-NEXT:    [[MISMATCH_START_INDEX:%.*]] = phi i32 [ [[TMP2]], [[MISMATCH_MEM_CHECK]] ], [ [[TMP2]], [[MISMATCH_MIN_IT_CHECK]] ]
; CHECK-NEXT:    br label [[MISMATCH_LOOP:%.*]]
; CHECK:       mismatch_loop:
; CHECK-NEXT:    [[MISMATCH_INDEX:%.*]] = phi i32 [ [[MISMATCH_START_INDEX]], [[MISMATCH_LOOP_PRE]] ], [ [[TMP45:%.*]], [[MISMATCH_LOOP_INC:%.*]] ]
; CHECK-NEXT:    [[TMP39:%.*]] = zext i32 [[MISMATCH_INDEX]] to i64
; CHECK-NEXT:    [[TMP40:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[TMP39]]
; CHECK-NEXT:    [[TMP41:%.*]] = load i8, ptr [[TMP40]], align 1
; CHECK-NEXT:    [[TMP42:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[TMP39]]
; CHECK-NEXT:    [[TMP43:%.*]] = load i8, ptr [[TMP42]], align 1
; CHECK-NEXT:    [[TMP44:%.*]] = icmp eq i8 [[TMP41]], [[TMP43]]
; CHECK-NEXT:    br i1 [[TMP44]], label [[MISMATCH_LOOP_INC]], label [[MISMATCH_END]]
; CHECK:       mismatch_loop_inc:
; CHECK-NEXT:    [[TMP45]] = add i32 [[MISMATCH_INDEX]], 1
; CHECK-NEXT:    [[TMP46:%.*]] = icmp eq i32 [[MISMATCH_INDEX]], [[N]]
; CHECK-NEXT:    br i1 [[TMP46]], label [[MISMATCH_END]], label [[MISMATCH_LOOP]]
; CHECK:       mismatch_end:
; CHECK-NEXT:    [[MISMATCH_RESULT:%.*]] = phi i32 [ [[N]], [[MISMATCH_LOOP_INC]] ], [ [[MISMATCH_INDEX]], [[MISMATCH_LOOP]] ], [ [[N]], [[MISMATCH_SVE_LOOP_INC]] ], [ [[TMP38]], [[MISMATCH_SVE_LOOP_FOUND]] ]
; CHECK-NEXT:    br i1 true, label [[BYTE_COMPARE:%.*]], label [[WHILE_COND:%.*]]
; CHECK:       while.cond:
; CHECK-NEXT:    [[LEN_PHI:%.*]] = phi i32 [ [[START]], [[MISMATCH_END]] ], [ [[MISMATCH_RESULT]], [[WHILE_BODY:%.*]] ]
; CHECK-NEXT:    [[INC:%.*]] = add i32 [[MISMATCH_RESULT]], 1
; CHECK-NEXT:    [[CMP_NOT:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], [[N]]
; CHECK-NEXT:    br i1 [[CMP_NOT]], label [[WHILE_END]], label [[WHILE_BODY]]
; CHECK:       while.body:
; CHECK-NEXT:    [[IDXPROM:%.*]] = zext i32 [[MISMATCH_RESULT]] to i64
; CHECK-NEXT:    [[IDX_A:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[IDXPROM]]
; CHECK-NEXT:    [[TMP47:%.*]] = load i8, ptr [[IDX_A]], align 1
; CHECK-NEXT:    [[IDX_B:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[IDXPROM]]
; CHECK-NEXT:    [[TMP48:%.*]] = load i8, ptr [[IDX_B]], align 1
; CHECK-NEXT:    [[CMP_NOT2:%.*]] = icmp eq i8 [[TMP47]], [[TMP48]]
; CHECK-NEXT:    br i1 [[CMP_NOT2]], label [[WHILE_COND]], label [[WHILE_END]]
; CHECK:       byte.compare:
; CHECK-NEXT:    [[TMP49:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], [[N]]
; CHECK-NEXT:    br i1 [[TMP49]], label [[WHILE_END]], label [[WHILE_END]]
; CHECK:       while.end:
; CHECK-NEXT:    [[RES:%.*]] = phi i32 [ [[N]], [[PH]] ], [ [[MISMATCH_RESULT]], [[WHILE_COND]] ], [ [[MISMATCH_RESULT]], [[WHILE_BODY]] ], [ [[MISMATCH_RESULT]], [[BYTE_COMPARE]] ], [ [[MISMATCH_RESULT]], [[BYTE_COMPARE]] ]
; CHECK-NEXT:    ret i32 [[RES]]
;
; LOOP-DEL-LABEL: define i32 @compare_bytes_umin
; LOOP-DEL-SAME: (ptr [[A:%.*]], ptr [[B:%.*]], i32 [[LEN:%.*]], i32 [[N:%.*]], i32 [[IDX1:%.*]], i32 [[IDX2:%.*]]) #[[ATTR0]] {
; LOOP-DEL-NEXT:  entry:
; LOOP-DEL-NEXT:    [[START:%.*]] = call i32 @llvm.umin.i32(i32 [[IDX1]], i32 [[IDX2]])
; LOOP-DEL-NEXT:    [[EXT:%.*]] = zext i32 [[START]] to i64
; LOOP-DEL-NEXT:    [[A0:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[EXT]]
; LOOP-DEL-NEXT:    [[TMP0:%.*]] = load i8, ptr [[A0]], align 1
; LOOP-DEL-NEXT:    [[A1:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[EXT]]
; LOOP-DEL-NEXT:    [[TMP1:%.*]] = load i8, ptr [[A1]], align 1
; LOOP-DEL-NEXT:    [[CMP:%.*]] = icmp eq i8 [[TMP0]], [[TMP1]]
; LOOP-DEL-NEXT:    br i1 [[CMP]], label [[WHILE_COND_PREHEADER:%.*]], label [[WHILE_END:%.*]]
; LOOP-DEL:       while.cond.preheader:
; LOOP-DEL-NEXT:    [[TMP2:%.*]] = add i32 [[START]], 1
; LOOP-DEL-NEXT:    [[TMP3:%.*]] = zext i32 [[TMP2]] to i64
; LOOP-DEL-NEXT:    [[TMP4:%.*]] = zext i32 [[N]] to i64
; LOOP-DEL-NEXT:    [[TMP5:%.*]] = icmp ule i32 [[TMP2]], [[N]]
; LOOP-DEL-NEXT:    br i1 [[TMP5]], label [[MISMATCH_MEM_CHECK:%.*]], label [[MISMATCH_LOOP_PRE:%.*]], !prof [[PROF0]]
; LOOP-DEL:       mismatch_mem_check:
; LOOP-DEL-NEXT:    [[TMP6:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP3]]
; LOOP-DEL-NEXT:    [[TMP7:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP3]]
; LOOP-DEL-NEXT:    [[TMP8:%.*]] = ptrtoint ptr [[TMP7]] to i64
; LOOP-DEL-NEXT:    [[TMP9:%.*]] = ptrtoint ptr [[TMP6]] to i64
; LOOP-DEL-NEXT:    [[TMP10:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP4]]
; LOOP-DEL-NEXT:    [[TMP11:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP4]]
; LOOP-DEL-NEXT:    [[TMP12:%.*]] = ptrtoint ptr [[TMP10]] to i64
; LOOP-DEL-NEXT:    [[TMP13:%.*]] = ptrtoint ptr [[TMP11]] to i64
; LOOP-DEL-NEXT:    [[TMP14:%.*]] = lshr i64 [[TMP9]], 12
; LOOP-DEL-NEXT:    [[TMP15:%.*]] = lshr i64 [[TMP12]], 12
; LOOP-DEL-NEXT:    [[TMP16:%.*]] = lshr i64 [[TMP8]], 12
; LOOP-DEL-NEXT:    [[TMP17:%.*]] = lshr i64 [[TMP13]], 12
; LOOP-DEL-NEXT:    [[TMP18:%.*]] = icmp ne i64 [[TMP14]], [[TMP15]]
; LOOP-DEL-NEXT:    [[TMP19:%.*]] = icmp ne i64 [[TMP16]], [[TMP17]]
; LOOP-DEL-NEXT:    [[TMP20:%.*]] = or i1 [[TMP18]], [[TMP19]]
; LOOP-DEL-NEXT:    br i1 [[TMP20]], label [[MISMATCH_LOOP_PRE]], label [[MISMATCH_SVE_LOOP_PREHEADER:%.*]], !prof [[PROF1]]
; LOOP-DEL:       mismatch_sve_loop_preheader:
; LOOP-DEL-NEXT:    [[TMP21:%.*]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP3]], i64 [[TMP4]])
; LOOP-DEL-NEXT:    [[TMP22:%.*]] = call i64 @llvm.vscale.i64()
; LOOP-DEL-NEXT:    [[TMP23:%.*]] = mul nuw nsw i64 [[TMP22]], 16
; LOOP-DEL-NEXT:    br label [[MISMATCH_SVE_LOOP:%.*]]
; LOOP-DEL:       mismatch_sve_loop:
; LOOP-DEL-NEXT:    [[MISMATCH_SVE_LOOP_PRED:%.*]] = phi <vscale x 16 x i1> [ [[TMP21]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP32:%.*]], [[MISMATCH_SVE_LOOP_INC:%.*]] ]
; LOOP-DEL-NEXT:    [[MISMATCH_SVE_INDEX:%.*]] = phi i64 [ [[TMP3]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP31:%.*]], [[MISMATCH_SVE_LOOP_INC]] ]
; LOOP-DEL-NEXT:    [[TMP24:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[MISMATCH_SVE_INDEX]]
; LOOP-DEL-NEXT:    [[TMP25:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP24]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; LOOP-DEL-NEXT:    [[TMP26:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[MISMATCH_SVE_INDEX]]
; LOOP-DEL-NEXT:    [[TMP27:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP26]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; LOOP-DEL-NEXT:    [[TMP28:%.*]] = icmp ne <vscale x 16 x i8> [[TMP25]], [[TMP27]]
; LOOP-DEL-NEXT:    [[TMP29:%.*]] = select <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i1> [[TMP28]], <vscale x 16 x i1> zeroinitializer
; LOOP-DEL-NEXT:    [[TMP30:%.*]] = call i1 @llvm.vector.reduce.or.nxv16i1(<vscale x 16 x i1> [[TMP29]])
; LOOP-DEL-NEXT:    br i1 [[TMP30]], label [[MISMATCH_SVE_LOOP_FOUND:%.*]], label [[MISMATCH_SVE_LOOP_INC]]
; LOOP-DEL:       mismatch_sve_loop_inc:
; LOOP-DEL-NEXT:    [[TMP31]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP23]]
; LOOP-DEL-NEXT:    [[TMP32]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP31]], i64 [[TMP4]])
; LOOP-DEL-NEXT:    [[TMP33:%.*]] = extractelement <vscale x 16 x i1> [[TMP32]], i64 0
; LOOP-DEL-NEXT:    br i1 [[TMP33]], label [[MISMATCH_SVE_LOOP]], label [[WHILE_END]]
; LOOP-DEL:       mismatch_sve_loop_found:
; LOOP-DEL-NEXT:    [[TMP34:%.*]] = and <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], [[TMP29]]
; LOOP-DEL-NEXT:    [[TMP35:%.*]] = call i32 @llvm.experimental.cttz.elts.nxv16i1(<vscale x 16 x i1> [[TMP34]])
; LOOP-DEL-NEXT:    [[TMP36:%.*]] = zext i32 [[TMP35]] to i64
; LOOP-DEL-NEXT:    [[TMP37:%.*]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP36]]
; LOOP-DEL-NEXT:    [[TMP38:%.*]] = trunc i64 [[TMP37]] to i32
; LOOP-DEL-NEXT:    br label [[WHILE_END]]
; LOOP-DEL:       mismatch_loop_pre:
; LOOP-DEL-NEXT:    [[MISMATCH_START_INDEX:%.*]] = phi i32 [ [[TMP2]], [[MISMATCH_MEM_CHECK]] ], [ [[TMP2]], [[WHILE_COND_PREHEADER]] ]
; LOOP-DEL-NEXT:    br label [[MISMATCH_LOOP:%.*]]
; LOOP-DEL:       mismatch_loop:
; LOOP-DEL-NEXT:    [[MISMATCH_INDEX:%.*]] = phi i32 [ [[MISMATCH_START_INDEX]], [[MISMATCH_LOOP_PRE]] ], [ [[TMP45:%.*]], [[MISMATCH_LOOP_INC:%.*]] ]
; LOOP-DEL-NEXT:    [[TMP39:%.*]] = zext i32 [[MISMATCH_INDEX]] to i64
; LOOP-DEL-NEXT:    [[TMP40:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[TMP39]]
; LOOP-DEL-NEXT:    [[TMP41:%.*]] = load i8, ptr [[TMP40]], align 1
; LOOP-DEL-NEXT:    [[TMP42:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[TMP39]]
; LOOP-DEL-NEXT:    [[TMP43:%.*]] = load i8, ptr [[TMP42]], align 1
; LOOP-DEL-NEXT:    [[TMP44:%.*]] = icmp eq i8 [[TMP41]], [[TMP43]]
; LOOP-DEL-NEXT:    br i1 [[TMP44]], label [[MISMATCH_LOOP_INC]], label [[WHILE_END]]
; LOOP-DEL:       mismatch_loop_inc:
; LOOP-DEL-NEXT:    [[TMP45]] = add i32 [[MISMATCH_INDEX]], 1
; LOOP-DEL-NEXT:    [[TMP46:%.*]] = icmp eq i32 [[MISMATCH_INDEX]], [[N]]
; LOOP-DEL-NEXT:    br i1 [[TMP46]], label [[WHILE_END]], label [[MISMATCH_LOOP]]
; LOOP-DEL:       while.end:
; LOOP-DEL-NEXT:    [[RES:%.*]] = phi i32 [ [[N]], [[ENTRY:%.*]] ], [ [[N]], [[MISMATCH_LOOP_INC]] ], [ [[MISMATCH_INDEX]], [[MISMATCH_LOOP]] ], [ [[N]], [[MISMATCH_SVE_LOOP_INC]] ], [ [[TMP38]], [[MISMATCH_SVE_LOOP_FOUND]] ]
; LOOP-DEL-NEXT:    ret i32 [[RES]]
;
entry:
  br label %ph

ph:
  %start = call i32 @llvm.umin.i32(i32 %idx1, i32 %idx2)
  %ext = zext i32 %start to i64
  %a0 = getelementptr inbounds i8, ptr %a, i64 %ext
  %0 = load i8, ptr %a0, align 1
  %a1 = getelementptr inbounds i8, ptr %b, i64 %ext
  %1 = load i8, ptr %a1, align 1
  %cmp = icmp eq i8 %0, %1
  br i1 %cmp, label %while.cond.preheader, label %while.end

while.cond.preheader:
  br label %while.cond

while.cond:
  %len.phi = phi i32 [ %start, %while.cond.preheader ], [ %inc, %while.body ]
  %inc = add i32 %len.phi, 1
  %cmp.not = icmp eq i32 %inc, %n
  br i1 %cmp.not, label %while.end, label %while.body

while.body:
  %idxprom = zext i32 %inc to i64
  %idx.a = getelementptr inbounds i8, ptr %a, i64 %idxprom
  %2 = load i8, ptr %idx.a, align 1
  %idx.b = getelementptr inbounds i8, ptr %b, i64 %idxprom
  %3 = load i8, ptr %idx.b, align 1
  %cmp.not2 = icmp eq i8 %2, %3
  br i1 %cmp.not2, label %while.cond, label %while.end

while.end:
  %res = phi i32 [ %n, %ph], [ %inc, %while.cond], [ %inc, %while.body ]
  ret i32 %res
}

declare i32 @llvm.umin.i32(i32, i32);

define i32 @compare_bytes_extra_cmp(ptr %a, ptr %b, i32 %len, i32 %n, i32 %x) {
; CHECK-LABEL: define i32 @compare_bytes_extra_cmp
; CHECK-SAME: (ptr [[A:%.*]], ptr [[B:%.*]], i32 [[LEN:%.*]], i32 [[N:%.*]], i32 [[X:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[CMP_X:%.*]] = icmp ult i32 [[N]], [[X]]
; CHECK-NEXT:    br i1 [[CMP_X]], label [[PH:%.*]], label [[WHILE_END:%.*]]
; CHECK:       ph:
; CHECK-NEXT:    [[TMP0:%.*]] = add i32 [[LEN]], 1
; CHECK-NEXT:    br label [[MISMATCH_MIN_IT_CHECK:%.*]]
; CHECK:       mismatch_min_it_check:
; CHECK-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; CHECK-NEXT:    [[TMP2:%.*]] = zext i32 [[N]] to i64
; CHECK-NEXT:    [[TMP3:%.*]] = icmp ule i32 [[TMP0]], [[N]]
; CHECK-NEXT:    br i1 [[TMP3]], label [[MISMATCH_MEM_CHECK:%.*]], label [[MISMATCH_LOOP_PRE:%.*]], !prof [[PROF0]]
; CHECK:       mismatch_mem_check:
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP1]]
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP1]]
; CHECK-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
; CHECK-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[TMP4]] to i64
; CHECK-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP2]]
; CHECK-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP2]]
; CHECK-NEXT:    [[TMP10:%.*]] = ptrtoint ptr [[TMP8]] to i64
; CHECK-NEXT:    [[TMP11:%.*]] = ptrtoint ptr [[TMP9]] to i64
; CHECK-NEXT:    [[TMP12:%.*]] = lshr i64 [[TMP7]], 12
; CHECK-NEXT:    [[TMP13:%.*]] = lshr i64 [[TMP10]], 12
; CHECK-NEXT:    [[TMP14:%.*]] = lshr i64 [[TMP6]], 12
; CHECK-NEXT:    [[TMP15:%.*]] = lshr i64 [[TMP11]], 12
; CHECK-NEXT:    [[TMP16:%.*]] = icmp ne i64 [[TMP12]], [[TMP13]]
; CHECK-NEXT:    [[TMP17:%.*]] = icmp ne i64 [[TMP14]], [[TMP15]]
; CHECK-NEXT:    [[TMP18:%.*]] = or i1 [[TMP16]], [[TMP17]]
; CHECK-NEXT:    br i1 [[TMP18]], label [[MISMATCH_LOOP_PRE]], label [[MISMATCH_SVE_LOOP_PREHEADER:%.*]], !prof [[PROF1]]
; CHECK:       mismatch_sve_loop_preheader:
; CHECK-NEXT:    [[TMP19:%.*]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP1]], i64 [[TMP2]])
; CHECK-NEXT:    [[TMP20:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP21:%.*]] = mul nuw nsw i64 [[TMP20]], 16
; CHECK-NEXT:    br label [[MISMATCH_SVE_LOOP:%.*]]
; CHECK:       mismatch_sve_loop:
; CHECK-NEXT:    [[MISMATCH_SVE_LOOP_PRED:%.*]] = phi <vscale x 16 x i1> [ [[TMP19]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP30:%.*]], [[MISMATCH_SVE_LOOP_INC:%.*]] ]
; CHECK-NEXT:    [[MISMATCH_SVE_INDEX:%.*]] = phi i64 [ [[TMP1]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP29:%.*]], [[MISMATCH_SVE_LOOP_INC]] ]
; CHECK-NEXT:    [[TMP22:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[MISMATCH_SVE_INDEX]]
; CHECK-NEXT:    [[TMP23:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP22]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; CHECK-NEXT:    [[TMP24:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[MISMATCH_SVE_INDEX]]
; CHECK-NEXT:    [[TMP25:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP24]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; CHECK-NEXT:    [[TMP26:%.*]] = icmp ne <vscale x 16 x i8> [[TMP23]], [[TMP25]]
; CHECK-NEXT:    [[TMP27:%.*]] = select <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i1> [[TMP26]], <vscale x 16 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP28:%.*]] = call i1 @llvm.vector.reduce.or.nxv16i1(<vscale x 16 x i1> [[TMP27]])
; CHECK-NEXT:    br i1 [[TMP28]], label [[MISMATCH_SVE_LOOP_FOUND:%.*]], label [[MISMATCH_SVE_LOOP_INC]]
; CHECK:       mismatch_sve_loop_inc:
; CHECK-NEXT:    [[TMP29]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP21]]
; CHECK-NEXT:    [[TMP30]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP29]], i64 [[TMP2]])
; CHECK-NEXT:    [[TMP31:%.*]] = extractelement <vscale x 16 x i1> [[TMP30]], i64 0
; CHECK-NEXT:    br i1 [[TMP31]], label [[MISMATCH_SVE_LOOP]], label [[MISMATCH_END:%.*]]
; CHECK:       mismatch_sve_loop_found:
; CHECK-NEXT:    [[TMP32:%.*]] = and <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], [[TMP27]]
; CHECK-NEXT:    [[TMP33:%.*]] = call i32 @llvm.experimental.cttz.elts.nxv16i1(<vscale x 16 x i1> [[TMP32]])
; CHECK-NEXT:    [[TMP34:%.*]] = zext i32 [[TMP33]] to i64
; CHECK-NEXT:    [[TMP35:%.*]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP34]]
; CHECK-NEXT:    [[TMP36:%.*]] = trunc i64 [[TMP35]] to i32
; CHECK-NEXT:    br label [[MISMATCH_END]]
; CHECK:       mismatch_loop_pre:
; CHECK-NEXT:    [[MISMATCH_START_INDEX:%.*]] = phi i32 [ [[TMP0]], [[MISMATCH_MEM_CHECK]] ], [ [[TMP0]], [[MISMATCH_MIN_IT_CHECK]] ]
; CHECK-NEXT:    br label [[MISMATCH_LOOP:%.*]]
; CHECK:       mismatch_loop:
; CHECK-NEXT:    [[MISMATCH_INDEX:%.*]] = phi i32 [ [[MISMATCH_START_INDEX]], [[MISMATCH_LOOP_PRE]] ], [ [[TMP43:%.*]], [[MISMATCH_LOOP_INC:%.*]] ]
; CHECK-NEXT:    [[TMP37:%.*]] = zext i32 [[MISMATCH_INDEX]] to i64
; CHECK-NEXT:    [[TMP38:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[TMP37]]
; CHECK-NEXT:    [[TMP39:%.*]] = load i8, ptr [[TMP38]], align 1
; CHECK-NEXT:    [[TMP40:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[TMP37]]
; CHECK-NEXT:    [[TMP41:%.*]] = load i8, ptr [[TMP40]], align 1
; CHECK-NEXT:    [[TMP42:%.*]] = icmp eq i8 [[TMP39]], [[TMP41]]
; CHECK-NEXT:    br i1 [[TMP42]], label [[MISMATCH_LOOP_INC]], label [[MISMATCH_END]]
; CHECK:       mismatch_loop_inc:
; CHECK-NEXT:    [[TMP43]] = add i32 [[MISMATCH_INDEX]], 1
; CHECK-NEXT:    [[TMP44:%.*]] = icmp eq i32 [[MISMATCH_INDEX]], [[N]]
; CHECK-NEXT:    br i1 [[TMP44]], label [[MISMATCH_END]], label [[MISMATCH_LOOP]]
; CHECK:       mismatch_end:
; CHECK-NEXT:    [[MISMATCH_RESULT:%.*]] = phi i32 [ [[N]], [[MISMATCH_LOOP_INC]] ], [ [[MISMATCH_INDEX]], [[MISMATCH_LOOP]] ], [ [[N]], [[MISMATCH_SVE_LOOP_INC]] ], [ [[TMP36]], [[MISMATCH_SVE_LOOP_FOUND]] ]
; CHECK-NEXT:    br i1 true, label [[BYTE_COMPARE:%.*]], label [[WHILE_COND:%.*]]
; CHECK:       while.cond:
; CHECK-NEXT:    [[LEN_ADDR:%.*]] = phi i32 [ [[LEN]], [[MISMATCH_END]] ], [ [[MISMATCH_RESULT]], [[WHILE_BODY:%.*]] ]
; CHECK-NEXT:    [[INC:%.*]] = add i32 [[MISMATCH_RESULT]], 1
; CHECK-NEXT:    [[CMP_NOT:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], [[N]]
; CHECK-NEXT:    br i1 [[CMP_NOT]], label [[WHILE_END]], label [[WHILE_BODY]]
; CHECK:       while.body:
; CHECK-NEXT:    [[IDXPROM:%.*]] = zext i32 [[MISMATCH_RESULT]] to i64
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[IDXPROM]]
; CHECK-NEXT:    [[TMP45:%.*]] = load i8, ptr [[ARRAYIDX]], align 1
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[IDXPROM]]
; CHECK-NEXT:    [[TMP46:%.*]] = load i8, ptr [[ARRAYIDX2]], align 1
; CHECK-NEXT:    [[CMP_NOT2:%.*]] = icmp eq i8 [[TMP45]], [[TMP46]]
; CHECK-NEXT:    br i1 [[CMP_NOT2]], label [[WHILE_COND]], label [[WHILE_END]]
; CHECK:       byte.compare:
; CHECK-NEXT:    [[TMP47:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], [[N]]
; CHECK-NEXT:    br i1 [[TMP47]], label [[WHILE_END]], label [[WHILE_END]]
; CHECK:       while.end:
; CHECK-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[MISMATCH_RESULT]], [[WHILE_BODY]] ], [ [[MISMATCH_RESULT]], [[WHILE_COND]] ], [ [[X]], [[ENTRY:%.*]] ], [ [[MISMATCH_RESULT]], [[BYTE_COMPARE]] ], [ [[MISMATCH_RESULT]], [[BYTE_COMPARE]] ]
; CHECK-NEXT:    ret i32 [[INC_LCSSA]]
;
; LOOP-DEL-LABEL: define i32 @compare_bytes_extra_cmp
; LOOP-DEL-SAME: (ptr [[A:%.*]], ptr [[B:%.*]], i32 [[LEN:%.*]], i32 [[N:%.*]], i32 [[X:%.*]]) #[[ATTR0]] {
; LOOP-DEL-NEXT:  entry:
; LOOP-DEL-NEXT:    [[CMP_X:%.*]] = icmp ult i32 [[N]], [[X]]
; LOOP-DEL-NEXT:    br i1 [[CMP_X]], label [[PH:%.*]], label [[WHILE_END:%.*]]
; LOOP-DEL:       ph:
; LOOP-DEL-NEXT:    [[TMP0:%.*]] = add i32 [[LEN]], 1
; LOOP-DEL-NEXT:    [[TMP1:%.*]] = zext i32 [[TMP0]] to i64
; LOOP-DEL-NEXT:    [[TMP2:%.*]] = zext i32 [[N]] to i64
; LOOP-DEL-NEXT:    [[TMP3:%.*]] = icmp ule i32 [[TMP0]], [[N]]
; LOOP-DEL-NEXT:    br i1 [[TMP3]], label [[MISMATCH_MEM_CHECK:%.*]], label [[MISMATCH_LOOP_PRE:%.*]], !prof [[PROF0]]
; LOOP-DEL:       mismatch_mem_check:
; LOOP-DEL-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP1]]
; LOOP-DEL-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP1]]
; LOOP-DEL-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP5]] to i64
; LOOP-DEL-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[TMP4]] to i64
; LOOP-DEL-NEXT:    [[TMP8:%.*]] = getelementptr i8, ptr [[A]], i64 [[TMP2]]
; LOOP-DEL-NEXT:    [[TMP9:%.*]] = getelementptr i8, ptr [[B]], i64 [[TMP2]]
; LOOP-DEL-NEXT:    [[TMP10:%.*]] = ptrtoint ptr [[TMP8]] to i64
; LOOP-DEL-NEXT:    [[TMP11:%.*]] = ptrtoint ptr [[TMP9]] to i64
; LOOP-DEL-NEXT:    [[TMP12:%.*]] = lshr i64 [[TMP7]], 12
; LOOP-DEL-NEXT:    [[TMP13:%.*]] = lshr i64 [[TMP10]], 12
; LOOP-DEL-NEXT:    [[TMP14:%.*]] = lshr i64 [[TMP6]], 12
; LOOP-DEL-NEXT:    [[TMP15:%.*]] = lshr i64 [[TMP11]], 12
; LOOP-DEL-NEXT:    [[TMP16:%.*]] = icmp ne i64 [[TMP12]], [[TMP13]]
; LOOP-DEL-NEXT:    [[TMP17:%.*]] = icmp ne i64 [[TMP14]], [[TMP15]]
; LOOP-DEL-NEXT:    [[TMP18:%.*]] = or i1 [[TMP16]], [[TMP17]]
; LOOP-DEL-NEXT:    br i1 [[TMP18]], label [[MISMATCH_LOOP_PRE]], label [[MISMATCH_SVE_LOOP_PREHEADER:%.*]], !prof [[PROF1]]
; LOOP-DEL:       mismatch_sve_loop_preheader:
; LOOP-DEL-NEXT:    [[TMP19:%.*]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP1]], i64 [[TMP2]])
; LOOP-DEL-NEXT:    [[TMP20:%.*]] = call i64 @llvm.vscale.i64()
; LOOP-DEL-NEXT:    [[TMP21:%.*]] = mul nuw nsw i64 [[TMP20]], 16
; LOOP-DEL-NEXT:    br label [[MISMATCH_SVE_LOOP:%.*]]
; LOOP-DEL:       mismatch_sve_loop:
; LOOP-DEL-NEXT:    [[MISMATCH_SVE_LOOP_PRED:%.*]] = phi <vscale x 16 x i1> [ [[TMP19]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP30:%.*]], [[MISMATCH_SVE_LOOP_INC:%.*]] ]
; LOOP-DEL-NEXT:    [[MISMATCH_SVE_INDEX:%.*]] = phi i64 [ [[TMP1]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP29:%.*]], [[MISMATCH_SVE_LOOP_INC]] ]
; LOOP-DEL-NEXT:    [[TMP22:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[MISMATCH_SVE_INDEX]]
; LOOP-DEL-NEXT:    [[TMP23:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP22]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; LOOP-DEL-NEXT:    [[TMP24:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[MISMATCH_SVE_INDEX]]
; LOOP-DEL-NEXT:    [[TMP25:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP24]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; LOOP-DEL-NEXT:    [[TMP26:%.*]] = icmp ne <vscale x 16 x i8> [[TMP23]], [[TMP25]]
; LOOP-DEL-NEXT:    [[TMP27:%.*]] = select <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i1> [[TMP26]], <vscale x 16 x i1> zeroinitializer
; LOOP-DEL-NEXT:    [[TMP28:%.*]] = call i1 @llvm.vector.reduce.or.nxv16i1(<vscale x 16 x i1> [[TMP27]])
; LOOP-DEL-NEXT:    br i1 [[TMP28]], label [[MISMATCH_SVE_LOOP_FOUND:%.*]], label [[MISMATCH_SVE_LOOP_INC]]
; LOOP-DEL:       mismatch_sve_loop_inc:
; LOOP-DEL-NEXT:    [[TMP29]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP21]]
; LOOP-DEL-NEXT:    [[TMP30]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP29]], i64 [[TMP2]])
; LOOP-DEL-NEXT:    [[TMP31:%.*]] = extractelement <vscale x 16 x i1> [[TMP30]], i64 0
; LOOP-DEL-NEXT:    br i1 [[TMP31]], label [[MISMATCH_SVE_LOOP]], label [[WHILE_END]]
; LOOP-DEL:       mismatch_sve_loop_found:
; LOOP-DEL-NEXT:    [[TMP32:%.*]] = and <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], [[TMP27]]
; LOOP-DEL-NEXT:    [[TMP33:%.*]] = call i32 @llvm.experimental.cttz.elts.nxv16i1(<vscale x 16 x i1> [[TMP32]])
; LOOP-DEL-NEXT:    [[TMP34:%.*]] = zext i32 [[TMP33]] to i64
; LOOP-DEL-NEXT:    [[TMP35:%.*]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP34]]
; LOOP-DEL-NEXT:    [[TMP36:%.*]] = trunc i64 [[TMP35]] to i32
; LOOP-DEL-NEXT:    br label [[WHILE_END]]
; LOOP-DEL:       mismatch_loop_pre:
; LOOP-DEL-NEXT:    [[MISMATCH_START_INDEX:%.*]] = phi i32 [ [[TMP0]], [[MISMATCH_MEM_CHECK]] ], [ [[TMP0]], [[PH]] ]
; LOOP-DEL-NEXT:    br label [[MISMATCH_LOOP:%.*]]
; LOOP-DEL:       mismatch_loop:
; LOOP-DEL-NEXT:    [[MISMATCH_INDEX:%.*]] = phi i32 [ [[MISMATCH_START_INDEX]], [[MISMATCH_LOOP_PRE]] ], [ [[TMP43:%.*]], [[MISMATCH_LOOP_INC:%.*]] ]
; LOOP-DEL-NEXT:    [[TMP37:%.*]] = zext i32 [[MISMATCH_INDEX]] to i64
; LOOP-DEL-NEXT:    [[TMP38:%.*]] = getelementptr inbounds i8, ptr [[A]], i64 [[TMP37]]
; LOOP-DEL-NEXT:    [[TMP39:%.*]] = load i8, ptr [[TMP38]], align 1
; LOOP-DEL-NEXT:    [[TMP40:%.*]] = getelementptr inbounds i8, ptr [[B]], i64 [[TMP37]]
; LOOP-DEL-NEXT:    [[TMP41:%.*]] = load i8, ptr [[TMP40]], align 1
; LOOP-DEL-NEXT:    [[TMP42:%.*]] = icmp eq i8 [[TMP39]], [[TMP41]]
; LOOP-DEL-NEXT:    br i1 [[TMP42]], label [[MISMATCH_LOOP_INC]], label [[WHILE_END]]
; LOOP-DEL:       mismatch_loop_inc:
; LOOP-DEL-NEXT:    [[TMP43]] = add i32 [[MISMATCH_INDEX]], 1
; LOOP-DEL-NEXT:    [[TMP44:%.*]] = icmp eq i32 [[MISMATCH_INDEX]], [[N]]
; LOOP-DEL-NEXT:    br i1 [[TMP44]], label [[WHILE_END]], label [[MISMATCH_LOOP]]
; LOOP-DEL:       while.end:
; LOOP-DEL-NEXT:    [[INC_LCSSA:%.*]] = phi i32 [ [[X]], [[ENTRY:%.*]] ], [ [[N]], [[MISMATCH_LOOP_INC]] ], [ [[MISMATCH_INDEX]], [[MISMATCH_LOOP]] ], [ [[N]], [[MISMATCH_SVE_LOOP_INC]] ], [ [[TMP36]], [[MISMATCH_SVE_LOOP_FOUND]] ]
; LOOP-DEL-NEXT:    ret i32 [[INC_LCSSA]]
;
entry:
  %cmp.x = icmp ult i32 %n, %x
  br i1 %cmp.x, label %ph, label %while.end

ph:
  br label %while.cond

while.cond:
  %len.addr = phi i32 [ %len, %ph ], [ %inc, %while.body ]
  %inc = add i32 %len.addr, 1
  %cmp.not = icmp eq i32 %inc, %n
  br i1 %cmp.not, label %while.end, label %while.body

while.body:
  %idxprom = zext i32 %inc to i64
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %idxprom
  %0 = load i8, ptr %arrayidx
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %idxprom
  %1 = load i8, ptr %arrayidx2
  %cmp.not2 = icmp eq i8 %0, %1
  br i1 %cmp.not2, label %while.cond, label %while.end

while.end:
  %inc.lcssa = phi i32 [ %inc, %while.body ], [ %inc, %while.cond ], [ %x, %entry ]
  ret i32 %inc.lcssa
}

define void @compare_bytes_cleanup_block(ptr %src1, ptr %src2) {
; CHECK-LABEL: define void @compare_bytes_cleanup_block
; CHECK-SAME: (ptr [[SRC1:%.*]], ptr [[SRC2:%.*]]) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[MISMATCH_MIN_IT_CHECK:%.*]]
; CHECK:       mismatch_min_it_check:
; CHECK-NEXT:    br i1 false, label [[MISMATCH_MEM_CHECK:%.*]], label [[MISMATCH_LOOP_PRE:%.*]], !prof [[PROF0]]
; CHECK:       mismatch_mem_check:
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr i8, ptr [[SRC1]], i64 1
; CHECK-NEXT:    [[TMP1:%.*]] = getelementptr i8, ptr [[SRC2]], i64 1
; CHECK-NEXT:    [[TMP2:%.*]] = ptrtoint ptr [[TMP1]] to i64
; CHECK-NEXT:    [[TMP3:%.*]] = ptrtoint ptr [[TMP0]] to i64
; CHECK-NEXT:    [[TMP4:%.*]] = getelementptr i8, ptr [[SRC1]], i64 0
; CHECK-NEXT:    [[TMP5:%.*]] = getelementptr i8, ptr [[SRC2]], i64 0
; CHECK-NEXT:    [[TMP6:%.*]] = ptrtoint ptr [[TMP4]] to i64
; CHECK-NEXT:    [[TMP7:%.*]] = ptrtoint ptr [[TMP5]] to i64
; CHECK-NEXT:    [[TMP8:%.*]] = lshr i64 [[TMP3]], 12
; CHECK-NEXT:    [[TMP9:%.*]] = lshr i64 [[TMP6]], 12
; CHECK-NEXT:    [[TMP10:%.*]] = lshr i64 [[TMP2]], 12
; CHECK-NEXT:    [[TMP11:%.*]] = lshr i64 [[TMP7]], 12
; CHECK-NEXT:    [[TMP12:%.*]] = icmp ne i64 [[TMP8]], [[TMP9]]
; CHECK-NEXT:    [[TMP13:%.*]] = icmp ne i64 [[TMP10]], [[TMP11]]
; CHECK-NEXT:    [[TMP14:%.*]] = or i1 [[TMP12]], [[TMP13]]
; CHECK-NEXT:    br i1 [[TMP14]], label [[MISMATCH_LOOP_PRE]], label [[MISMATCH_SVE_LOOP_PREHEADER:%.*]], !prof [[PROF1]]
; CHECK:       mismatch_sve_loop_preheader:
; CHECK-NEXT:    [[TMP15:%.*]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 1, i64 0)
; CHECK-NEXT:    [[TMP16:%.*]] = call i64 @llvm.vscale.i64()
; CHECK-NEXT:    [[TMP17:%.*]] = mul nuw nsw i64 [[TMP16]], 16
; CHECK-NEXT:    br label [[MISMATCH_SVE_LOOP:%.*]]
; CHECK:       mismatch_sve_loop:
; CHECK-NEXT:    [[MISMATCH_SVE_LOOP_PRED:%.*]] = phi <vscale x 16 x i1> [ [[TMP15]], [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP26:%.*]], [[MISMATCH_SVE_LOOP_INC:%.*]] ]
; CHECK-NEXT:    [[MISMATCH_SVE_INDEX:%.*]] = phi i64 [ 1, [[MISMATCH_SVE_LOOP_PREHEADER]] ], [ [[TMP25:%.*]], [[MISMATCH_SVE_LOOP_INC]] ]
; CHECK-NEXT:    [[TMP18:%.*]] = getelementptr inbounds i8, ptr [[SRC1]], i64 [[MISMATCH_SVE_INDEX]]
; CHECK-NEXT:    [[TMP19:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP18]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; CHECK-NEXT:    [[TMP20:%.*]] = getelementptr inbounds i8, ptr [[SRC2]], i64 [[MISMATCH_SVE_INDEX]]
; CHECK-NEXT:    [[TMP21:%.*]] = call <vscale x 16 x i8> @llvm.masked.load.nxv16i8.p0(ptr [[TMP20]], i32 1, <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i8> zeroinitializer)
; CHECK-NEXT:    [[TMP22:%.*]] = icmp ne <vscale x 16 x i8> [[TMP19]], [[TMP21]]
; CHECK-NEXT:    [[TMP23:%.*]] = select <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], <vscale x 16 x i1> [[TMP22]], <vscale x 16 x i1> zeroinitializer
; CHECK-NEXT:    [[TMP24:%.*]] = call i1 @llvm.vector.reduce.or.nxv16i1(<vscale x 16 x i1> [[TMP23]])
; CHECK-NEXT:    br i1 [[TMP24]], label [[MISMATCH_SVE_LOOP_FOUND:%.*]], label [[MISMATCH_SVE_LOOP_INC]]
; CHECK:       mismatch_sve_loop_inc:
; CHECK-NEXT:    [[TMP25]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP17]]
; CHECK-NEXT:    [[TMP26]] = call <vscale x 16 x i1> @llvm.get.active.lane.mask.nxv16i1.i64(i64 [[TMP25]], i64 0)
; CHECK-NEXT:    [[TMP27:%.*]] = extractelement <vscale x 16 x i1> [[TMP26]], i64 0
; CHECK-NEXT:    br i1 [[TMP27]], label [[MISMATCH_SVE_LOOP]], label [[MISMATCH_END:%.*]]
; CHECK:       mismatch_sve_loop_found:
; CHECK-NEXT:    [[TMP28:%.*]] = and <vscale x 16 x i1> [[MISMATCH_SVE_LOOP_PRED]], [[TMP23]]
; CHECK-NEXT:    [[TMP29:%.*]] = call i32 @llvm.experimental.cttz.elts.nxv16i1(<vscale x 16 x i1> [[TMP28]])
; CHECK-NEXT:    [[TMP30:%.*]] = zext i32 [[TMP29]] to i64
; CHECK-NEXT:    [[TMP31:%.*]] = add nuw nsw i64 [[MISMATCH_SVE_INDEX]], [[TMP30]]
; CHECK-NEXT:    [[TMP32:%.*]] = trunc i64 [[TMP31]] to i32
; CHECK-NEXT:    br label [[MISMATCH_END]]
; CHECK:       mismatch_loop_pre:
; CHECK-NEXT:    [[MISMATCH_START_INDEX:%.*]] = phi i32 [ 1, [[MISMATCH_MEM_CHECK]] ], [ 1, [[MISMATCH_MIN_IT_CHECK]] ]
; CHECK-NEXT:    br label [[MISMATCH_LOOP:%.*]]
; CHECK:       mismatch_loop:
; CHECK-NEXT:    [[MISMATCH_INDEX:%.*]] = phi i32 [ [[MISMATCH_START_INDEX]], [[MISMATCH_LOOP_PRE]] ], [ [[TMP39:%.*]], [[MISMATCH_LOOP_INC:%.*]] ]
; CHECK-NEXT:    [[TMP33:%.*]] = zext i32 [[MISMATCH_INDEX]] to i64
; CHECK-NEXT:    [[TMP34:%.*]] = getelementptr inbounds i8, ptr [[SRC1]], i64 [[TMP33]]
; CHECK-NEXT:    [[TMP35:%.*]] = load i8, ptr [[TMP34]], align 1
; CHECK-NEXT:    [[TMP36:%.*]] = getelementptr inbounds i8, ptr [[SRC2]], i64 [[TMP33]]
; CHECK-NEXT:    [[TMP37:%.*]] = load i8, ptr [[TMP36]], align 1
; CHECK-NEXT:    [[TMP38:%.*]] = icmp eq i8 [[TMP35]], [[TMP37]]
; CHECK-NEXT:    br i1 [[TMP38]], label [[MISMATCH_LOOP_INC]], label [[MISMATCH_END]]
; CHECK:       mismatch_loop_inc:
; CHECK-NEXT:    [[TMP39]] = add i32 [[MISMATCH_INDEX]], 1
; CHECK-NEXT:    [[TMP40:%.*]] = icmp eq i32 [[MISMATCH_INDEX]], 0
; CHECK-NEXT:    br i1 [[TMP40]], label [[MISMATCH_END]], label [[MISMATCH_LOOP]]
; CHECK:       mismatch_end:
; CHECK-NEXT:    [[MISMATCH_RESULT:%.*]] = phi i32 [ 0, [[MISMATCH_LOOP_INC]] ], [ [[MISMATCH_INDEX]], [[MISMATCH_LOOP]] ], [ 0, [[MISMATCH_SVE_LOOP_INC]] ], [ [[TMP32]], [[MISMATCH_SVE_LOOP_FOUND]] ]
; CHECK-NEXT:    br i1 true, label [[BYTE_COMPARE:%.*]], label [[WHILE_COND:%.*]]
; CHECK:       while.cond:
; CHECK-NEXT:    [[LEN:%.*]] = phi i32 [ [[MISMATCH_RESULT]], [[WHILE_BODY:%.*]] ], [ 0, [[MISMATCH_END]] ]
; CHECK-NEXT:    [[INC:%.*]] = add i32 [[MISMATCH_RESULT]], 1
; CHECK-NEXT:    [[CMP_NOT:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], 0
; CHECK-NEXT:    br i1 [[CMP_NOT]], label [[CLEANUP_THREAD:%.*]], label [[WHILE_BODY]]
; CHECK:       while.body:
; CHECK-NEXT:    [[IDXPROM:%.*]] = zext i32 [[MISMATCH_RESULT]] to i64
; CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr i8, ptr [[SRC1]], i64 [[IDXPROM]]
; CHECK-NEXT:    [[TMP41:%.*]] = load i8, ptr [[ARRAYIDX]], align 1
; CHECK-NEXT:    [[ARRAYIDX2:%.*]] = getelementptr i8, ptr [[SRC2]], i64 [[IDXPROM]]
; CHECK-NEXT:    [[TMP42:%.*]] = load i8, ptr [[ARRAYIDX2]], align 1
; CHECK-NEXT:    [[CMP_NOT2:%.*]] = icmp eq i8 [[TMP41]], [[TMP42]]
; CHECK-NEXT:    br i1 [[CMP_NOT2]], label [[WHILE_COND]], label [[IF_END:%.*]]
; CHECK:       byte.compare:
; CHECK-NEXT:    [[TMP43:%.*]] = icmp eq i32 [[MISMATCH_RESULT]], 0
; CHECK-NEXT:    br i1 [[TMP43]], label [[CLEANUP_THREAD]], label [[IF_END]]
; CHECK:       cleanup.thread:
; CHECK-NEXT:    ret void
; CHECK:       if.end:
; CHECK-NEXT:    [[RES:%.*]] = phi i32 [ [[MISMATCH_RESULT]], [[WHILE_BODY]] ], [ [[MISMATCH_RESULT]], [[BYTE_COMPARE]] ]
; CHECK-NEXT:    ret void
;
; LOOP-DEL-LABEL: define void @compare_bytes_cleanup_block
; LOOP-DEL-SAME: (ptr [[SRC1:%.*]], ptr [[SRC2:%.*]]) #[[ATTR0]] {
; LOOP-DEL-NEXT:  entry:
; LOOP-DEL-NEXT:    br label [[MISMATCH_LOOP:%.*]]
; LOOP-DEL:       mismatch_loop:
; LOOP-DEL-NEXT:    [[MISMATCH_INDEX:%.*]] = phi i32 [ 1, [[ENTRY:%.*]] ], [ [[TMP6:%.*]], [[MISMATCH_LOOP]] ]
; LOOP-DEL-NEXT:    [[TMP0:%.*]] = zext i32 [[MISMATCH_INDEX]] to i64
; LOOP-DEL-NEXT:    [[TMP1:%.*]] = getelementptr inbounds i8, ptr [[SRC1]], i64 [[TMP0]]
; LOOP-DEL-NEXT:    [[TMP2:%.*]] = load i8, ptr [[TMP1]], align 1
; LOOP-DEL-NEXT:    [[TMP3:%.*]] = getelementptr inbounds i8, ptr [[SRC2]], i64 [[TMP0]]
; LOOP-DEL-NEXT:    [[TMP4:%.*]] = load i8, ptr [[TMP3]], align 1
; LOOP-DEL-NEXT:    [[TMP5:%.*]] = icmp ne i8 [[TMP2]], [[TMP4]]
; LOOP-DEL-NEXT:    [[TMP6]] = add i32 [[MISMATCH_INDEX]], 1
; LOOP-DEL-NEXT:    [[TMP7:%.*]] = icmp eq i32 [[MISMATCH_INDEX]], 0
; LOOP-DEL-NEXT:    [[OR_COND:%.*]] = or i1 [[TMP5]], [[TMP7]]
; LOOP-DEL-NEXT:    br i1 [[OR_COND]], label [[COMMON_RET:%.*]], label [[MISMATCH_LOOP]]
; LOOP-DEL:       common.ret:
; LOOP-DEL-NEXT:    ret void
;
entry:
  br label %while.cond

while.cond:
  %len = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %inc = add i32 %len, 1
  %cmp.not = icmp eq i32 %len, 0
  br i1 %cmp.not, label %cleanup.thread, label %while.body

while.body:
  %idxprom = zext i32 %inc to i64
  %arrayidx = getelementptr i8, ptr %src1, i64 %idxprom
  %0 = load i8, ptr %arrayidx, align 1
  %arrayidx2 = getelementptr i8, ptr %src2, i64 %idxprom
  %1 = load i8, ptr %arrayidx2, align 1
  %cmp.not2 = icmp eq i8 %0, %1
  br i1 %cmp.not2, label %while.cond, label %if.end

cleanup.thread:
  ret void

if.end:
  %res = phi i32 [ %len, %while.body ]
  ret void
}

;
; NEGATIVE TESTS
;

define i32 @compare_bytes_sign_ext(ptr %a, ptr %b, i32 %len, i32 %n) {
; CHECK-LABEL: @compare_bytes_sign_ext(
; CHECK-NOT: call i32 @llvm.find.mismatch
;
; LOOP-DEL-LABEL: @compare_bytes_sign_ext(
; LOOP-DEL-NOT: call i32 @llvm.find.mismatch
;
entry:
  br label %while.cond

while.cond:
  %len.addr = phi i32 [ %len, %entry ], [ %inc, %while.body ]
  %inc = add i32 %len.addr, 1
  %cmp.not = icmp eq i32 %inc, %n
  br i1 %cmp.not, label %while.end, label %while.body

while.body:
  %idxprom = sext i32 %inc to i64
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %idxprom
  %0 = load i8, ptr %arrayidx
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %idxprom
  %1 = load i8, ptr %arrayidx2
  %cmp.not2 = icmp eq i8 %0, %1
  br i1 %cmp.not2, label %while.cond, label %while.end

while.end:
  %inc.lcssa = phi i32 [ %inc, %while.body ], [ %inc, %while.cond ]
  ret i32 %inc.lcssa
}

define i32 @compare_bytes_signed_wrap(ptr %a, ptr %b, i32 %len, i32 %n) {
; CHECK-LABEL: @compare_bytes_signed_wrap(
; CHECK-NOT: call i32 @llvm.find.mismatch
;
; LOOP-DEL-LABEL: @compare_bytes_signed_wrap(
; LOOP-DEL-NOT: call i32 @llvm.find.mismatch
;
entry:
  br label %while.cond

while.cond:
  %len.addr = phi i32 [ %len, %entry ], [ %inc, %while.body ]
  %inc = add nsw i32 %len.addr, 1
  %cmp.not = icmp eq i32 %inc, %n
  br i1 %cmp.not, label %while.end, label %while.body

while.body:
  %idxprom = zext i32 %inc to i64
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %idxprom
  %0 = load i8, ptr %arrayidx
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %idxprom
  %1 = load i8, ptr %arrayidx2
  %cmp.not2 = icmp eq i8 %0, %1
  br i1 %cmp.not2, label %while.cond, label %while.end

while.end:
  %inc.lcssa = phi i32 [ %inc, %while.body ], [ %inc, %while.cond ]
  ret i32 %inc.lcssa
}

define i32 @compare_bytes_outside_uses(ptr %a, ptr %b, i32 %len, i32 %n) {
; CHECK-LABEL: @compare_bytes_outside_uses(
; CHECK-NOT: call i32 @llvm.find.mismatch
;
; LOOP-DEL-LABEL: @compare_bytes_outside_uses(
; LOOP-DEL-NOT: call i32 @llvm.find.mismatch
;
entry:
  br label %while.cond

while.cond:
  %iv = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  %inc = add i32 %iv, 1
  %cmp.not = icmp eq i32 %inc, %len
  br i1 %cmp.not, label %while.end, label %while.body

while.body:
  %idxprom = zext i32 %inc to i64
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %idxprom
  %0 = load i8, ptr %arrayidx
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %idxprom
  %1 = load i8, ptr %arrayidx2
  %cmp.not2 = icmp eq i8 %0, %1
  br i1 %cmp.not2, label %while.cond, label %while.end

while.end:
  %res = phi i1 [ %cmp.not, %while.body ], [ %cmp.not, %while.cond ]
  %ext_res = zext i1 %res to i32
  ret i32 %ext_res
}

define i64 @compare_bytes_i64_index(ptr %a, ptr %b, i64 %len, i64 %n) {
; CHECK-LABEL: @compare_bytes_i64_index(
; CHECK-NOT: call i32 @llvm.find.mismatch
;
; LOOP-DEL-LABEL: @compare_bytes_i64_index(
; LOOP-DEL-NOT: call i32 @llvm.find.mismatch
;
entry:
  br label %while.cond

while.cond:
  %len.addr = phi i64 [ %len, %entry ], [ %inc, %while.body ]
  %inc = add i64 %len.addr, 1
  %cmp.not = icmp eq i64 %inc, %n
  br i1 %cmp.not, label %while.end, label %while.body

while.body:
  %arrayidx = getelementptr inbounds i8, ptr %a, i64 %inc
  %0 = load i8, ptr %arrayidx
  %arrayidx2 = getelementptr inbounds i8, ptr %b, i64 %inc
  %1 = load i8, ptr %arrayidx2
  %cmp.not2 = icmp eq i8 %0, %1
  br i1 %cmp.not2, label %while.cond, label %while.end

while.end:
  %inc.lcssa = phi i64 [ %inc, %while.body ], [ %inc, %while.cond ]
  ret i64 %inc.lcssa
}
