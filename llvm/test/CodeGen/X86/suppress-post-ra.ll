; Check if the function attribute "disable-post-ra" suppresses the post-RA scheduler.
; Note that we ignore the first line of the print-before and print-after output,
; which should be the only difference.
;
; RUN: llc < %s -O2 -print-before=post-RA-sched 2>&1 > /dev/null - | tail -n +2 > %t1
; RUN: llc < %s -O2 -print-after=post-RA-sched 2>&1 > /dev/null - | tail -n +2 > %t2
; RUN: diff %t1 %t2
;
;
; ModuleID = 't.cpp'

@data = global i32 17, align 4
@sum = global i32 0, align 4
@zero = global i32 0, align 4
@ptr = global i32* null, align 8

define i32 @main() #0 {
entry:
  %val = alloca i32, align 4
  %0 = bitcast i32* %val to i8*
  call void @llvm.lifetime.start(i64 4, i8* %0) #3
  %1 = load i32, i32* @data, align 4, !tbaa !2
  store i32 %1, i32* %val, align 4, !tbaa !2
  call void @_Z3fooiPi(i32 1, i32* nonnull %val) #3
  call void @_Z3fooiPi(i32 2, i32* nonnull @data) #3
  %2 = load i32, i32* @zero, align 4, !tbaa !2
  call void @llvm.lifetime.end(i64 4, i8* %0) #3
  %3 = load i32, i32* %val, align 4, !tbaa !2
  call void (...) @llvm.fake.use(i32 %3)
  ret i32 %2
}

declare void @llvm.lifetime.start(i64, i8* nocapture) #1

declare void @_Z3fooiPi(i32, i32*)

declare void @llvm.lifetime.end(i64, i8* nocapture)

declare void @llvm.fake.use(...)

attributes #0 = { "disable-post-ra" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 3.8.1"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
