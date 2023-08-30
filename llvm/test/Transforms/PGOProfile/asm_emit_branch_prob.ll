; REQUIRES: x86_64-linux
; REQUIRES: x86-registered-target

; RUN: llc < %s -O1 -mtriple=x86_64 -filetype=obj -o %t -emit-asm-branch-probabilities
; RUN: llvm-readelf --hex-dump=.branch_probabilities %t \
; RUN:   | FileCheck %s --check-prefix=CHECK-BP-SEC

; RUN: llc < %s -O1 -mtriple=x86_64 -filetype=obj -o %t
; RUN: llvm-readelf --hex-dump=.branch_probabilities %t 2>&1 \
; RUN:   | FileCheck %s --check-prefix=CHECK-NO-BP-SEC

; When enabled, the branch probabilities are dumped with the jump address and
; the percent encoded as a char from 0 to 10000.

; // Test original source code
; void sink(int &);
; int test(int a, int b, int c) {
;   if (a > 100)
;     sink(a);
;   sink(c);
;   if (b > 100)
;     sink(b);
;   return a + b + c;
; }

define dso_local noundef i32 @_Z4testiii(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 %0, ptr %4, align 4
  store i32 %1, ptr %5, align 4
  store i32 %2, ptr %6, align 4
  %7 = icmp sgt i32 %0, 100
  br i1 %7, label %8, label %9, !prof !0

8:                                                ; preds = %3
  call void @_Z4sinkRi(ptr noundef nonnull align 4 dereferenceable(4) %4)
  br label %9

9:                                                ; preds = %8, %3
  call void @_Z4sinkRi(ptr noundef nonnull align 4 dereferenceable(4) %6)
  %10 = icmp sgt i32 %1, 100
  br i1 %10, label %11, label %13, !prof !1

11:                                               ; preds = %9
  call void @_Z4sinkRi(ptr noundef nonnull align 4 dereferenceable(4) %5)
  %12 = load i32, ptr %5, align 4
  br label %13

13:                                               ; preds = %11, %9
  %14 = phi i32 [ %12, %11 ], [ %1, %9 ]
  %15 = load i32, ptr %4, align 4
  %16 = add nsw i32 %14, %15
  %17 = load i32, ptr %6, align 4
  %18 = add nsw i32 %16, %17
  ret i32 %18
}

declare void @_Z4sinkRi(ptr noundef nonnull align 4 dereferenceable(4)) local_unnamed_addr #1


!0 = !{!"branch_weights", i32 1000, i32 3000}
!1 = !{!"branch_weights", i32 3600, i32 400}

; CHECK-BP-SEC:      Hex dump of section '.branch_probabilities':
; CHECK-BP-SEC-NEXT: 0x00000000 {{([[:xdigit:]]{8}) ([[:xdigit:]]{8})}} {{4c1d|c409}}0000 00000000 {{(.{16})}}
; CHECK-BP-SEC-NEXT: 0x00000010 {{([[:xdigit:]]{8}) ([[:xdigit:]]{8})}} {{2823|e803}}0000 00000000 {{(.{16})}}

; CHECK-NO-BP-SEC: warning: '{{.*}}': could not find section '.branch_probabilities'
