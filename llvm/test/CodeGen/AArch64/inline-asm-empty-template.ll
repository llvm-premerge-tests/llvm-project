; RUN: llc -mtriple=aarch64 %s -o - | FileCheck %s

@c = global double 0.000000e+00, align 8

; CHECK:       a:                                      // @a
define void @a(double %c) {
; CHECK:      // %bb.0:                               // %entry
entry:
  %0 = load double, ptr @c, align 8
; CHECK:  	    //APP
; CHECK-NEXT:	  //NO_APP
  %1 = tail call double asm sideeffect "", "=w,0"(double %0)
  %2 = load double, ptr @c, align 8
; CHECK:  	    //APP
; CHECK-NEXT:	  //NO_APP
  %3 = tail call double asm sideeffect "", "=w,0"(double %2)
  %fneg = fneg double %1
  %cmp = fcmp oeq double %3, %fneg
  br i1 %cmp, label %if.then, label %if.end

if.then:
  tail call void @b(double noundef %1)
  br label %if.end

if.end:
  ret void
}

declare void @b(double noundef)
