; Just run it through opt, no passes needed.
; RUN: opt < %s -S | FileCheck %s

define double @fpmathTest(double %a, double %b) {
entry:
  %ret = fmul double %a, %b, !fpmath !0
  ret double %ret
}

!0 = !{float 3.0}
