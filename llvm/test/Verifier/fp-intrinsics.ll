; RUN: not opt -passes=verify -disable-output < %s 2>&1 | FileCheck %s

declare double @llvm.experimental.constrained.fadd.f64(double, double, metadata, metadata)
declare double @llvm.experimental.constrained.sqrt.f64(double, metadata, metadata)

; Test an illegal value for the rounding mode argument.
; CHECK: invalid rounding mode argument
; CHECK-NEXT: %fadd = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.dynomic", metadata !"fpexcept.strict")

define double @f2(double %a, double %b) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.fadd.f64(
                                          double %a, double %b,
                                          metadata !"round.dynomic",
                                          metadata !"fpexcept.strict")
  ret double %fadd
}

; Test an illegal value for the exception behavior argument.
; CHECK-NEXT: invalid exception behavior argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.fadd.f64(double %a, double %b, metadata !"round.dynamic", metadata !"fpexcept.restrict")
define double @f3(double %a, double %b) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.fadd.f64(
                                        double %a, double %b,
                                        metadata !"round.dynamic",
                                        metadata !"fpexcept.restrict")
  ret double %fadd
}

; Test an illegal value for the rounding mode argument.
; CHECK-NEXT: invalid rounding mode argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.sqrt.f64(double %a, metadata !"round.dynomic", metadata !"fpexcept.strict")
define double @f4(double %a) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.sqrt.f64(
                                          double %a,
                                          metadata !"round.dynomic",
                                          metadata !"fpexcept.strict")
  ret double %fadd
}

; Test an illegal value for the exception behavior argument.
; CHECK-NEXT: invalid exception behavior argument
; CHECK-NEXT:   %fadd = call double @llvm.experimental.constrained.sqrt.f64(double %a, metadata !"round.dynamic", metadata !"fpexcept.restrict")
define double @f5(double %a) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.sqrt.f64(
                                        double %a,
                                        metadata !"round.dynamic",
                                        metadata !"fpexcept.restrict")
  ret double %fadd
}

; Test that constrained intrinsics get the strictfp attribute by default.
; No diagnostic shall be printed.
define double @f6(double %a) #0 {
entry:
  %fadd = call double @llvm.experimental.constrained.sqrt.f64(
                                          double %a,
                                          metadata !"round.dynamic",
                                          metadata !"fpexcept.strict")
  ret double %fadd
}

; Test for mismatched function and function call attributes
; CHECK-NEXT: Constrained FP intrinsics require functions have strictfp attribute.
; CHECK-NEXT: ptr @f7
define double @f7(double %a) {
entry:
  %fadd = call double @llvm.experimental.constrained.sqrt.f64(
                                          double %a,
                                          metadata !"round.dynamic",
                                          metadata !"fpexcept.strict")
  ret double %fadd
}

; Test that strictfp attribute on function declaration counts.
; No diagnostic shall be printed.
declare double @arbitrarystrictfp.f64(double) #0
define double @f8(double %a) #0 {
entry:
  %fadd = call double @arbitrarystrictfp.f64(double %a)
  ret double %fadd
}

; Test that strictfp attribute on function declaration counts.
; No diagnostic shall be printed.
define double @f9(double %a) #0 {
entry:
  %fadd = call double @arbitrarystrictfp.f64(double %a)
  ret double %fadd
}

; Test for mismatched function and function call attributes
; CHECK-NEXT: Functions with the strictfp attribute and their contained calls and invokes must match in use of attribute strictfp!
; CHECK-NEXT: ptr @f10
; CHECK-NEXT: %sqrt = call double @llvm.sqrt.f64(double %a)
declare double @llvm.sqrt.f64(double)
define double @f10(double %a) #0 {
entry:
  %sqrt = call double @llvm.sqrt.f64(double %a)
  ret double %sqrt
}

; Test for _matched_ function and function call attributes, alias edition
; No diagnostic shall be printed.
@f10_alias = alias double (double), ptr @f10
define double @f11(double %a) #0 {
entry:
  %val = call double @f10_alias(double %a) #0
  ret double %val
}

; Test for mismatched function and function call attributes, alias edition
; CHECK-NEXT: Functions with the strictfp attribute and their contained calls and invokes must match in use of attribute strictfp!
; CHECK-NEXT: ptr @f12
; CHECK-NEXT: %val = call double @f10_alias(double %a)
define double @f12(double %a) #0 {
entry:
  %val = call double @f10_alias(double %a)
  ret double %val
}

; Test for _matched_ function and function call attributes, indirect edition
; No diagnostic shall be printed.
define double @f13(double %a, double (double)* %func) #0 {
entry:
  %val = call double %func(double %a) #0
  ret double %val
}

; Test for mismatched function and function call attributes, indirect edition
; CHECK-NEXT: Functions with the strictfp attribute and their contained calls and invokes must match in use of attribute strictfp!
; CHECK-NEXT: ptr @f14
; CHECK-NEXT: %val = call double %func(double %a)
define double @f14(double %a, double (double)* %func) #0 {
entry:
  %val = call double %func(double %a)
  ret double %val
}

; Test for mismatched function and (invoke) function call attributes
; CHECK-NEXT: Functions with the strictfp attribute and their contained calls and invokes must match in use of attribute strictfp!
; CHECK-NEXT: ptr @f100
; CHECK-NEXT:   %0 = invoke double @doublefoo(double %a)
; CHECK-NEXT:           to label %conta unwind label %contb
declare double @doublefoo(double)
declare i32 @__gxx_personality_v0(...)
define void @f100(double %a) #0 personality ptr @__gxx_personality_v0 {
entry:
  invoke double @doublefoo(double %a)
    to label %conta unwind label %contb

conta:
  ret void

contb:
  %1 = landingpad { ptr, i32 }
          filter [0 x ptr] zeroinitializer
  ret void
}

attributes #0 = { strictfp }
