; RUN: llc < %s -mattr=+neon | FileCheck %s
target triple = "armebv8a-arm-none-eabihf"

define i32 @ele0(<4 x i32> %a) {
entry:
;CHECK-LABEL: ele0:
;CHECK-NEXT: .fnstart
;CHECK-NEXT: @ %bb.0:
;CHECK-NEXT: vrev64.32 q8, q0
;CHECK-NEXT: vmov.32   r0, d16[1]
;CHECK-NEXT: bx        lr

  %vget_lane = extractelement <4 x i32> %a, i64 3
  ret i32 %vget_lane
}

define i32 @ele1(<4 x i32> %a) {
entry:
;CHECK-LABEL: ele1:
;CHECK-NEXT: .fnstart
;CHECK-NEXT: @ %bb.0:
;CHECK-NEXT: vrev64.32 q8, q0
;CHECK-NEXT: vmov.32   r0, d16[0]
;CHECK-NEXT: bx        lr
  %vget_lane = extractelement <4 x i32> %a, i64 2
  ret i32 %vget_lane
}

define i32 @ele2(<4 x i32> %a) {
entry:
;CHECK-LABEL: ele2:
;CHECK-NEXT: .fnstart
;CHECK-NEXT: @ %bb.0:
;CHECK-NEXT: vrev64.32 q8, q0
;CHECK-NEXT: vmov.32   r0, d17[1]
;CHECK-NEXT: bx        lr
  %vget_lane = extractelement <4 x i32> %a, i64 1
  ret i32 %vget_lane
}

define i32 @ele3(<4 x i32> %a) {
entry:
;CHECK-LABEL: ele3:
;CHECK-NEXT: .fnstart
;CHECK-NEXT: @ %bb.0:
;CHECK-NEXT: vrev64.32 q8, q0
;CHECK-NEXT: vmov.32   r0, d17[0]
;CHECK-NEXT: bx        lr
  %vget_lane = extractelement <4 x i32> %a, i64 0
  ret i32 %vget_lane
}
