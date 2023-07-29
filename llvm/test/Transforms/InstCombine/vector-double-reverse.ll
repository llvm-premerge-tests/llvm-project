; RUN: opt -S -passes=instcombine < %s | FileCheck %s

define <32 x i1> @icmp32_8_eq(<32 x i8> %0) {
; CHECK-LABEL: @icmp32_8_eq
; CHECK-NEXT: [[T0:%.*]] = icmp eq <32 x i8> %0, <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
; CHECK-NEXT: ret <32 x i1> [[T0]]
  %t1 = shufflevector <32 x i8> %0, <32 x i8> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp eq <32 x i8> %t1, <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
  %t3 = shufflevector <32 x i1> %t2, <32 x i1> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <32 x i1> %t3
}

define <8 x i1> @icmp8_8_eq(<8 x i8> %0) {
; CHECK-LABEL: @icmp8_8_eq
; CHECK-NEXT: [[T0:%.*]] = icmp eq <8 x i8> %0, <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
; CHECK-NEXT: ret <8 x i1> [[T0]]
  %t1 = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp eq <8 x i8> %t1,  <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}

define <32 x i1> @icmp32_8_neq(<32 x i8> %0) {
; CHECK-LABEL: @icmp32_8_neq
; CHECK-NEXT: [[T0:%.*]] = icmp ne <32 x i8> %0, <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
; CHECK-NEXT: ret <32 x i1> [[T0]]
  %t1 = shufflevector <32 x i8> %0, <32 x i8> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp ne <32 x i8> %t1, <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
  %t3 = shufflevector <32 x i1> %t2, <32 x i1> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <32 x i1> %t3
}

define <8 x i1> @icmp8_8_neq(<8 x i8> %0) {
; CHECK-LABEL: @icmp8_8_neq
; CHECK-NEXT: [[T0:%.*]] = icmp ne <8 x i8> %0, <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
; CHECK-NEXT: ret <8 x i1> [[T0]]
  %t1 = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp ne <8 x i8> %t1,  <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}

define <32 x i1> @icmp32_8_ult(<32 x i8> %0) {
; CHECK-LABEL: @icmp32_8_ult
; CHECK-NEXT: [[T0:%.*]] = icmp ult <32 x i8> %0, <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
; CHECK-NEXT: ret <32 x i1> [[T0]]
  %t1 = shufflevector <32 x i8> %0, <32 x i8> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp ult <32 x i8> %t1, <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
  %t3 = shufflevector <32 x i1> %t2, <32 x i1> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <32 x i1> %t3
}

define <8 x i1> @icmp8_8_ult(<8 x i8> %0) {
; CHECK-LABEL: @icmp8_8_ult
; CHECK-NEXT: [[T0:%.*]] = icmp ult <8 x i8> %0, <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
; CHECK-NEXT: ret <8 x i1> [[T0]]
  %t1 = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp ult <8 x i8> %t1,  <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}

define <32 x i1> @fcmp32_8_oeq(<32 x float> %0) {
; CHECK-LABEL: @fcmp32_8_oeq
; CHECK-NEXT: [[T0:%.*]] = fcmp oeq <32 x float> %0, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
; CHECK-NEXT: ret <32 x i1> [[T0]]
  %t1 = shufflevector <32 x float> %0, <32 x float> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = fcmp oeq <32 x float> %t1, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
  %t3 = shufflevector <32 x i1> %t2, <32 x i1> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <32 x i1> %t3
}

define <8 x i1> @fcmp8_8_oeq(<8 x float> %0) {
; CHECK-LABEL: @fcmp8_8_oeq
; CHECK-NEXT: [[T0:%.*]] = fcmp oeq <8 x float> %0, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
; CHECK-NEXT: ret <8 x i1> [[T0]]
  %t1 = shufflevector <8 x float> %0, <8 x float> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = fcmp oeq <8 x float> %t1,  <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}

define <32 x i1> @fcmp32_8_olt(<32 x float> %0) {
; CHECK-LABEL: @fcmp32_8_olt
; CHECK-NEXT: [[T0:%.*]] = fcmp olt <32 x float> %0, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
; CHECK-NEXT: ret <32 x i1> [[T0]]
  %t1 = shufflevector <32 x float> %0, <32 x float> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = fcmp olt <32 x float> %t1, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
  %t3 = shufflevector <32 x i1> %t2, <32 x i1> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <32 x i1> %t3
}

define <8 x i1> @fcmp8_8_olt(<8 x float> %0) {
; CHECK-LABEL: @fcmp8_8_olt
; CHECK-NEXT: [[T0:%.*]] = fcmp olt <8 x float> %0, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
; CHECK-NEXT: ret <8 x i1> [[T0]]
  %t1 = shufflevector <8 x float> %0, <8 x float> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = fcmp olt <8 x float> %t1,  <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}

define <32 x i1> @fcmp32_8_uge(<32 x float> %0) {
; CHECK-LABEL: @fcmp32_8_uge
; CHECK-NEXT: [[T0:%.*]] = fcmp uge <32 x float> %0, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
; CHECK-NEXT: ret <32 x i1> [[T0]]
  %t1 = shufflevector <32 x float> %0, <32 x float> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = fcmp uge <32 x float> %t1, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
  %t3 = shufflevector <32 x i1> %t2, <32 x i1> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <32 x i1> %t3
}

define <8 x i1> @fcmp8_8_uge(<8 x float> %0) {
; CHECK-LABEL: @fcmp8_8_uge
; CHECK-NEXT: [[T0:%.*]] = fcmp uge <8 x float> %0, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
; CHECK-NEXT: ret <8 x i1> [[T0]]
  %t1 = shufflevector <8 x float> %0, <8 x float> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = fcmp uge <8 x float> %t1,  <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}

define <32 x i1> @fcmp32_8_ult(<32 x float> %0) {
; CHECK-LABEL: @fcmp32_8_ult
; CHECK-NEXT: [[T0:%.*]] = fcmp ult <32 x float> %0, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
; CHECK-NEXT: ret <32 x i1> [[T0]]
  %t1 = shufflevector <32 x float> %0, <32 x float> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = fcmp ult <32 x float> %t1, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
  %t3 = shufflevector <32 x i1> %t2, <32 x i1> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <32 x i1> %t3
}

define <8 x i1> @fcmp8_8_ult(<8 x float> %0) {
; CHECK-LABEL: @fcmp8_8_ult
; CHECK-NEXT: [[T0:%.*]] = fcmp ult <8 x float> %0, <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
; CHECK-NEXT: ret <8 x i1> [[T0]]
  %t1 = shufflevector <8 x float> %0, <8 x float> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = fcmp ult <8 x float> %t1,  <float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01, float 4.600000e+01>
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}


define <32 x i1> @icmp32_8_eq_constrev(<32 x i8> %0) {
; CHECK-LABEL: @icmp32_8_eq_constrev
; CHECK-NEXT: [[T0:%.*]] = icmp eq <32 x i8> %0, <i8 78, i8 77, i8 76, i8 75, i8 74, i8 73, i8 72, i8 71, i8 70, i8 69, i8 68, i8 67, i8 66, i8 65, i8 64, i8 63, i8 62, i8 61, i8 60, i8 59, i8 58, i8 57, i8 56, i8 55, i8 54, i8 53, i8 52, i8 51, i8 50, i8 49, i8 48, i8 46>
; CHECK-NEXT: ret <32 x i1> [[T0]]
  %t1 = shufflevector <32 x i8> %0, <32 x i8> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp eq <32 x i8> %t1, <i8 46, i8 48, i8 49, i8 50, i8 51, i8 52, i8 53, i8 54, i8 55, i8 56, i8 57, i8 58, i8 59, i8 60, i8 61, i8 62, i8 63, i8 64, i8 65, i8 66, i8 67, i8 68, i8 69, i8 70, i8 71, i8 72, i8 73, i8 74, i8 75, i8 76, i8 77, i8 78>
  %t3 = shufflevector <32 x i1> %t2, <32 x i1> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <32 x i1> %t3
}

define <8 x i1> @icmp8_8_eq_constrev(<8 x i8> %0) {
; CHECK-LABEL: @icmp8_8_eq_constrev
; CHECK-NEXT: [[T0:%.*]] = icmp eq <8 x i8> %0, <i8 53, i8 52, i8 51, i8 50, i8 49, i8 48, i8 47, i8 46>
; CHECK-NEXT: ret <8 x i1> [[T0]]
  %t1 = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp eq <8 x i8> %t1,  <i8 46, i8 47, i8 48, i8 49, i8 50, i8 51, i8 52, i8 53>
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}


define <32 x i1> @icmp32_8_ult_nonconst_lhs_rev(<32 x i8> %0, <32 x i8> %1) {
; CHECK-LABEL: @icmp32_8_ult_nonconst_lhs_rev
; CHECK-NEXT: [[T0:%.*]] = shufflevector <32 x i8> %1, <32 x i8> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: [[T1:%.*]] = icmp ugt <32 x i8> [[T0]], %0
; CHECK-NEXT: ret <32 x i1> [[T1]]
  %t1 = shufflevector <32 x i8> %0, <32 x i8> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp ult <32 x i8> %t1, %1
  %t3 = shufflevector <32 x i1> %t2, <32 x i1> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <32 x i1> %t3
}

define <8 x i1> @icmp8_8_ult_nonconst_lhs_rev(<8 x i8> %0, <8 x i8> %1) {
; CHECK-LABEL: @icmp8_8_ult_nonconst_lhs_rev
; CHECK-NEXT: [[T0:%.*]] = shufflevector <8 x i8> %1, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: [[T1:%.*]] = icmp ugt <8 x i8> [[T0]], %0
; CHECK-NEXT: ret <8 x i1> [[T1]]
  %t1 = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp ult <8 x i8> %t1, %1
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}

define <32 x i1> @icmp32_8_ult_nonconst_rhs_rev(<32 x i8> %0, <32 x i8> %1) {
; CHECK-LABEL: @icmp32_8_ult_nonconst_rhs_rev
; CHECK-NEXT: [[T0:%.*]] = shufflevector <32 x i8> %1, <32 x i8> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: [[T1:%.*]] = icmp ult <32 x i8> [[T0]], %0
; CHECK-NEXT: ret <32 x i1> [[T1]]
  %t1 = shufflevector <32 x i8> %0, <32 x i8> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp ult <32 x i8> %1, %t1
  %t3 = shufflevector <32 x i1> %t2, <32 x i1> poison, <32 x i32> <i32 31, i32 30, i32 29, i32 28, i32 27, i32 26, i32 25, i32 24, i32 23, i32 22, i32 21, i32 20, i32 19, i32 18, i32 17, i32 16, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <32 x i1> %t3
}

define <8 x i1> @icmp8_8_ult_nonconst_rhs_rev(<8 x i8> %0, <8 x i8> %1) {
; CHECK-LABEL: @icmp8_8_ult_nonconst_rhs_rev
; CHECK-NEXT: [[T0:%.*]] = shufflevector <8 x i8> %1, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: [[T1:%.*]] = icmp ult <8 x i8> [[T0]], %0
; CHECK-NEXT: ret <8 x i1> [[T1]]
  %t1 = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp ult <8 x i8> %1, %t1
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}

define <8 x i1> @icmp32_8_eq_negative_non_reverse1(<8 x i8> %0) {
; CHECK-LABEL: @icmp32_8_eq_negative_non_reverse1
; CHECK-NEXT: [[T0:%.*]] = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 3, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: [[T1:%.*]] = icmp eq <8 x i8> [[T0:%.*]], <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
; CHECK-NEXT: [[T2:%.*]] = shufflevector <8 x i1> [[T1]], <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: ret <8 x i1> [[T2]]
  %t1 = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 3, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp eq <8 x i8> %t1,  <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32>  <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}

define <8 x i1> @icmp32_8_eq_negative_non_reverse2(<8 x i8> %0) {
; CHECK-LABEL: @icmp32_8_eq_negative_non_reverse2
; CHECK-NEXT: [[T0:%.*]] = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: [[T1:%.*]] = icmp eq <8 x i8> [[T0:%.*]], <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
; CHECK-NEXT: [[T2:%.*]] = shufflevector <8 x i1> [[T1]], <8 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 3, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: ret <8 x i1> [[T2]]
  %t1 = shufflevector <8 x i8> %0, <8 x i8> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp eq <8 x i8> %t1,  <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
  %t3 = shufflevector <8 x i1> %t2, <8 x i1> poison, <8 x i32>  <i32 7, i32 6, i32 5, i32 3, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}

define <8 x i1> @icmp32_8_eq_negative_length_changing(<8 x i8> %0) {
; CHECK-LABEL: @icmp32_8_eq_negative_length_changing
; CHECK-NEXT: [[T0:%.*]] = shufflevector <8 x i8> %0, <8 x i8> poison, <10 x i32> <i32 poison, i32 poison, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: [[T1:%.*]] = icmp eq <10 x i8> [[T0:%.*]], <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
; CHECK-NEXT: [[T2:%.*]] = shufflevector <10 x i1> [[T1]], <10 x i1> poison, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT: ret <8 x i1> [[T2]]
  %t1 = shufflevector <8 x i8> %0, <8 x i8> poison, <10 x i32> <i32 9, i32 8, i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  %t2 = icmp eq <10 x i8> %t1,  <i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46, i8 46>
  %t3 = shufflevector <10 x i1> %t2, <10 x i1> poison, <8 x i32>  <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
  ret <8 x i1> %t3
}

