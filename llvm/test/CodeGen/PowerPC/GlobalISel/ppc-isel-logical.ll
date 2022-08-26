; RUN: llc -mtriple ppc64le-linux -global-isel -o - < %s | FileCheck %s -check-prefixes=CHECK,LINUX

; CHECK-LABEL: test_andi64:
; LINUX: and 3, 3, 4
; LINUX: blr
define i64 @test_andi64(i64 %a, i64 %b) {
  %res = and i64 %a, %b
  ret i64 %res
}

; CHECK-LABEL: test_andui32:
; LINUX: li 5, -1
; LINUX: and 3, 3, 4
; LINUX: rldic 4, 5, 0, 32
; LINUX: and 3, 3, 4
; LINUX: blr
define zeroext i32 @test_andui32(i32 zeroext %a, i32 zeroext %b) {
  %res = and i32 %a, %b
  ret i32 %res
}

; CHECK-LABEL: test_andsi32:
; LINUX: and 3, 3, 4
; LINUX: extsw 3, 3
; LINUX: blr
define signext i32 @test_andsi32(i32 signext %a, i32 signext %b) {
  %res = and i32 %a, %b
  ret i32 %res
}

; CHECK-LABEL: test_andui16:
; LINUX: li 5, -1
; LINUX: and 3, 3, 4
; LINUX: rldic 4, 5, 0, 16
; LINUX: and 3, 3, 4
; LINUX: blr
define zeroext i16 @test_andui16(i16 zeroext %a, i16 zeroext %b) {
  %res = and i16 %a, %b
  ret i16 %res
}

; CHECK-LABEL: test_andsi16:
; LINUX: and 3, 3, 4
; LINUX: extsh 3, 3
; LINUX: blr
define signext i16 @test_andsi16(i16 signext %a, i16 signext %b) {
  %res = and i16 %a, %b
  ret i16 %res
}

; CHECK-LABEL: test_andui8:
; LINUX: li 5, 255
; LINUX: and 3, 3, 4
; LINUX: and 3, 3, 5
; LINUX: blr
define zeroext i8 @test_andui8(i8 zeroext %a, i8 zeroext %b) {
  %res = and i8 %a, %b
  ret i8 %res
}

; CHECK-LABEL: test_andsi8:
; LINUX: and 3, 3, 4
; LINUX: extsb 3, 3
; LINUX: blr
define signext i8 @test_andsi8(i8 signext %a, i8 signext %b) {
  %res = and i8 %a, %b
  ret i8 %res
}

; CHECK-LABEL: test_nandi64:
; LINUX: nand 3, 3, 4
; LINUX: blr
define i64 @test_nandi64(i64 %a, i64 %b) {
  %and = and i64 %a, %b
  %neg = xor i64 %and, -1
  ret i64 %neg
}

; CHECK-LABEL: test_andci64:
; LINUX: andc 3, 3, 4
; LINUX: blr
define i64 @test_andci64(i64 %a, i64 %b) {
  %neg = xor i64 %b, -1
  %and = and i64 %a, %neg
  ret i64 %and
}

; CHECK-LABEL: test_ori64:
; LINUX: or 3, 3, 4
; LINUX: blr
define i64 @test_ori64(i64 %a, i64 %b) {
  %res = or i64 %a, %b
  ret i64 %res
}

; CHECK-LABEL: test_orci64:
; LINUX: orc 3, 3, 4
; LINUX: blr
define i64 @test_orci64(i64 %a, i64 %b) {
  %neg = xor i64 %b, -1
  %or = or i64 %a, %neg
  ret i64 %or
}

; CHECK-LABEL: test_nori64:
; LINUX: nor 3, 3, 4
; LINUX: blr
define i64 @test_nori64(i64 %a, i64 %b) {
  %or = or i64 %a, %b
  %neg = xor i64 %or, -1
  ret i64 %neg
}

; CHECK-LABEL: test_xori64:
; LINUX: xor 3, 3, 4
; LINUX: blr
define i64 @test_xori64(i64 %a, i64 %b) {
  %res = xor i64 %a, %b
  ret i64 %res
}
