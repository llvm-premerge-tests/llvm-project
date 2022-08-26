; RUN: llc -mtriple ppc64le-linux -global-isel -o - < %s | FileCheck %s -check-prefixes=CHECK,LINUX

; CHECK-LABEL: test_addi8:
; LINUX: add 3, 3, 4
; LINUX: blr
define i8 @test_addi8(i8 %a, i8 %b) {
  %res = add i8 %a, %b
  ret i8 %res
}

; CHECK-LABEL: test_addi16:
; LINUX: add 3, 3, 4
; LINUX: blr
define i16 @test_addi16(i16 %a, i16 %b) {
  %res = add i16 %a, %b
  ret i16 %res
}

; CHECK-LABEL: test_addi32:
; LINUX: add 3, 3, 4
; LINUX: blr
define i32 @test_addi32(i32 %a, i32 %b) {
  %res = add i32 %a, %b
  ret i32 %res
}

; CHECK-LABEL: test_addi64:
; LINUX: add 3, 3, 4
; LINUX: blr
define i64 @test_addi64(i64 %a, i64 %b) {
  %res = add i64 %a, %b
  ret i64 %res
}

; CHECK-LABEL: test_subi8:
; LINUX: sub 3, 3, 4
; LINUX: blr
define i8 @test_subi8(i8 %a, i8 %b) {
  %res = sub i8 %a, %b
  ret i8 %res
}

; CHECK-LABEL: test_subi16:
; LINUX: sub 3, 3, 4
; LINUX: blr
define i16 @test_subi16(i16 %a, i16 %b) {
  %res = sub i16 %a, %b
  ret i16 %res
}

; CHECK-LABEL: test_subi32:
; LINUX: sub 3, 3, 4
; LINUX: blr
define i32 @test_subi32(i32 %a, i32 %b) {
  %res = sub i32 %a, %b
  ret i32 %res
}

; CHECK-LABEL: test_subi64:
; LINUX: sub 3, 3, 4
; LINUX: blr
define i64 @test_subi64(i64 %a, i64 %b) {
  %res = sub i64 %a, %b
  ret i64 %res
}
