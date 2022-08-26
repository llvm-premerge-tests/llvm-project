; RUN: llc -mtriple ppc64le-linux -global-isel -o - < %s | FileCheck %s -check-prefixes=CHECK,LINUX

; CHECK-LABEL: f1:
; LINUX: li 3, 1
; LINUX: blr
define zeroext i32 @f1() {
  ret i32 1
}

; CHECK-LABEL: f2:
; LINUX: li 3, -1
; LINUX: rldic 3, 3, 0, 32
; LINUX: blr
define zeroext i32 @f2() {
  ret i32 -1
}

; CHECK-LABEL: f3:
; LINUX: li 3, 1
; LINUX: blr
define signext i32 @f3() {
  ret i32 1
}

; CHECK-LABEL: f4:
; LINUX: li 3, -1
; LINUX: blr
define signext i32 @f4() {
  ret i32 -1
}

; CHECK-LABEL: f5:
; LINUX: li 3, -52
; LINUX: blr
define signext i32 @f5() {
  ret i32 -52
}

; CHECK-LABEL: f6:
; LINUX: li 3, -13142
; LINUX: rldic 3, 3, 0, 16
; LINUX: blr
define zeroext i32 @f6() {
  ret i32 52394
}

; CHECK-LABEL: f7:
; LINUX: lis 3, -18
; LINUX: ori 3, 3, 56780
; LINUX: rldic 3, 3, 0, 32
; LINUX: blr
define zeroext i32 @f7() {
  ret i32 4293844428
}
