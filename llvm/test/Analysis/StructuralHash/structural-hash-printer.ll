; RUN: opt -passes='print<structural-hash>' -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes='print<structural-hash><detailed>' -disable-output %s 2>&1 | FileCheck %s -check-prefix=DETAILED-HASH

; Add a declaration so that we can test we skip it.
declare i64 @d1()

define i64 @f1(i64 %a) {
	%b = add i64 %a, 1
	ret i64 %b
}

define i32 @f2(i32 %a) {
	%b = add i32 %a, 2
	ret i32 %b
}

; CHECK: Module Hash: 39c8037f62da229
; CHECK-NEXT: Function f1 Hash: b677567fe628d557
; CHECK-NEXT: Function f2 Hash: b677567fe628d557

; DETAILED-HASH: Module Hash: {{([a-z0-9]{16})}}
; DETAILED-HASH-NEXT: Function f1 Hash: {{([a-z0-9]{16})}}
; DETAILED-HASH-NEXT: Function f2 Hash: {{([a-z0-9]{16})}}

