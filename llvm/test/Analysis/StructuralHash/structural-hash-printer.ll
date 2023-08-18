; RUN: opt -passes='print<structural-hash>' -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes='print<structural-hash>' -enable-detailed-structural-hash -disable-output %s 2>&1 | FileCheck %s -check-prefix=DETAILED-HASH

define i64 @f1(i64 %a) {
	%b = add i64 %a, 1
	ret i64 %b
}

define i32 @f2(i32 %a) {
	%b = add i32 %a, 2
	ret i32 %b
}

; CHECK: Module Hash: {{([a-z0-9]{16})}}
; CHECK: Function f1 Hash: {{([a-z0-9]{16})}}
; CHECK: Function f2 Hash: {{([a-z0-9]{16})}}

; DETAILED-HASH: Module Hash: {{([a-z0-9]{16})}}
; DETAILED-HASH: Function f1 Hash: {{([a-z0-9]{16})}}
; DETAILED-HASH: Function f2 Hash: {{([a-z0-9]{16})}}

