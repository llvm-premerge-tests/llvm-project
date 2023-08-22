; RUN: opt -passes='print<structural-hash>' -disable-output %s 2>&1 | FileCheck %s
; RUN: opt -passes='print<structural-hash><enable-detailed-structural-hash>' -disable-output %s 2>&1 | FileCheck %s -check-prefix=DETAILED-HASH

define i64 @f1(i64 %a) {
	%b = add i64 %a, 1
	ret i64 %b
}

define i32 @f2(i32 %a) {
	%b = add i32 %a, 2
	ret i32 %b
}

; CHECK: Module Hash: 39c8037f62da229
; CHECK: Function f1 Hash: b677567fe628d557
; CHECK: Function f2 Hash: b677567fe628d557

; DETAILED-HASH: Module Hash: 4319a994e79f670f
; DETAILED-HASH: Function f1 Hash: 7c3de1848de089eb
; DETAILED-HASH: Function f2 Hash: e8f145a33b501984

