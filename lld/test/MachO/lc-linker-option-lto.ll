; REQUIRES: aarch64
; RUN: rm -rf %t; split-file %s %t

; RUN: llc -filetype=obj %t/q.ll -o %t/q.o
; RUN: llvm-ar cru %t/libq.a %t/q.o

; RUN: llc -filetype=obj %t/f.ll -o %t/f.nolto.o
; RUN: opt --thinlto-bc %t/f.ll -o %t/f.thinlto.o
; RUN: opt %t/f.ll -o %t/f.lto.o

; RUN: llc -filetype=obj %t/b.ll -o %t/b.nolto.o
; RUN: opt --thinlto-bc %t/b.ll -o %t/b.thinlto.o
; RUN: opt %t/b.ll -o %t/b.lto.o

; (1) NoLTO-NoLTO
; RUN: %no-arg-lld -dylib -arch arm64 -o %t/nolto-nolto.out -platform_version ios 7.0.0 7.0.0 -L%t %t/f.nolto.o %t/b.nolto.o
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/nolto-nolto.out --disassemble-symbols=_weak1 | FileCheck %s --check-prefix=WEAK1
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/nolto-nolto.out --disassemble-symbols=_weak2 | FileCheck %s --check-prefix=WEAK2

; (2) NoLTO-ThinLTO
; RUN: %no-arg-lld -dylib -arch arm64 -o %t/nolto-thinlto.out -platform_version ios 7.0.0 7.0.0 -L%t %t/f.nolto.o %t/b.thinlto.o
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/nolto-thinlto.out --disassemble-symbols=_weak1 | FileCheck %s --check-prefix=WEAK1
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/nolto-thinlto.out --disassemble-symbols=_weak2 | FileCheck %s --check-prefix=WEAK2

; (3) ThinLTO-NoLTO
; RUN: %no-arg-lld -dylib -arch arm64 -o %t/thinlto-nolto.out -platform_version ios 7.0.0 7.0.0 -L%t %t/f.thinlto.o %t/b.nolto.o
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/thinlto-nolto.out --disassemble-symbols=_weak1 | FileCheck %s --check-prefix=WEAK1
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/thinlto-nolto.out --disassemble-symbols=_weak2 | FileCheck %s --check-prefix=WEAK2

; (4) NoLTO-LTO
; RUN: %no-arg-lld -dylib -arch arm64 -o %t/nolto-lto.out -platform_version ios 7.0.0 7.0.0 -L%t %t/f.nolto.o %t/b.lto.o
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/nolto-lto.out --disassemble-symbols=_weak1 | FileCheck %s --check-prefix=WEAK1
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/nolto-lto.out --disassemble-symbols=_weak2 | FileCheck %s --check-prefix=WEAK2

; (5) LTO-NoLTO
; RUN: %no-arg-lld -dylib -arch arm64 -o %t/lto-nolto.out -platform_version ios 7.0.0 7.0.0 -L%t %t/f.lto.o %t/b.nolto.o
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/lto-nolto.out --disassemble-symbols=_weak1 | FileCheck %s --check-prefix=WEAK1
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/lto-nolto.out --disassemble-symbols=_weak2 | FileCheck %s --check-prefix=WEAK2

; (6) LTO-ThinLTO
; RUN: %no-arg-lld -dylib -arch arm64 -o %t/lto-thinlto.out -platform_version ios 7.0.0 7.0.0 -L%t %t/f.lto.o %t/b.thinlto.o
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/lto-thinlto.out --disassemble-symbols=_weak1 | FileCheck %s --check-prefix=WEAK1
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/lto-thinlto.out --disassemble-symbols=_weak2 | FileCheck %s --check-prefix=WEAK2

; (7) ThinLTO-NoLTO
; RUN: %no-arg-lld -dylib -arch arm64 -o %t/thinlto-lto.out -platform_version ios 7.0.0 7.0.0 -L%t %t/f.thinlto.o %t/b.lto.o
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/thinlto-lto.out --disassemble-symbols=_weak1 | FileCheck %s --check-prefix=WEAK1
; RUN: llvm-objdump --no-print-imm-hex --no-leading-addr --no-show-raw-insn -d %t/thinlto-lto.out --disassemble-symbols=_weak2 | FileCheck %s --check-prefix=WEAK2

; We expect to resolve _weak1 from f.ll and _weak2 from b.ll as per the input order.
; As _weak2 from q.ll pulled in via LC_LINKER_OPTION is processed
; in the second pass, it won't prevail due to _weak2 from b.ll.

; WEAK1-LABEL: <_weak1>:
; WEAK1:         bl
; WEAK2-LABEL: <_weak2>:
; WEAK2:         mov w0, #4

;--- q.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

define hidden i32 @weak2() {
entry:
  ret i32 2
}

;--- f.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

!0 = !{!"-lq"}
!llvm.linker.options = !{!0}

define weak hidden i32 @weak1() {
entry:
  %call = call i32 @weak2()
  %add = add nsw i32 %call, 1
  ret i32 %add
}

declare i32 @weak2(...)

;--- b.ll
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios"

define weak hidden i32 @weak1() {
entry:
  ret i32 3
}

define weak hidden i32 @weak2() {
entry:
  ret i32 4
}
