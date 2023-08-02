; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

declare void @fn()
define void @test1() {
entry:
  call unwindabort void @fn()
  ret void
; CHECK: Call with unwindabort needs to be in a function with a personality.
}
