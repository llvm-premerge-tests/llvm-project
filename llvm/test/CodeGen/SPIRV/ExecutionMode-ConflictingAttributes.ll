; RUN: not --crash llc -O0 -mtriple=spirv-unknown-unknown %s -o - 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: Invalid IR: 'reqd_work_group_size' and 'hlsl.numthreads' cannot be present together.

define void @main() #1 !reqd_work_group_size !3 {
entry:
  ret void
}

attributes #1 = { "hlsl.numthreads"="4,8,16" "hlsl.shader"="compute" }
!3 = !{i32 4, i32 8, i32 12}
