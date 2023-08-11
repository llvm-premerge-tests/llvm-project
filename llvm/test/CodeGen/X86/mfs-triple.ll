; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -debug-pass=Structure -fs-profile-file=%S/Inputs/fsloader-mfs.afdo -enable-fs-discriminator=true -improved-fs-discriminator=true -split-machine-functions 2>&1 | FileCheck %s --check-prefix=MFS_ON
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -debug-pass=Structure -fs-profile-file=%S/Inputs/fsloader-mfs.afdo -enable-fs-discriminator=true -improved-fs-discriminator=true -split-machine-functions 2>&1 | FileCheck %s --check-prefix=MFS_WARN

; MFS_ON: Machine Function Splitter Transformation
; MFS_WARN: warning: -fsplit-machine-functions is only valid for X86.
; MFS_WARN_NO: Machine Function Splitter Transformation

define void @foo4(i1 zeroext %0, i1 zeroext %1) nounwind {
  br i1 %0, label %3, label %7

3:
  %4 = call i32 @bar()
  br label %7

5:
  %6 = call i32 @baz()
  br label %7

7:
  br i1 %1, label %8, label %10

8:
  %9 = call i32 @bam()
  br label %12

10:
  %11 = call i32 @baz()
  br label %12

12:
  %13 = tail call i32 @qux()
  ret void
}

declare i32 @bar()
declare i32 @baz()
declare i32 @bam()
declare i32 @qux()
