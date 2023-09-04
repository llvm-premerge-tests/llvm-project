; RUN: llc %s -stop-after=codegenprepare -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Blub = type { i8, i8, ptr }

@indirectBrPtr = external hidden global ptr

define dso_local noundef ptr @testFunc(ptr noundef readonly %array, i1 %skip) {
entry:
  br i1 %skip, label %loopHeader, label %endBlock

loopHeader:                                                ; preds = %2, %1
  %currentArrayElement = phi ptr [ %array, %entry ], [ %nextArrayElement, %loopFooter ]
  %elementValuePtr = getelementptr inbounds i8, ptr %currentArrayElement, i64 1
  %elementValue = load i8, ptr %elementValuePtr, align 1
  indirectbr ptr @indirectBrPtr, [label %loopFooter, label %endBlock]

loopFooter:
  %isGoodValue = icmp eq i8 %elementValue, 0
  ; CHECK: %nextArrayElement = getelementptr inbounds i8, ptr %elementValuePtr, i64 15
  %nextArrayElement = getelementptr inbounds %struct.Blub, ptr %currentArrayElement, i64 1
  br i1 %isGoodValue, label %loopHeader, label %endBlock

endBlock:                                                ; preds = %2
  %retVal = phi ptr [ %array, %entry ], [ %elementValuePtr, %loopFooter ], [ %elementValuePtr, %loopHeader ]
  ret ptr %retVal
}
