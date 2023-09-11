; RUN: llc -mtriple=i386-unknown-linux-gnu -o - -verify-machineinstrs %s | FileCheck %s

define void @fn(i1 %cmp11) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %p.0 = phi i8 [ 0, %entry ], [ %p.2, %for.inc ]
  %k.0 = phi i8 [ 0, %entry ], [ %k.2, %for.inc ]
  br i1 %cmp11, label %for.cond35, label %if.end

if.end:                                           ; preds = %for.cond
  br i1 %cmp11, label %if.then13, label %if.end26

if.then13:                                        ; preds = %if.end
  br i1 %cmp11, label %for.inc, label %for.cond35

if.end26:                                         ; preds = %for.cond35, %if.end
  %p.1 = phi i8 [ %p.4, %for.cond35 ], [ %k.0, %if.end ]
  %k.1 = phi i8 [ %k.4, %for.cond35 ], [ 0, %if.end ]
  %tobool30 = icmp ne i8 %p.0, 0
  %spec.select1 = select i1 %tobool30, i8 %k.1, i8 0
  br label %for.inc

for.inc:                                          ; preds = %if.end26, %if.then13
  %p.2 = phi i8 [ poison, %if.then13 ], [ %p.1, %if.end26 ]
  %k.2 = phi i8 [ 0, %if.then13 ], [ %spec.select1, %if.end26 ]
  %0 = load i32, ptr null, align 4
  br label %for.cond

for.cond35:                                       ; preds = %if.then13, %for.cond
  %p.4 = phi i8 [ %k.0, %if.then13 ], [ 0, %for.cond ]
  %k.4 = phi i8 [ 0, %if.then13 ], [ %k.0, %for.cond ]
  %tobool36 = icmp eq i32 0, 0
  br label %if.end26
}
