; RUN: opt < %s -O1 -S | FileCheck %s

%"struct.std::coroutine_handle" = type { ptr }
%"struct.std::coroutine_handle.0" = type { %"struct.std::coroutine_handle" }
%"struct.lean_future<int>::Awaiter" = type { i32, %"struct.std::coroutine_handle.0" }

declare ptr @malloc(i64)
%empty = type { i8, i8 }
declare void @produce(ptr)
declare void @consume(ptr)
declare void @consume.i8(i8)

; Although the use of %testval lives across suspend points, it might be optimized
; out by previous optimizations. Tests that the call to `@llvm.coro.opt.blocker`
; can block optimizations for %testval so that it won't be optimized out and can
; live on the coroutine frame.
define void @foo(ptr %to_store) presplitcoroutine {
entry:
  %testval = alloca %empty
  %a = alloca i8
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %alloc = call ptr @malloc(i64 16) #3
  %vFrame = call noalias nonnull ptr @llvm.coro.begin(token %id, ptr %alloc)

  call void @produce(ptr %a)
  %testval.i = getelementptr inbounds %empty, ptr %testval, i32 0, i32 0
  store i8 0, ptr %testval.i
  %testval.ii = getelementptr inbounds %empty, ptr %testval, i32 0, i32 1
  store i8 1, ptr %testval.ii

  %save = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.opt.blocker(ptr %testval)
  %suspend = call i8 @llvm.coro.suspend(token %save, i1 false)
  switch i8 %suspend, label %exit [
    i8 0, label %await.ready
    i8 1, label %exit
  ]
await.ready:
  %testval.i.l.addr = getelementptr inbounds %empty, ptr %testval, i32 0, i32 0
  %testval.i.l = load i8, ptr %testval.i.l.addr
  %testval.ii.l.addr = getelementptr inbounds %empty, ptr %testval, i32 0, i32 1
  %testval.ii.l = load i8, ptr %testval.ii.l.addr
  call void @consume.i8(i8 %testval.i.l)
  call void @consume.i8(i8 %testval.ii.l)
  call void @consume(ptr %a)
  br label %exit
exit:
  call i1 @llvm.coro.end(ptr null, i1 false)
  ret void
}

; Verify that the %testval lives on the frame.
; CHECK: %foo.Frame = type { ptr, ptr, %empty, i1, i8 }

; Check that the call to @llvm.coro.opt.blocker get erased.
; CHECK-NOT: call void @llvm.coro.opt.blocker

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare i1 @llvm.coro.alloc(token) #3
declare i64 @llvm.coro.size.i64() #5
declare ptr @llvm.coro.begin(token, ptr writeonly) #3
declare token @llvm.coro.save(ptr) #3
declare ptr @llvm.coro.frame() #5
declare i8 @llvm.coro.suspend(token, i1) #3
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #2
declare i1 @llvm.coro.end(ptr, i1) #3
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #4
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #4
declare void @llvm.coro.opt.blocker(ptr)
