; RUN: opt -S -mtriple=aarch64-linux-gnu -mattr=+sme -aarch64-sme-abi %s | FileCheck %s
; RUN: opt -S -mtriple=aarch64-linux-gnu -mattr=+sme -aarch64-sme-abi -aarch64-sme-abi %s | FileCheck %s

declare i32 @normal_callee()

; Simple try-catch with no ZA state. No lazy-save is required, but we must restart pstate.sm in the
; exception handler.

define i32 @no_za_streaming_enabled() "aarch64_pstate_sm_enabled" personality i32 1 {
; CHECK-LABEL: define {{[^@]+}}@no_za_streaming_enabled() #1 personality i32 1 {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    invoke void @normal_callee()
; CHECK-NEXT:    to label [[RETURN:%.*]] unwind label [[LPAD:%.*]]
; CHECK:       lpad:
; CHECK-NEXT:    [[TMP0:%.*]] = landingpad { ptr, i32 }
; CHECK-NEXT:    catch ptr null
; CHECK-NEXT:    call void @llvm.aarch64.sme.invoke.resume.pstatesm(i64 1)
; CHECK-NEXT:    [[TMP1:%.*]] = extractvalue { ptr, i32 } [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = tail call ptr @__cxa_begin_catch(ptr [[TMP1]])
; CHECK-NEXT:    tail call void @__cxa_end_catch()
; CHECK-NEXT:    br label [[RETURN]]
; CHECK:       return:
; CHECK-NEXT:    [[RETVAL:%.*]] = phi i32 [ 23, [[LPAD]] ], [ 15, [[ENTRY:%.*]] ]
; CHECK-NEXT:    ret i32 [[RETVAL]]
;
entry:
  invoke void @normal_callee() to label %return unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
  catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = tail call ptr @__cxa_begin_catch(ptr %1)
  tail call void @__cxa_end_catch()
  br label %return

return:
  %retval = phi i32 [ 23, %lpad ], [ 15, %entry ]
  ret i32 %retval
}

; As above, but the function is streaming_compatible

define i32 @no_za_streaming_compatible() "aarch64_pstate_sm_compatible" personality i32 1 {
; CHECK-LABEL: define {{[^@]+}}@no_za_streaming_compatible() #2 personality i32 1 {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[TMP0:%.*]] = call aarch64_sme_preservemost_from_x2 [[TMP0]] @__arm_sme_state()
; CHECK-NEXT:    [[TMP1:%.*]] = extractvalue [[TMP0]] [[TMP0]], 0
; CHECK-NEXT:    [[TMP2:%.*]] = and i64 [[TMP1]], 1
; CHECK-NEXT:    invoke void @normal_callee()
; CHECK-NEXT:    to label [[RETURN:%.*]] unwind label [[LPAD:%.*]]
; CHECK:       lpad:
; CHECK-NEXT:    [[TMP3:%.*]] = landingpad { ptr, i32 }
; CHECK-NEXT:    catch ptr null
; CHECK-NEXT:    call void @llvm.aarch64.sme.invoke.resume.pstatesm(i64 [[TMP2]])
; CHECK-NEXT:    [[TMP4:%.*]] = extractvalue { ptr, i32 } [[TMP3]], 0
; CHECK-NEXT:    [[TMP5:%.*]] = tail call ptr @__cxa_begin_catch(ptr [[TMP4]])
; CHECK-NEXT:    tail call void @__cxa_end_catch()
; CHECK-NEXT:    br label [[RETURN]]
; CHECK:       return:
; CHECK-NEXT:    [[RETVAL:%.*]] = phi i32 [ 23, [[LPAD]] ], [ 15, [[ENTRY:%.*]] ]
; CHECK-NEXT:    ret i32 [[RETVAL]]
;
entry:
  invoke void @normal_callee() to label %return unwind label %lpad

lpad:
  %0 = landingpad { ptr, i32 }
  catch ptr null
  %1 = extractvalue { ptr, i32 } %0, 0
  %2 = tail call ptr @__cxa_begin_catch(ptr %1)
  tail call void @__cxa_end_catch()
  br label %return

return:
  %retval = phi i32 [ 23, %lpad ], [ 15, %entry ]
  ret i32 %retval
}

declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()

; CHECK: declare %0 @__arm_sme_state() #4

; CHECK: attributes #0 = { "target-features"="+sme" }
; CHECK: attributes #1 = { "aarch64_expanded_pstate_za" "aarch64_pstate_sm_enabled" "target-features"="+sme" }
; CHECK: attributes #2 = { "aarch64_expanded_pstate_za" "aarch64_pstate_sm_compatible" "target-features"="+sme" }
; CHECK: attributes #3 = { nocallback nofree nosync nounwind willreturn }
; CHECK: attributes #4 = { "aarch64_pstate_sm_compatible" "aarch64_pstate_za_preserved" }
; CHECK: attributes #5 = { "aarch64_pstate_za_preserved" }
