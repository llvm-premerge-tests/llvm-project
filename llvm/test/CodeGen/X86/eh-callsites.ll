; RUN: llc -mtriple=x86_64-linux-gnu -global-isel=0 -fast-isel=0 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu -global-isel=1 -fast-isel=0 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-linux-gnu -global-isel=0 -fast-isel=1 < %s | FileCheck %s

;; These tests verify that we construct proper LSDA call-site tables for the normal DWARF personality style.

declare void @throw()
declare void @nothrow() #0

declare i32 @__gxx_personality_v0(...)

;; A function with only unwindabort should generate an empty callsite
;; table in the LSDA.
; CHECK-LABEL: abort_only:
; CHECK:         .cfi_personality 3, __gxx_personality_v0
; CHECK:         .cfi_lsda 3, [[EXTABLE:.Lexception[0-9]+]]
; CHECK:         callq throw
; CHECK:       [[EXTABLE]]:
; CHECK-NEXT:    .byte   255                     # @LPStart Encoding = omit
; CHECK-NEXT:    .byte   255                     # @TType Encoding = omit
; CHECK-NEXT:    .byte   1                       # Call site Encoding = uleb128
; CHECK-NEXT:    .uleb128 [[CST_END:.Lcst_end[0-9]+]]-[[CST_BEGIN:.Lcst_begin[0-9]+]]
; CHECK-NEXT:  [[CST_BEGIN]]:
; CHECK-NEXT:  [[CST_END]]:
define void @abort_only() personality ptr @__gxx_personality_v0 {
entry:
  call unwindabort void @throw()
  unreachable
}

;; A function with only a potentially-throwing calls should not
;; generate any LSDA at all.
; CHECK-LABEL: throwing_call_only:
; CHECK-NOT:   .cfi_personality
; CHECK-NOT:   .cfi_lsda
; CHECK:       callq throw
define void @throwing_call_only() personality ptr @__gxx_personality_v0 {
entry:
  call void @throw()
  unreachable
}

;; Here, we have some unwindabort calls, some plain calls that unwind
;; out of the function, and some interspersed calls to nounwind
;; functions.
;; - The potentially-unwinding callsite must be included in the
;;   callsite table.
;; - The unwindabort callsite must NOT be included in the table.
;; - Nounwind callsites can be included into any region, whatever's convenient.

; CHECK-LABEL: interspersed_nounwinds:
; CHECK:       [[BEGIN:.Lfunc_begin[0-9]+]]:
; CHECK:         .cfi_personality 3, __gxx_personality_v0
; CHECK:         .cfi_lsda 3, [[EXTABLE:.Lexception[0-9]+]]
; CHECK:       [[TMP1:.Ltmp[0-9]+]]:
; CHECK-NEXT:    callq throw
; CHECK-NEXT:  [[TMP2:.Ltmp[0-9]+]]:
; CHECK-NEXT:    callq nothrow
; CHECK-NEXT:  [[TMP3:.Ltmp[0-9]+]]:
; CHECK-NEXT:    callq throw
; CHECK-NEXT:  [[TMP4:.Ltmp[0-9]+]]:
; CHECK-NEXT:    callq throw
; CHECK-NEXT:    callq nothrow
; CHECK-NEXT:    callq throw
; CHECK-NEXT:  [[END:.Lfunc_end[0-9]+]]:
; CHECK:       [[EXTABLE]]
; CHECK-NEXT:    .byte  255                             # @LPStart Encoding = omit
; CHECK-NEXT:    .byte  255                             # @TType Encoding = omit
; CHECK-NEXT:    .byte  1                               # Call site Encoding = uleb128
; CHECK-NEXT:    .uleb128 [[CST_END:.Lcst_end[0-9]+]]-[[CST_BEGIN:.Lcst_begin[0-9]+]]
; CHECK-NEXT:  [[CST_BEGIN]]:
; CHECK-NEXT:    .uleb128 [[TMP4]]-[[BEGIN]]            # >> Call Site 1 <<
; CHECK-NEXT:    .uleb128 [[END]]-[[TMP4]]              #   Call between [[TMP4]] and [[END]]
; CHECK-NEXT:    .byte  0                               #     has no landing pad
; CHECK-NEXT:    .byte  0                               #   On action: cleanup
; CHECK-NEXT:  [[CST_END]]:


define void @interspersed_nounwinds() personality ptr @__gxx_personality_v0 {
entry:
  call unwindabort void @throw()
  call void @nothrow() #0
  call unwindabort void @throw()

  call void @throw()
  call void @nothrow() #0
  call void @throw()
  unreachable
}

;; This tests the LSDA callsite and action maps resulting from various sorts of invoke landingpads are appropriate.
;;

; CHECK-LABEL: invokes:
; CHECK:       [[BEGIN:.Lfunc_begin[0-9]+]]:
; CHECK:         .cfi_personality 3, __gxx_personality_v0
; CHECK:         .cfi_lsda 3, [[EXTABLE:.Lexception[0-9]+]]
; CHECK:       [[TMP1:.Ltmp[0-9]+]]:
; CHECK-NEXT:    callq throw
; CHECK-NEXT:  [[TMP2:.Ltmp[0-9]+]]:
; CHECK:       [[TMP3:.Ltmp[0-9]+]]:
; CHECK-NEXT:    callq throw
; CHECK-NEXT:  [[TMP4:.Ltmp[0-9]+]]:
; CHECK:       [[TMP5:.Ltmp[0-9]+]]:
; CHECK-NEXT:    callq throw
; CHECK-NEXT:  [[TMP6:.Ltmp[0-9]+]]:
; CHECK:         callq _Unwind_Resume
; CHECK:       [[TMP7:.Ltmp[0-9]+]]:
; CHECK-NEXT:  [[TMP8:.Ltmp[0-9]+]]:
; CHECK:         callq _Unwind_Resume
; CHECK:       [[END:.Lfunc_end[0-9]+]]:
; CHECK:       [[EXTABLE]]:
; CHECK-NEXT:    .byte  255                             # @LPStart Encoding = omit
; CHECK-NEXT:    .byte  3                               # @TType Encoding = udata4
; CHECK-NEXT:    .uleb128 [[TTBASE:.Lttbase[0-9]+]]-[[TTBASEREF:.Lttbaseref[0-9]+]]
; CHECK-NEXT:  [[TTBASEREF]]:
; CHECK-NEXT:    .byte  1                               # Call site Encoding = uleb128
; CHECK-NEXT:    .uleb128 [[CST_END:.Lcst_end[0-9]+]]-[[CST_BEGIN:.Lcst_begin[0-9]+]]
; CHECK-NEXT:  [[CST_BEGIN]]:
; CHECK-NEXT:    .uleb128 [[TMP1]]-[[BEGIN]]            # >> Call Site 1 <<
; CHECK-NEXT:    .uleb128 [[TMP2]]-[[TMP1]]             #   Call between [[TMP1]] and [[TMP2]]
; CHECK-NEXT:    .uleb128 {{.Ltmp[0-9]+}}-[[BEGIN]]     #     jumps to {{.Ltmp[0-9]+}}
; CHECK-NEXT:    .byte  3                               #   On action: 2
; CHECK-NEXT:    .uleb128 [[TMP3]]-[[BEGIN]]            # >> Call Site 2 <<
; CHECK-NEXT:    .uleb128 [[TMP4]]-[[TMP3]]             #   Call between [[TMP3]] and [[TMP4]]
; CHECK-NEXT:    .uleb128 {{.Ltmp[0-9]+}}-[[BEGIN]]     #     jumps to {{.Ltmp[0-9]+}}
; CHECK-NEXT:    .byte  0                               #   On action: cleanup
; CHECK-NEXT:    .uleb128 [[TMP5]]-[[BEGIN]]            # >> Call Site 3 <<
; CHECK-NEXT:    .uleb128 [[TMP6]]-[[TMP5]]             #   Call between [[TMP5]] and [[TMP6]]
; CHECK-NEXT:    .uleb128 {{.Ltmp[0-9]+}}-[[BEGIN]]     #     jumps to {{.Ltmp[0-9]+}}
; CHECK-NEXT:    .byte  5                               #   On action: 3
; CHECK-NEXT:    .uleb128 [[TMP6]]-[[BEGIN]]            # >> Call Site 4 <<
; CHECK-NEXT:    .uleb128 [[TMP8]]-[[TMP6]]             #   Call between [[TMP6]] and [[TMP8]]
; CHECK-NEXT:    .byte  0                               #     has no landing pad
; CHECK-NEXT:    .byte  0                               #   On action: cleanup
; CHECK-NEXT:  [[CST_END]]:
; CHECK-NEXT:    .byte  0                               # >> Action Record 1 <<
; CHECK-NEXT:                                           #   Cleanup
; CHECK-NEXT:    .byte  0                               #   No further actions
; CHECK-NEXT:    .byte  1                               # >> Action Record 2 <<
; CHECK-NEXT:                                           #   Catch TypeInfo 1
; CHECK-NEXT:    .byte  125                             #   Continue to action 1
; CHECK-NEXT:    .byte  1                               # >> Action Record 3 <<
; CHECK-NEXT:                                           #   Catch TypeInfo 1
; CHECK-NEXT:    .byte  0                               #   No further actions
; CHECK-NEXT:    .p2align  2
; CHECK-NEXT:                                           # >> Catch TypeInfos <<
; CHECK-NEXT:    .long  0                               # TypeInfo 1
; CHECK-NEXT:  [[TTBASE]]:

define void @invokes(i8 %x) personality ptr @__gxx_personality_v0 {
entry:
  switch i8 %x, label %one [ i8 0, label %two
                             i8 1, label %three ]
one:
  invoke void @throw()
          to label %cont unwind label %catch
two:
  invoke void @throw()
          to label %cont unwind label %cleanup
three:
  invoke void @throw()
          to label %cont unwind label %catch_cleanup

cont:
  ret void

catch:
  %0 = landingpad { ptr, i32 }
          catch ptr null
  ret void

cleanup:
  %1 = landingpad { ptr, i32 }
          cleanup
  resume { ptr, i32 } %1
catch_cleanup:
  %2 = landingpad { ptr, i32 }
          cleanup
          catch ptr null
  resume unwindabort { ptr, i32 } %2
}

attributes #0 = { nounwind }
