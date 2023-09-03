; RUN: opt -passes='forceattrs' -func-annotator-csv-file-path="%S/FunctionAnnotation.csv" -opt-level-attribute-name="opt-level" -S < %s | FileCheck %s
define void @first_function() {
; CHECK: @first_function() #0
  ret void
}

define void @second_function() {
; CHECK: @second_function() #1
  ret void
}

define void @third_function() {
; CHECK: @third_function() #0
  ret void
}

define void @fourth_function() {
; CHECK: @fourth_function() #2
  ret void
}

define void @fifth_function() {
; CHECK: @fifth_function() #3
  ret void
}

; CHECK-LABEL: attributes #0 = { "opt-level"="O1" }

; CHECK-LABEL: attributes #1 = { cold }

; CHECK-LABEL: attributes #2 = { "opt-level"="O2" }

; CHECK-LABEL: attributes #3 = { "foo"="bar" }
