; RUN: opt -passes='function-annotator' -func-annotator-csv-file-paths="%S/FunctionAnnotation.csv" -opt-level-attribute-names="opt-level" -S < %s | FileCheck %s
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
; CHECK: @fifth_function() #1
  ret void
}

; CHECK-LABEL: attributes #0 = { "opt-level"="O1" }

; CHECK-LABEL: attributes #1 = { "opt-level"="O3" }

; CHECK-LABEL: attributes #2 = { "opt-level"="O2" }
