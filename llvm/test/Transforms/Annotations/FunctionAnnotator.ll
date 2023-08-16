; RUN: opt -passes='function-annotator' -func-annotator-csv-file-path="%S/FunctionAnnotation.csv" -opt-level-attribute-name="opt-level" -S < %s | FileCheck %s
define void @first_function() #0 {
; CHECK: first_function
  ret void
}

define void @second_function() #1 {
; CHECK: second_function 
  ret void
}

define void @third_function() #0 {
; CHECK: third_function
  ret void
}

define void @fourth_function() #2 {
; CHECK: fourth_function
  ret void
}

define void @fifth_function() #1 {
; CHECK: fifth_function
  ret void
}

; CHECK-LABEL: attributes #0
attributes #0 = { "opt-level"="O1" }

; CHECK-LABEL: attributes #1
attributes #1 = { "opt-level"="O3" }

; CHECK-LABEL: attributes #2
attributes #2 = { "opt-level"="O2" }
