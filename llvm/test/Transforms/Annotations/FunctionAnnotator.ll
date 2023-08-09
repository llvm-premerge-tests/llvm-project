; RUN: opt -passes='module(function-annotator)' -func-annotator-csv-file-path="%S/FunctionAnnotation.csv" -S < %s | FileCheck %s
; CHECK-LABEL: @first_function
define void @first_function() {
  ret void
}

; CHECK-LABEL: @second_function
define void @second_function() {
  ret void
}

; CHECK-LABEL: @third_function
define void @third_function() {
  ret void
}
