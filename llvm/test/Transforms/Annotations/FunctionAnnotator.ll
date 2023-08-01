; RUN: opt -passes='module(function-annotator)' -csv-file-path="./FunctionAnnotation.csv" < %s | FileCheck %s

; CHECK_LABEL: AttributeList
; CHECK-NEXT: AttributeList[
; CHECK-NEXT:  { function => noinline nounwind ssp uwtable(sync) "O[0123]" [a-zA-Z_][a-zA-Z0-9_]["*\.\+-,]
; CHECK-NEXT: ]
