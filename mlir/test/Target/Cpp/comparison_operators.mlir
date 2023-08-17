// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

func.func @cmp(%arg0 : i32, %arg1 : f32, %arg2 : i64, %arg3 : f64, %arg4 : !emitc.opaque<"unsigned">, %arg5 : !emitc.opaque<"std::valarray<int>">, %arg6 : !emitc.opaque<"custom">) {
  %1 = emitc.cmp eq, %arg0, %arg0 : (i32, i32) -> i1
  %2 = emitc.cmp ne, %arg1, %arg1 : (f32, f32) -> i1
  %3 = emitc.cmp lt, %arg2, %arg2 : (i64, i64) -> i1
  %4 = emitc.cmp le, %arg3, %arg3 : (f64, f64) -> i1
  %5 = emitc.cmp gt, %arg4, %arg4 : (!emitc.opaque<"unsigned">, !emitc.opaque<"unsigned">) -> i1
  %6 = emitc.cmp ge, %arg5, %arg5 : (!emitc.opaque<"std::valarray<int>">, !emitc.opaque<"std::valarray<int>">) -> !emitc.opaque<"std::valarray<bool>">
  %7 = emitc.cmp threeway, %arg6, %arg6 : (!emitc.opaque<"custom">, !emitc.opaque<"custom">) -> !emitc.opaque<"custom">
  
  return
}
// CHECK-LABEL: void cmp
// CHECK-NEXT:  bool [[V7:[^ ]*]] = [[V0:[^ ]*]] == [[V0:[^ ]*]]
// CHECK-NEXT:  bool [[V8:[^ ]*]] = [[V1:[^ ]*]] != [[V1:[^ ]*]]
// CHECK-NEXT:  bool [[V9:[^ ]*]] = [[V2:[^ ]*]] < [[V2:[^ ]*]]
// CHECK-NEXT:  bool [[V10:[^ ]*]] = [[V3:[^ ]*]] <= [[V3:[^ ]*]]
// CHECK-NEXT:  bool [[V11:[^ ]*]] = [[V4:[^ ]*]] > [[V4:[^ ]*]]
// CHECK-NEXT:  std::valarray<bool> [[V12:[^ ]*]] = [[V5:[^ ]*]] >= [[V5:[^ ]*]]
// CHECK-NEXT:  custom [[V13:[^ ]*]] = [[V6:[^ ]*]] <=> [[V6:[^ ]*]]
