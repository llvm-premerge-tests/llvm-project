// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(linalg-update-address-space{src-addr-space=3 dest-addr-space=3}))" -verify-diagnostics

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// expected-error @+1 {{Source and destination address spaces must be different}}
func.func @linalg_generic_update_ignore_all(%arg0: memref<1x2x3x4xf32>, %arg1: memref<1x2x3x4xf32>) -> memref<1x2x3x4xf32, 3> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x3x4xf32, 3>
  linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%arg0, %arg1 : memref<1x2x3x4xf32>, memref<1x2x3x4xf32>) outs(%alloc : memref<1x2x3x4xf32, 3>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %1 = arith.addf %in, %in_1 : f32
    linalg.yield %1 : f32
  }
  return %alloc : memref<1x2x3x4xf32, 3>
}
