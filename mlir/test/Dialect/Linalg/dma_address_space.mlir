// RUN: mlir-opt %s -pass-pipeline="builtin.module(func.func(linalg-dma-address-space{src-addr-space=1 dest-addr-space=3}))" -split-input-file | FileCheck %s


// CHECK-LABEL:   func.func @linalg_generic_dma_all_function_inputs_outputs(
// CHECK-SAME:                                                  %[[VAL_0:.*]]: memref<1x2x3x4xf32, 1>,
// CHECK-SAME:                                                  %[[VAL_1:.*]]: memref<1x2x3x4xf32, 1>) -> memref<1x2x3x4xf32, 1> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x2x3x4xf32, 1>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<1x2x3x4xf32, 3>
// CHECK:           %[[VAL_5:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           memref.dma_start %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_6]], %[[VAL_6]]], %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_6]], %[[VAL_6]]], %[[VAL_5]], %[[VAL_3]]{{\[}}%[[VAL_6]]] : memref<1x2x3x4xf32, 1>, memref<1x2x3x4xf32, 3>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_3]]{{\[}}%[[VAL_6]]], %[[VAL_5]] : memref<1xi32>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<1x2x3x4xf32, 3>
// CHECK:           %[[VAL_9:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           memref.dma_start %[[VAL_1]]{{\[}}%[[VAL_10]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]]], %[[VAL_8]]{{\[}}%[[VAL_10]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]]], %[[VAL_9]], %[[VAL_7]]{{\[}}%[[VAL_10]]] : memref<1x2x3x4xf32, 1>, memref<1x2x3x4xf32, 3>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_7]]{{\[}}%[[VAL_10]]], %[[VAL_9]] : memref<1xi32>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_12:.*]] = memref.alloc() : memref<1x2x3x4xf32, 3>
// CHECK:           linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%[[VAL_4]], %[[VAL_8]] : memref<1x2x3x4xf32, 3>, memref<1x2x3x4xf32, 3>) outs(%[[VAL_12]] : memref<1x2x3x4xf32, 3>) {
// CHECK:           ^bb0(%[[VAL_13:.*]]: f32, %[[VAL_14:.*]]: f32, %[[VAL_15:.*]]: f32):
// CHECK:             %[[VAL_16:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f32
// CHECK:             linalg.yield %[[VAL_16]] : f32
// CHECK:           }
// CHECK:           %[[VAL_17:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:           memref.dma_start %[[VAL_12]]{{\[}}%[[VAL_18]], %[[VAL_18]], %[[VAL_18]], %[[VAL_18]]], %[[VAL_2]]{{\[}}%[[VAL_18]], %[[VAL_18]], %[[VAL_18]], %[[VAL_18]]], %[[VAL_17]], %[[VAL_11]]{{\[}}%[[VAL_18]]] : memref<1x2x3x4xf32, 3>, memref<1x2x3x4xf32, 1>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_11]]{{\[}}%[[VAL_18]]], %[[VAL_17]] : memref<1xi32>
// CHECK:           return %[[VAL_2]] : memref<1x2x3x4xf32, 1>
// CHECK:         }
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @linalg_generic_dma_all_function_inputs_outputs(%arg0: memref<1x2x3x4xf32, 1>, %arg1: memref<1x2x3x4xf32, 1>) -> memref<1x2x3x4xf32, 1> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x3x4xf32, 1>
  linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%arg0, %arg1 : memref<1x2x3x4xf32, 1>, memref<1x2x3x4xf32, 1>) outs(%alloc : memref<1x2x3x4xf32, 1>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %1 = arith.addf %in, %in_1 : f32
    linalg.yield %1 : f32
  }
  return %alloc : memref<1x2x3x4xf32, 1>
}

// -----

// CHECK-LABEL:   func.func @linalg_generic_dma_ignore_inputs_addr_space(
// CHECK-SAME:                                                                  %[[VAL_0:.*]]: memref<1x2x3x4xf32, 2>,
// CHECK-SAME:                                                                  %[[VAL_1:.*]]: memref<1x2x3x4xf32, 2>) -> memref<1x2x3x4xf32, 1> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x2x3x4xf32, 1>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<1x2x3x4xf32, 3>
// CHECK:           linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%[[VAL_0]], %[[VAL_1]] : memref<1x2x3x4xf32, 2>, memref<1x2x3x4xf32, 2>) outs(%[[VAL_4]] : memref<1x2x3x4xf32, 3>) {
// CHECK:           ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32, %[[VAL_7:.*]]: f32):
// CHECK:             %[[VAL_8:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f32
// CHECK:             linalg.yield %[[VAL_8]] : f32
// CHECK:           }
// CHECK:           %[[VAL_9:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           memref.dma_start %[[VAL_4]]{{\[}}%[[VAL_10]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]]], %[[VAL_2]]{{\[}}%[[VAL_10]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]]], %[[VAL_9]], %[[VAL_3]]{{\[}}%[[VAL_10]]] : memref<1x2x3x4xf32, 3>, memref<1x2x3x4xf32, 1>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_3]]{{\[}}%[[VAL_10]]], %[[VAL_9]] : memref<1xi32>
// CHECK:           return %[[VAL_2]] : memref<1x2x3x4xf32, 1>
// CHECK:         }
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @linalg_generic_dma_ignore_inputs_addr_space(%arg0: memref<1x2x3x4xf32, 2>, %arg1: memref<1x2x3x4xf32, 2>) -> memref<1x2x3x4xf32, 1> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x3x4xf32, 1>
  linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%arg0, %arg1 : memref<1x2x3x4xf32, 2>, memref<1x2x3x4xf32, 2>) outs(%alloc : memref<1x2x3x4xf32, 1>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %1 = arith.addf %in, %in_1 : f32
    linalg.yield %1 : f32
  }
  return %alloc : memref<1x2x3x4xf32, 1>
}

// -----

// CHECK-LABEL:   func.func @linalg_generic_dma_ignore_outputs_addr_space(
// CHECK-SAME:                                                            %[[VAL_0:.*]]: memref<1x2x3x4xf32, 1>,
// CHECK-SAME:                                                            %[[VAL_1:.*]]: memref<1x2x3x4xf32, 1>) -> memref<1x2x3x4xf32, 3> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x2x3x4xf32, 3>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<1x2x3x4xf32, 3>
// CHECK:           %[[VAL_5:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           memref.dma_start %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_6]], %[[VAL_6]]], %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_6]], %[[VAL_6]], %[[VAL_6]]], %[[VAL_5]], %[[VAL_3]]{{\[}}%[[VAL_6]]] : memref<1x2x3x4xf32, 1>, memref<1x2x3x4xf32, 3>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_3]]{{\[}}%[[VAL_6]]], %[[VAL_5]] : memref<1xi32>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<1x2x3x4xf32, 3>
// CHECK:           %[[VAL_9:.*]] = arith.constant 24 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           memref.dma_start %[[VAL_1]]{{\[}}%[[VAL_10]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]]], %[[VAL_8]]{{\[}}%[[VAL_10]], %[[VAL_10]], %[[VAL_10]], %[[VAL_10]]], %[[VAL_9]], %[[VAL_7]]{{\[}}%[[VAL_10]]] : memref<1x2x3x4xf32, 1>, memref<1x2x3x4xf32, 3>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_7]]{{\[}}%[[VAL_10]]], %[[VAL_9]] : memref<1xi32>
// CHECK:           linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%[[VAL_4]], %[[VAL_8]] : memref<1x2x3x4xf32, 3>, memref<1x2x3x4xf32, 3>) outs(%[[VAL_2]] : memref<1x2x3x4xf32, 3>) {
// CHECK:           ^bb0(%[[VAL_11:.*]]: f32, %[[VAL_12:.*]]: f32, %[[VAL_13:.*]]: f32):
// CHECK:             %[[VAL_14:.*]] = arith.addf %[[VAL_11]], %[[VAL_12]] : f32
// CHECK:             linalg.yield %[[VAL_14]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_2]] : memref<1x2x3x4xf32, 3>
// CHECK:         }
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @linalg_generic_dma_ignore_outputs_addr_space(%arg0: memref<1x2x3x4xf32, 1>, %arg1: memref<1x2x3x4xf32, 1>) -> memref<1x2x3x4xf32, 3> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x3x4xf32, 3>
  linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%arg0, %arg1 : memref<1x2x3x4xf32, 1>, memref<1x2x3x4xf32, 1>) outs(%alloc : memref<1x2x3x4xf32, 3>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %1 = arith.addf %in, %in_1 : f32
    linalg.yield %1 : f32
  }
  return %alloc : memref<1x2x3x4xf32, 3>
}

// -----

// CHECK-LABEL:   func.func @linalg_generic_dma_ignore_all(
// CHECK-SAME:                                             %[[VAL_0:.*]]: memref<1x2x3x4xf32>,
// CHECK-SAME:                                             %[[VAL_1:.*]]: memref<1x2x3x4xf32>) -> memref<1x2x3x4xf32, 3> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<1x2x3x4xf32, 3>
// CHECK:           linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%[[VAL_0]], %[[VAL_1]] : memref<1x2x3x4xf32>, memref<1x2x3x4xf32>) outs(%[[VAL_2]] : memref<1x2x3x4xf32, 3>) {
// CHECK:           ^bb0(%[[VAL_3:.*]]: f32, %[[VAL_4:.*]]: f32, %[[VAL_5:.*]]: f32):
// CHECK:             %[[VAL_6:.*]] = arith.addf %[[VAL_3]], %[[VAL_4]] : f32
// CHECK:             linalg.yield %[[VAL_6]] : f32
// CHECK:           }
// CHECK:           return %[[VAL_2]] : memref<1x2x3x4xf32, 3>
// CHECK:         }
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @linalg_generic_dma_ignore_all(%arg0: memref<1x2x3x4xf32>, %arg1: memref<1x2x3x4xf32>) -> memref<1x2x3x4xf32, 3> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x2x3x4xf32, 3>
  linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%arg0, %arg1 : memref<1x2x3x4xf32>, memref<1x2x3x4xf32>) outs(%alloc : memref<1x2x3x4xf32, 3>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %1 = arith.addf %in, %in_1 : f32
    linalg.yield %1 : f32
  }
  return %alloc : memref<1x2x3x4xf32, 3>
}

// -----

// CHECK-LABEL:   func.func @linalg_generic_dma_only_one_input(
// CHECK-SAME:                                                 %[[VAL_0:.*]]: memref<3x4xi32, 1>,
// CHECK-SAME:                                                 %[[VAL_1:.*]]: memref<3x4xi32>) -> memref<3x4xi32, 3> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3x4xi32, 3>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<3x4xi32, 3>
// CHECK:           %[[VAL_5:.*]] = arith.constant 12 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           memref.dma_start %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_6]]], %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_6]]], %[[VAL_5]], %[[VAL_3]]{{\[}}%[[VAL_6]]] : memref<3x4xi32, 1>, memref<3x4xi32, 3>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_3]]{{\[}}%[[VAL_6]]], %[[VAL_5]] : memref<1xi32>
// CHECK:           linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%[[VAL_4]], %[[VAL_1]] : memref<3x4xi32, 3>, memref<3x4xi32>) outs(%[[VAL_2]] : memref<3x4xi32, 3>) {
// CHECK:           ^bb0(%[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32):
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:             linalg.yield %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_2]] : memref<3x4xi32, 3>
// CHECK:         }
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @linalg_generic_dma_only_one_input(%arg0: memref<3x4xi32, 1>, %arg1: memref<3x4xi32>) -> memref<3x4xi32, 3> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x4xi32, 3>
  linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%arg0, %arg1 : memref<3x4xi32, 1>, memref<3x4xi32>) outs(%alloc : memref<3x4xi32, 3>) {
  ^bb0(%in: i32, %in_1: i32, %out: i32):
    %1 = arith.addi %in, %in_1 : i32
    linalg.yield %1 : i32
  }
  return %alloc : memref<3x4xi32, 3>
}

// -----

// CHECK-LABEL:   func.func @linalg_generic_ignore_dyanmic_shape(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: memref<3x4xi32, 1>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: memref<?x?xi32, 1>) -> memref<3x4xi32, 1> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() {alignment = 64 : i64} : memref<3x4xi32, 1>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<3x4xi32, 3>
// CHECK:           %[[VAL_5:.*]] = arith.constant 12 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           memref.dma_start %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_6]]], %[[VAL_4]]{{\[}}%[[VAL_6]], %[[VAL_6]]], %[[VAL_5]], %[[VAL_3]]{{\[}}%[[VAL_6]]] : memref<3x4xi32, 1>, memref<3x4xi32, 3>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_3]]{{\[}}%[[VAL_6]]], %[[VAL_5]] : memref<1xi32>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<1xi32>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<3x4xi32, 3>
// CHECK:           linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%[[VAL_4]], %[[VAL_1]] : memref<3x4xi32, 3>, memref<?x?xi32, 1>) outs(%[[VAL_8]] : memref<3x4xi32, 3>) {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32):
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:             linalg.yield %[[VAL_12]] : i32
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = arith.constant 12 : index
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK:           memref.dma_start %[[VAL_8]]{{\[}}%[[VAL_14]], %[[VAL_14]]], %[[VAL_2]]{{\[}}%[[VAL_14]], %[[VAL_14]]], %[[VAL_13]], %[[VAL_7]]{{\[}}%[[VAL_14]]] : memref<3x4xi32, 3>, memref<3x4xi32, 1>, memref<1xi32>
// CHECK:           memref.dma_wait %[[VAL_7]]{{\[}}%[[VAL_14]]], %[[VAL_13]] : memref<1xi32>
// CHECK:           return %[[VAL_2]] : memref<3x4xi32, 1>
// CHECK:         }
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @linalg_generic_ignore_dyanmic_shape(%arg0: memref<3x4xi32, 1>, %arg1: memref<?x?xi32, 1>) -> memref<3x4xi32, 1> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<3x4xi32, 1>
  linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%arg0, %arg1 : memref<3x4xi32, 1>, memref<?x?xi32, 1>) outs(%alloc : memref<3x4xi32, 1>) {
  ^bb0(%in: i32, %in_1: i32, %out: i32):
    %1 = arith.addi %in, %in_1 : i32
    linalg.yield %1 : i32
  }
  return %alloc : memref<3x4xi32, 1>
}
