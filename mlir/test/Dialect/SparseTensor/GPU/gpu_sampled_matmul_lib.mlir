// RUN: mlir-opt %s --sparsification="enable-gpu-libgen" | FileCheck %s

#trait_sampled_dense_dense = {
  indexing_maps = [
  affine_map<(i,j,k) -> (i,k)>,  // A
  affine_map<(i,j,k) -> (k,j)>,  // B
  affine_map<(i,j,k) -> (i,j)>   // S (out)
  ],
  iterator_types = ["parallel", "parallel", "reduction"],
  doc = "X(i,j) += S(i,j) SUM_k A(i,k) B(k,j)"
}

#trait_vec_op = {
  indexing_maps = [
  affine_map<(i,j) -> (i,j)>,  // a (in)
  affine_map<(i,j) -> (i,j)>,  // b (in)
  affine_map<(i,j) -> (i,j)>   // x (out)
  ],
  iterator_types = ["parallel", "parallel"]
}

#CSR = #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>

// CHECK-LABEL:   func.func @sparse_sampled_dd(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<8x8xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: tensor<8x8xf64>,
// CHECK-SAME:                                 %[[VAL_2:.*]]: tensor<8x8xf64>) -> tensor<8x8xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.number_of_entries %[[VAL_0]] : tensor<8x8xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>
// CHECK:           %[[VAL_6:.*]] = bufferization.to_memref %[[VAL_1]] : memref<8x8xf64>
// CHECK:           %[[VAL_7:.*]] = bufferization.to_memref %[[VAL_2]] : memref<8x8xf64>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<8x8xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<8x8xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<8x8xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xf64>
// CHECK:           %[[VAL_11:.*]] = gpu.wait async
// CHECK:           %[[VAL_12:.*]], %[[VAL_13:.*]] = gpu.alloc async {{\[}}%[[VAL_11]]] () : memref<8x8xf64>
// CHECK:           %[[VAL_14:.*]] = gpu.memcpy async {{\[}}%[[VAL_13]]] %[[VAL_12]], %[[VAL_6]] : memref<8x8xf64>, memref<8x8xf64>
// CHECK:           %[[VAL_15:.*]] = gpu.wait async
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = gpu.alloc async {{\[}}%[[VAL_15]]] () : memref<8x8xf64>
// CHECK:           %[[VAL_18:.*]] = gpu.memcpy async {{\[}}%[[VAL_17]]] %[[VAL_16]], %[[VAL_7]] : memref<8x8xf64>, memref<8x8xf64>
// CHECK:           %[[VAL_19:.*]] = gpu.wait async
// CHECK:           %[[VAL_20:.*]] = memref.dim %[[VAL_8]], %[[VAL_4]] : memref<?xindex>
// CHECK:           %[[VAL_21:.*]], %[[VAL_22:.*]] = gpu.alloc async {{\[}}%[[VAL_19]]] (%[[VAL_20]]) : memref<?xindex>
// CHECK:           %[[VAL_23:.*]] = gpu.memcpy async {{\[}}%[[VAL_22]]] %[[VAL_21]], %[[VAL_8]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_24:.*]] = gpu.wait async
// CHECK:           %[[VAL_25:.*]] = memref.dim %[[VAL_9]], %[[VAL_4]] : memref<?xindex>
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = gpu.alloc async {{\[}}%[[VAL_24]]] (%[[VAL_25]]) : memref<?xindex>
// CHECK:           %[[VAL_28:.*]] = gpu.memcpy async {{\[}}%[[VAL_27]]] %[[VAL_26]], %[[VAL_9]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_29:.*]] = gpu.wait async
// CHECK:           %[[VAL_30:.*]] = memref.dim %[[VAL_10]], %[[VAL_4]] : memref<?xf64>
// CHECK:           %[[VAL_31:.*]], %[[VAL_32:.*]] = gpu.alloc async {{\[}}%[[VAL_29]]] (%[[VAL_30]]) : memref<?xf64>
// CHECK:           %[[VAL_33:.*]] = gpu.memcpy async {{\[}}%[[VAL_32]]] %[[VAL_31]], %[[VAL_10]] : memref<?xf64>, memref<?xf64>
// CHECK:           gpu.wait {{\[}}%[[VAL_14]], %[[VAL_18]], %[[VAL_23]], %[[VAL_28]], %[[VAL_33]]]
// CHECK:           %[[VAL_34:.*]] = gpu.wait async
// CHECK:           %[[VAL_35:.*]], %[[VAL_36:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_34]]] %[[VAL_12]], %[[VAL_3]], %[[VAL_3]] : index, index into memref<8x8xf64>
// CHECK:           %[[VAL_37:.*]], %[[VAL_38:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_36]]] %[[VAL_16]], %[[VAL_3]], %[[VAL_3]] : index, index into memref<8x8xf64>
// CHECK:           %[[VAL_39:.*]], %[[VAL_40:.*]] = gpu.create_csr async {{\[}}%[[VAL_38]]] %[[VAL_3]], %[[VAL_3]], %[[VAL_5]], %[[VAL_21]], %[[VAL_26]], %[[VAL_31]] : memref<?xindex>, memref<?xindex>, memref<?xf64>
// CHECK:           %[[VAL_41:.*]], %[[VAL_42:.*]] = gpu.sddmm_buffer_size async {{\[}}%[[VAL_40]]] %[[VAL_35]], %[[VAL_37]], %[[VAL_39]] into f64
// CHECK:           %[[VAL_43:.*]], %[[VAL_44:.*]] = gpu.alloc async {{\[}}%[[VAL_42]]] (%[[VAL_41]]) : memref<?xi8>
// CHECK:           %[[VAL_45:.*]] = gpu.sddmm async {{\[}}%[[VAL_44]]] %[[VAL_35]], %[[VAL_37]], %[[VAL_39]], %[[VAL_43]] : memref<?xi8> into f64
// CHECK:           %[[VAL_46:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_45]]] %[[VAL_35]]
// CHECK:           %[[VAL_47:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_46]]] %[[VAL_37]]
// CHECK:           %[[VAL_48:.*]] = gpu.destroy_sp_mat async {{\[}}%[[VAL_47]]] %[[VAL_39]]
// CHECK:           %[[VAL_49:.*]] = gpu.dealloc async {{\[}}%[[VAL_48]]] %[[VAL_43]] : memref<?xi8>
// CHECK:           %[[VAL_50:.*]] = gpu.dealloc async {{\[}}%[[VAL_49]]] %[[VAL_12]] : memref<8x8xf64>
// CHECK:           %[[VAL_51:.*]] = gpu.dealloc async {{\[}}%[[VAL_50]]] %[[VAL_16]] : memref<8x8xf64>
// CHECK:           %[[VAL_52:.*]] = gpu.dealloc async {{\[}}%[[VAL_51]]] %[[VAL_21]] : memref<?xindex>
// CHECK:           %[[VAL_53:.*]] = gpu.dealloc async {{\[}}%[[VAL_52]]] %[[VAL_26]] : memref<?xindex>
// CHECK:           %[[VAL_54:.*]] = gpu.memcpy async {{\[}}%[[VAL_53]]] %[[VAL_10]], %[[VAL_31]] : memref<?xf64>, memref<?xf64>
// CHECK:           %[[VAL_55:.*]] = gpu.dealloc async {{\[}}%[[VAL_54]]] %[[VAL_31]] : memref<?xf64>
// CHECK:           gpu.wait {{\[}}%[[VAL_55]]]
// CHECK:           %[[VAL_56:.*]] = sparse_tensor.load %[[VAL_0]] : tensor<8x8xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>
// CHECK:           return %[[VAL_56]] : tensor<8x8xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>
// CHECK:         }
//
// A kernel that computes a direct sampled matrix matrix multiplication
// (with sparse result).
// Compute SDDMM C = C\spy AB
//
func.func @sparse_sampled_dd(%argS: tensor<8x8xf64, #CSR>,
                               %argA: tensor<8x8xf64>,
                               %argB: tensor<8x8xf64>) -> tensor<8x8xf64, #CSR> {
    %result = linalg.generic #trait_sampled_dense_dense
      ins(%argA, %argB: tensor<8x8xf64>, tensor<8x8xf64>)
      outs(%argS: tensor<8x8xf64, #CSR>) {
        ^bb(%a: f64, %b: f64, %s: f64):
           %f0 = arith.constant 0.0 : f64
           %u = sparse_tensor.unary %s : f64 to f64
             present={
                ^bb0(%p: f64):
                  %mul = arith.mulf %a, %b : f64
                  sparse_tensor.yield %mul : f64
             }
             absent={}
           %r = sparse_tensor.reduce %s, %u, %f0 : f64 {
              ^bb0(%p: f64, %q: f64):
                %add = arith.addf %p, %q : f64
                sparse_tensor.yield %add : f64
            }
           linalg.yield %r : f64
    } -> tensor<8x8xf64, #CSR>
    return %result : tensor<8x8xf64, #CSR>
}
