// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:             --sparsification="enable-gpu-libgen" | FileCheck %s

#CSR = #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>

//
// Compute matrix matrix C = AB
//
// CHECK-LABEL:   func.func @matmul(
// CHECK-SAME:                      %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{{{.*}}}>>,
// CHECK-SAME:                      %[[VAL_1:.*]]: tensor<?x?xf64>,
// CHECK-SAME:                      %[[VAL_2:.*]]: tensor<?x?xf64>) -> tensor<?x?xf64> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.number_of_entries %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>
// CHECK:           %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>>
// CHECK:           %[[VAL_8:.*]] = tensor.dim %[[VAL_1]], %[[VAL_4]] : tensor<?x?xf64>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.positions %[[VAL_0]] {level = 1 : index} : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xindex>
// CHECK:           %[[VAL_11:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>> to memref<?xf64>
// CHECK:           %[[VAL_12:.*]] = bufferization.to_memref %[[VAL_1]] : memref<?x?xf64>
// CHECK:           %[[VAL_13:.*]] = bufferization.to_memref %[[VAL_2]] : memref<?x?xf64>
// CHECK:           %[[VAL_14:.*]] = gpu.wait async
// CHECK:           %[[VAL_15:.*]] = memref.dim %[[VAL_9]], %[[VAL_3]] : memref<?xindex>
// CHECK:           %[[VAL_16:.*]], %[[VAL_17:.*]] = gpu.alloc async {{\[}}%[[VAL_14]]] (%[[VAL_15]]) : memref<?xindex>
// CHECK:           %[[VAL_18:.*]] = gpu.memcpy async {{\[}}%[[VAL_17]]] %[[VAL_16]], %[[VAL_9]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_19:.*]] = gpu.wait async
// CHECK:           %[[VAL_20:.*]] = memref.dim %[[VAL_10]], %[[VAL_3]] : memref<?xindex>
// CHECK:           %[[VAL_21:.*]], %[[VAL_22:.*]] = gpu.alloc async {{\[}}%[[VAL_19]]] (%[[VAL_20]]) : memref<?xindex>
// CHECK:           %[[VAL_23:.*]] = gpu.memcpy async {{\[}}%[[VAL_22]]] %[[VAL_21]], %[[VAL_10]] : memref<?xindex>, memref<?xindex>
// CHECK:           %[[VAL_24:.*]] = gpu.wait async
// CHECK:           %[[VAL_25:.*]] = memref.dim %[[VAL_11]], %[[VAL_3]] : memref<?xf64>
// CHECK:           %[[VAL_26:.*]], %[[VAL_27:.*]] = gpu.alloc async {{\[}}%[[VAL_24]]] (%[[VAL_25]]) : memref<?xf64>
// CHECK:           %[[VAL_28:.*]] = gpu.memcpy async {{\[}}%[[VAL_27]]] %[[VAL_26]], %[[VAL_11]] : memref<?xf64>, memref<?xf64>
// CHECK:           %[[VAL_29:.*]] = gpu.wait async
// CHECK:           %[[VAL_30:.*]] = memref.dim %[[VAL_12]], %[[VAL_3]] : memref<?x?xf64>
// CHECK:           %[[VAL_31:.*]] = memref.dim %[[VAL_12]], %[[VAL_4]] : memref<?x?xf64>
// CHECK:           %[[VAL_32:.*]], %[[VAL_33:.*]] = gpu.alloc async {{\[}}%[[VAL_29]]] (%[[VAL_30]], %[[VAL_31]]) : memref<?x?xf64>
// CHECK:           %[[VAL_34:.*]] = gpu.memcpy async {{\[}}%[[VAL_33]]] %[[VAL_32]], %[[VAL_12]] : memref<?x?xf64>, memref<?x?xf64>
// CHECK:           %[[VAL_35:.*]] = gpu.wait async
// CHECK:           %[[VAL_36:.*]] = memref.dim %[[VAL_13]], %[[VAL_3]] : memref<?x?xf64>
// CHECK:           %[[VAL_37:.*]] = memref.dim %[[VAL_13]], %[[VAL_4]] : memref<?x?xf64>
// CHECK:           %[[VAL_38:.*]], %[[VAL_39:.*]] = gpu.alloc async {{\[}}%[[VAL_35]]] (%[[VAL_36]], %[[VAL_37]]) : memref<?x?xf64>
// CHECK:           %[[VAL_40:.*]] = gpu.memcpy async {{\[}}%[[VAL_39]]] %[[VAL_38]], %[[VAL_13]] : memref<?x?xf64>, memref<?x?xf64>
// CHECK:           gpu.wait {{\[}}%[[VAL_18]], %[[VAL_23]], %[[VAL_28]], %[[VAL_34]], %[[VAL_40]]]
// CHECK:           %[[VAL_41:.*]] = gpu.wait async
// CHECK:           %[[VAL_42:.*]], %[[VAL_43:.*]] = gpu.create_csr async {{\[}}%[[VAL_41]]] %[[VAL_6]], %[[VAL_7]], %[[VAL_5]], %[[VAL_16]], %[[VAL_21]], %[[VAL_26]] : memref<?xindex>, memref<?xindex>, memref<?xf64>
// CHECK:           %[[VAL_44:.*]], %[[VAL_45:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_43]]] %[[VAL_32]], %[[VAL_7]], %[[VAL_8]] : index, index into memref<?x?xf64>
// CHECK:           %[[VAL_46:.*]], %[[VAL_47:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_45]]] %[[VAL_38]], %[[VAL_6]], %[[VAL_8]] : index, index into memref<?x?xf64>
// CHECK:           %[[VAL_48:.*]], %[[VAL_49:.*]] = gpu.spmm_buffer_size async {{\[}}%[[VAL_47]]] %[[VAL_42]], %[[VAL_44]], %[[VAL_46]] : index into f64
// CHECK:           %[[VAL_50:.*]], %[[VAL_51:.*]] = gpu.alloc async {{\[}}%[[VAL_49]]] (%[[VAL_48]]) : memref<?xi8>
// CHECK:           %[[VAL_52:.*]] = gpu.spmm async {{\[}}%[[VAL_51]]] %[[VAL_42]], %[[VAL_44]], %[[VAL_46]], %[[VAL_50]] : memref<?xi8> into f64
// CHECK:           %[[VAL_53:.*]] = gpu.destroy_sp_mat async {{\[}}%[[VAL_52]]] %[[VAL_42]]
// CHECK:           %[[VAL_54:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_53]]] %[[VAL_44]]
// CHECK:           %[[VAL_55:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_54]]] %[[VAL_46]]
// CHECK:           %[[VAL_56:.*]] = gpu.dealloc async {{\[}}%[[VAL_55]]] %[[VAL_16]] : memref<?xindex>
// CHECK:           %[[VAL_57:.*]] = gpu.dealloc async {{\[}}%[[VAL_56]]] %[[VAL_21]] : memref<?xindex>
// CHECK:           %[[VAL_58:.*]] = gpu.dealloc async {{\[}}%[[VAL_57]]] %[[VAL_26]] : memref<?xf64>
// CHECK:           %[[VAL_59:.*]] = gpu.dealloc async {{\[}}%[[VAL_58]]] %[[VAL_50]] : memref<?xi8>
// CHECK:           %[[VAL_60:.*]] = gpu.dealloc async {{\[}}%[[VAL_59]]] %[[VAL_32]] : memref<?x?xf64>
// CHECK:           %[[VAL_61:.*]] = gpu.memcpy async {{\[}}%[[VAL_60]]] %[[VAL_13]], %[[VAL_38]] : memref<?x?xf64>, memref<?x?xf64>
// CHECK:           %[[VAL_62:.*]] = gpu.dealloc async {{\[}}%[[VAL_61]]] %[[VAL_38]] : memref<?x?xf64>
// CHECK:           gpu.wait {{\[}}%[[VAL_62]]]
// CHECK:           %[[VAL_63:.*]] = bufferization.to_tensor %[[VAL_13]] : memref<?x?xf64>
// CHECK:           return %[[VAL_63]] : tensor<?x?xf64>
// CHECK:         }
func.func @matmul(%A: tensor<?x?xf64, #CSR>, %B: tensor<?x?xf64>, %C_in: tensor<?x?xf64>) -> tensor<?x?xf64> {
  %C_out = linalg.matmul
      ins(%A, %B: tensor<?x?xf64, #CSR>, tensor<?x?xf64>)
      outs(%C_in: tensor<?x?xf64>) -> tensor<?x?xf64>
  return %C_out : tensor<?x?xf64>
}
