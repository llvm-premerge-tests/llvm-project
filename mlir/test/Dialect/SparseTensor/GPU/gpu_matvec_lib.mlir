// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:             --sparsification="enable-gpu-libgen" | FileCheck %s

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>

module {

// CHECK-LABEL:   func.func @matvec(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xf64, #sparse_tensor.encoding<{{{.*}}}>>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<?xf64>,
// CHECK-SAME:      %[[VAL_2:.*]]: tensor<?xf64>) -> tensor<?xf64> {
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = sparse_tensor.number_of_entries %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>>
// CHECK:           %[[VAL_6:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>>
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>>
// CHECK:           %[[VAL_8:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 0 : index} : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex, strided<[?], offset: ?>>
// CHECK:           %[[VAL_9:.*]] = sparse_tensor.coordinates %[[VAL_0]] {level = 1 : index} : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xindex, strided<[?], offset: ?>>
// CHECK:           %[[VAL_10:.*]] = sparse_tensor.values %[[VAL_0]] : tensor<?x?xf64, #sparse_tensor.encoding<{ lvlTypes = [ "compressed-nu", "singleton" ] }>> to memref<?xf64>
// CHECK:           %[[VAL_11:.*]] = bufferization.to_memref %[[VAL_1]] : memref<?xf64>
// CHECK:           %[[VAL_12:.*]] = bufferization.to_memref %[[VAL_2]] : memref<?xf64>
// CHECK:           %[[VAL_13:.*]] = gpu.wait async
// CHECK:           %[[VAL_14:.*]] = memref.dim %[[VAL_8]], %[[VAL_3]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:           %[[VAL_15:.*]], %[[VAL_16:.*]] = gpu.alloc async {{\[}}%[[VAL_13]]] (%[[VAL_14]]) : memref<?xindex>
// CHECK:           %[[VAL_17:.*]] = gpu.memcpy async {{\[}}%[[VAL_16]]] %[[VAL_15]], %[[VAL_8]] : memref<?xindex>, memref<?xindex, strided<[?], offset: ?>>
// CHECK:           %[[VAL_18:.*]] = gpu.wait async
// CHECK:           %[[VAL_19:.*]] = memref.dim %[[VAL_9]], %[[VAL_3]] : memref<?xindex, strided<[?], offset: ?>>
// CHECK:           %[[VAL_20:.*]], %[[VAL_21:.*]] = gpu.alloc async {{\[}}%[[VAL_18]]] (%[[VAL_19]]) : memref<?xindex>
// CHECK:           %[[VAL_22:.*]] = gpu.memcpy async {{\[}}%[[VAL_21]]] %[[VAL_20]], %[[VAL_9]] : memref<?xindex>, memref<?xindex, strided<[?], offset: ?>>
// CHECK:           %[[VAL_23:.*]] = gpu.wait async
// CHECK:           %[[VAL_24:.*]] = memref.dim %[[VAL_10]], %[[VAL_3]] : memref<?xf64>
// CHECK:           %[[VAL_25:.*]], %[[VAL_26:.*]] = gpu.alloc async {{\[}}%[[VAL_23]]] (%[[VAL_24]]) : memref<?xf64>
// CHECK:           %[[VAL_27:.*]] = gpu.memcpy async {{\[}}%[[VAL_26]]] %[[VAL_25]], %[[VAL_10]] : memref<?xf64>, memref<?xf64>
// CHECK:           %[[VAL_28:.*]] = gpu.wait async
// CHECK:           %[[VAL_29:.*]] = memref.dim %[[VAL_11]], %[[VAL_3]] : memref<?xf64>
// CHECK:           %[[VAL_30:.*]], %[[VAL_31:.*]] = gpu.alloc async {{\[}}%[[VAL_28]]] (%[[VAL_29]]) : memref<?xf64>
// CHECK:           %[[VAL_32:.*]] = gpu.memcpy async {{\[}}%[[VAL_31]]] %[[VAL_30]], %[[VAL_11]] : memref<?xf64>, memref<?xf64>
// CHECK:           %[[VAL_33:.*]] = gpu.wait async
// CHECK:           %[[VAL_34:.*]] = memref.dim %[[VAL_12]], %[[VAL_3]] : memref<?xf64>
// CHECK:           %[[VAL_35:.*]], %[[VAL_36:.*]] = gpu.alloc async {{\[}}%[[VAL_33]]] (%[[VAL_34]]) : memref<?xf64>
// CHECK:           %[[VAL_37:.*]] = gpu.memcpy async {{\[}}%[[VAL_36]]] %[[VAL_35]], %[[VAL_12]] : memref<?xf64>, memref<?xf64>
// CHECK:           gpu.wait {{\[}}%[[VAL_17]], %[[VAL_22]], %[[VAL_27]], %[[VAL_32]], %[[VAL_37]]]
// CHECK:           %[[VAL_38:.*]] = gpu.wait async
// CHECK:           %[[VAL_39:.*]], %[[VAL_40:.*]] = gpu.create_coo async {{\[}}%[[VAL_38]]] %[[VAL_6]], %[[VAL_7]], %[[VAL_5]], %[[VAL_15]], %[[VAL_20]], %[[VAL_25]] : memref<?xindex>, memref<?xindex>, memref<?xf64>
// CHECK:           %[[VAL_41:.*]], %[[VAL_42:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_40]]] %[[VAL_30]], %[[VAL_7]] : index into memref<?xf64>
// CHECK:           %[[VAL_43:.*]], %[[VAL_44:.*]] = gpu.create_dn_tensor async {{\[}}%[[VAL_42]]] %[[VAL_35]], %[[VAL_6]] : index into memref<?xf64>
// CHECK:           %[[VAL_45:.*]], %[[VAL_46:.*]] = gpu.spmv_buffer_size async {{\[}}%[[VAL_44]]] %[[VAL_39]], %[[VAL_41]], %[[VAL_43]] into f64
// CHECK:           %[[VAL_47:.*]], %[[VAL_48:.*]] = gpu.alloc async {{\[}}%[[VAL_46]]] (%[[VAL_45]]) : memref<?xi8>
// CHECK:           %[[VAL_49:.*]] = gpu.spmv async {{\[}}%[[VAL_48]]] %[[VAL_39]], %[[VAL_41]], %[[VAL_43]], %[[VAL_47]] : memref<?xi8> into f64
// CHECK:           %[[VAL_50:.*]] = gpu.destroy_sp_mat async {{\[}}%[[VAL_49]]] %[[VAL_39]]
// CHECK:           %[[VAL_51:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_50]]] %[[VAL_41]]
// CHECK:           %[[VAL_52:.*]] = gpu.destroy_dn_tensor async {{\[}}%[[VAL_51]]] %[[VAL_43]]
// CHECK:           %[[VAL_53:.*]] = gpu.dealloc async {{\[}}%[[VAL_52]]] %[[VAL_15]] : memref<?xindex>
// CHECK:           %[[VAL_54:.*]] = gpu.dealloc async {{\[}}%[[VAL_53]]] %[[VAL_20]] : memref<?xindex>
// CHECK:           %[[VAL_55:.*]] = gpu.dealloc async {{\[}}%[[VAL_54]]] %[[VAL_25]] : memref<?xf64>
// CHECK:           %[[VAL_56:.*]] = gpu.dealloc async {{\[}}%[[VAL_55]]] %[[VAL_47]] : memref<?xi8>
// CHECK:           %[[VAL_57:.*]] = gpu.dealloc async {{\[}}%[[VAL_56]]] %[[VAL_30]] : memref<?xf64>
// CHECK:           %[[VAL_58:.*]] = gpu.memcpy async {{\[}}%[[VAL_57]]] %[[VAL_12]], %[[VAL_35]] : memref<?xf64>, memref<?xf64>
// CHECK:           %[[VAL_59:.*]] = gpu.dealloc async {{\[}}%[[VAL_58]]] %[[VAL_35]] : memref<?xf64>
// CHECK:           gpu.wait {{\[}}%[[VAL_59]]]
// CHECK:           %[[VAL_60:.*]] = bufferization.to_tensor %[[VAL_12]] : memref<?xf64>
// CHECK:           return %[[VAL_60]] : tensor<?xf64>
// CHECK:         }
func.func @matvec(%A: tensor<?x?xf64, #SortedCOO>,
                  %x: tensor<?xf64>,
                  %y_in: tensor<?xf64>) -> tensor<?xf64> {
  %y_out = linalg.matvec
    ins(%A, %x: tensor<?x?xf64, #SortedCOO>, tensor<?xf64>)
    outs(%y_in: tensor<?xf64>) -> tensor<?xf64>
  return %y_out : tensor<?xf64>
}

}
