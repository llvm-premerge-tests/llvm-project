// This file tests the conversion of GPU WMMA ops to ROCDL dialect.
// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx1100 index-bitwidth=32' -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-gpu-to-rocdl='chipset=gfx1100 index-bitwidth=32 opselect' -split-input-file | FileCheck %s --check-prefix=OPSEL

gpu.module @main {
  // CHECK-LABEL: load_a_op_16_16_16_no_transpose
  func.func @load_a_op_16_16_16_no_transpose()->(!gpu.mma_matrix<16x16xf16, "AOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[BASE:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:       %[[C32_0:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK-NEXT:  %[[TIDX:.*]] = rocdl.workitem.id.x : i32
    // CHECK-NEXT:  %[[TIDY:.*]] = rocdl.workitem.id.y : i32
    // CHECK-NEXT:  %[[TIDZ:.*]] = rocdl.workitem.id.z : i32
    // CHECK-NEXT:  %[[BDIMX:.*]] = rocdl.workgroup.dim.x : i64
    // CHECK-NEXT:  %[[BDIMX16:.*]] = llvm.trunc %[[BDIMX]] : i64 to i32
    // CHECK-NEXT:  %[[BDIMY:.*]] = rocdl.workgroup.dim.y : i64
    // CHECK-NEXT:  %[[BDIMY16:.*]] = llvm.trunc %[[BDIMY]] : i64 to i32
    // CHECK-NEXT:  %[[BDIMXY:.*]] = llvm.mul %[[BDIMX16]], %[[BDIMY16]]  : i32
    // CHECK-NEXT:  %[[TIDZL:.*]] = llvm.mul %[[BDIMXY]], %[[TIDZ:.*]]  : i32
    // CHECK-NEXT:  %[[TIDYL:.*]] = llvm.mul %[[TIDY]], %[[BDIMX16]]  : i32
    // CHECK-NEXT:  %[[TIDZY:.*]] = llvm.add %[[TIDZL]], %[[TIDYL]]  : i32
    // CHECK-NEXT:  %[[LTID:.*]] = llvm.add %[[TIDZY]], %[[TIDX]]  : i32
    // CHECK-NEXT:  %[[C32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK-NEXT:  %[[LTIDDIV32:.*]] = llvm.sdiv %[[LTID]], %[[C32]]  : i32
    // CHECK-NEXT:  %[[LOCALTIDBASE:.*]] = llvm.mul %[[LTIDDIV32]], %[[C32]]  : i32
    // CHECK-NEXT:  %[[WARPLOCALTID:.*]] = llvm.sub %[[LTID]], %[[LOCALTIDBASE]]  : i32
    // CHECK-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT:  %[[WRAPPEDTID:.*]] = llvm.srem %[[WARPLOCALTID]], %[[C16]]  : i32
    // The part checked up to this point will be common in most of the WMMA op
    // lowerings. Checking all of these lines will be skipped in the subsequent
    // tests as the same utility emits the IR up to this point. Only some
    // values which are used later will be matched.
    // CHECK-NEXT:  %[[LOADEDVALS:.*]] = llvm.mlir.undef : vector<16xf16>
    // CHECK-NEXT:  %[[WRAPPEDTID32:.*]] = llvm.mul %[[WRAPPEDTID]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:  %[[OFFSET0:.*]] = llvm.add %[[WRAPPEDTID32]], %[[C0]]  : i32
    // CHECK-NEXT:  %[[LOADADDR0:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET0]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED0:.*]] = llvm.load %[[LOADADDR0]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[LOADEDVALS0:.*]] = llvm.insertelement %[[LOADED0]], %[[LOADEDVALS]][%[[C0]] : i32] : vector<16xf16>
    // CHECK-NEXT:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:  %[[OFFSET1:.*]] = llvm.add %[[WRAPPEDTID32]], %[[C1]]  : i32
    // CHECK-NEXT:  %[[LOADADDR1:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED1:.*]] = llvm.load %[[LOADADDR1]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[LOADEDVALS1:.*]] = llvm.insertelement %[[LOADED1]], %[[LOADEDVALS0]][%[[C1]] : i32] : vector<16xf16>
    // We just check the loading and insertion of two values only, rest of the
    // values need not be checked as they are emitted in a loop just with
    // different parameters.
    // CHECK:       %[[C15:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK:       %[[OFFSET15:.*]] = llvm.add %[[WRAPPEDTID32]], %{{.*}} : i32
    // CHECK-NEXT:  %[[LOADADDR15:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET15]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADEDVALS15:.*]] = llvm.load %[[LOADADDR15]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[RES:.*]] = llvm.insertelement %[[LOADEDVALS15]], %{{.*}}[%[[C15]] : i32] : vector<16xf16>
    // CHECK-NEXT:  llvm.return %[[RES]] : vector<16xf16>
    return %0 : !gpu.mma_matrix<16x16xf16, "AOp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_a_op_16_16_16_transpose 
  func.func @load_a_op_16_16_16_transpose()->(!gpu.mma_matrix<16x16xf16, "AOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index, transpose} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "AOp">
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[BASE:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:       %[[C32_0:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[WRAPPEDTID:.*]] = llvm.srem %{{.*}}, {{.*}}  : i32
    // CHECK-NEXT:  %[[LOADEDVALS:.*]] = llvm.mlir.undef : vector<16xf16>
    // CHECK-NEXT:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:  %[[ROW0:.*]] = llvm.mul %[[C0]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET0:.*]] = llvm.add %[[ROW0]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS0:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET0]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED0:.*]] = llvm.load %[[ADDRESS0]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[LOADEDVALS0:.*]] = llvm.insertelement %[[LOADED0]], %[[LOADEDVALS]][%[[C0]] : i32] : vector<16xf16>
    // CHECK-NEXT:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:  %[[ROW1:.*]] = llvm.mul %[[C1]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET1:.*]] = llvm.add %[[ROW1]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS1:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED1:.*]] = llvm.load %[[ADDRESS1]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  llvm.insertelement %[[LOADED1]], %[[LOADEDVALS0]][%[[C1]] : i32] : vector<16xf16>
    // We just check the loading and insertion of two values only, rest of the
    // values need not be checked as they are emitted in a loop just with
    // different parameters.
    // CHECK:       %[[C15:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK-NEXT:  %[[ROW15:.*]] = llvm.mul %[[C15]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET15:.*]] = llvm.add %[[ROW15]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS15:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET15]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED15:.*]] = llvm.load %[[ADDRESS15]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[RES:.*]] = llvm.insertelement %[[LOADED15]], %{{.*}}[%[[C15]] : i32] : vector<16xf16>
    // CHECK-NEXT:  llvm.return %[[RES]] : vector<16xf16>
    return %0 : !gpu.mma_matrix<16x16xf16, "AOp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_b_op_16_16_16_no_transpose
  func.func @load_b_op_16_16_16_no_transpose()->(!gpu.mma_matrix<16x16xf16, "BOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[BASE:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:       %[[C32_0:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[WRAPPEDTID:.*]] = llvm.srem %{{.*}}, {{.*}}  : i32
    // CHECK:       %[[LOADEDVALS:.*]] = llvm.mlir.undef : vector<16xf16>
    // CHECK-NEXT:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:  %[[ROW0:.*]] = llvm.mul %[[C0]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET0:.*]] = llvm.add %[[ROW0]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS0:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET0]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED0:.*]] = llvm.load %[[ADDRESS0]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[LOADEDVALS0:.*]] = llvm.insertelement %[[LOADED0]], %[[LOADEDVALS]][%[[C0]] : i32] : vector<16xf16>
    // CHECK-NEXT:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:  %[[ROW1:.*]] = llvm.mul %[[C1]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET1:.*]] = llvm.add %[[ROW1]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS1:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED1:.*]] = llvm.load %[[ADDRESS1]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  llvm.insertelement %[[LOADED1]], %[[LOADEDVALS0]][%[[C1]] : i32] : vector<16xf16>
    // We just check the loading and insertion of two values only, rest of the
    // values need not be checked as they are emitted in a loop just with
    // different parameters.
    // CHECK:       %[[C15:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK-NEXT:  %[[ROW15:.*]] = llvm.mul %[[C15]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET15:.*]] = llvm.add %[[ROW15]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS15:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET15]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED15:.*]] = llvm.load %[[ADDRESS15]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[RES:.*]] = llvm.insertelement %[[LOADED15]], %{{.*}}[%[[C15]] : i32] : vector<16xf16>
    // CHECK-NEXT:  llvm.return %[[RES]] : vector<16xf16>
    return %0 : !gpu.mma_matrix<16x16xf16, "BOp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_b_op_16_16_16_transpose
  func.func @load_b_op_16_16_16_transpose()->(!gpu.mma_matrix<16x16xf16, "BOp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index, transpose} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "BOp">
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[BASE:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:       %[[C32_0:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[WRAPPEDTID:.*]] = llvm.srem %{{.*}}, %{{.*}}  : i32
    // CHECK-NEXT:  %[[LOADEDVALS:.*]] = llvm.mlir.undef : vector<16xf16>
    // CHECK-NEXT:  %[[WRAPPEDTID32:.*]] = llvm.mul %[[WRAPPEDTID]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:  %[[OFFSET0:.*]] = llvm.add %[[WRAPPEDTID32]], %[[C0]]  : i32
    // CHECK-NEXT:  %[[LOADADDR0:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET0]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED0:.*]] = llvm.load %[[LOADADDR0]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[LOADEDVALS0:.*]] = llvm.insertelement %[[LOADED0]], %[[LOADEDVALS]][%[[C0]] : i32] : vector<16xf16>
    // CHECK-NEXT:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:  %[[OFFSET1:.*]] = llvm.add %[[WRAPPEDTID32]], %[[C1]]  : i32
    // CHECK-NEXT:  %[[LOADADDR1:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED1:.*]] = llvm.load %[[LOADADDR1]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[LOADEDVALS1:.*]] = llvm.insertelement %[[LOADED1]], %[[LOADEDVALS0]][%[[C1]] : i32] : vector<16xf16>
    // We just check the loading and insertion of two values only, rest of the
    // values need not be checked as they are emitted in a loop just with
    // different parameters.
    // CHECK:       %[[C15:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK:       %[[OFFSET15:.*]] = llvm.add %[[WRAPPEDTID32]], %{{.*}} : i32
    // CHECK-NEXT:  %[[LOADADDR15:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET15]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADEDVALS15:.*]] = llvm.load %[[LOADADDR15]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[RES:.*]] = llvm.insertelement %[[LOADEDVALS15]], %{{.*}}[%[[C15]] : i32] : vector<16xf16>
    // CHECK-NEXT:  llvm.return %[[RES]] : vector<16xf16>
    return %0 : !gpu.mma_matrix<16x16xf16, "BOp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_c_op_16_16_16_no_opselect
  func.func @load_c_op_16_16_16_no_opselect()->(!gpu.mma_matrix<16x16xf32, "COp">) {
    %wg_1 = memref.alloca() {alignment = 32} : memref<32x32xf32, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg_1[%i, %j] {leadDimension = 32 : index} : memref<32x32xf32, 3> -> !gpu.mma_matrix<16x16xf32, "COp">
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[BASE:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    // CHECK:       %[[C32_0:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[WARPLOCALTID:.*]] = llvm.sub %{{.*}}, %{{.*}}  : i32
    // CHECK-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT:  %[[WRAPPEDTID:.*]] = llvm.srem %[[WARPLOCALTID]], %[[C16]]  : i32
    // CHECK-NEXT:  %[[LOADEDVALS:.*]] = llvm.mlir.undef : vector<8xf32>
    // CHECK-NEXT:  %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT:  %[[WTIDDIV16:.*]] = llvm.sdiv %[[WARPLOCALTID]], %[[C16]]  : i32
    // CHECK-NEXT:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:  %[[ITER0:.*]] = llvm.mul %[[C0]], %[[C2]]  : i32
    // CHECK-NEXT:  %[[ROW0:.*]] = llvm.add %[[ITER0]], %[[WTIDDIV16]]  : i32
    // CHECK-NEXT:  %[[ROWLDM0:.*]] = llvm.mul %[[ROW0]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET0:.*]] = llvm.add %[[ROWLDM0]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS0:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET0]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    // CHECK-NEXT:  %[[LOADED0:.*]] = llvm.load %[[ADDRESS0]] : !llvm.ptr<3> -> f32
    // CHECK-NEXT:  %[[LOADEDVAL0:.*]] = llvm.insertelement %[[LOADED0]], %[[LOADEDVALS]][%[[C0]] : i32] : vector<8xf32>
    // CHECK-NEXT:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:  %[[ITER1:.*]] = llvm.mul %[[C1]], %[[C2]]  : i32
    // CHECK-NEXT:  %[[ROW1:.*]] = llvm.add %[[ITER1]], %[[WTIDDIV16]]  : i32
    // CHECK-NEXT:  %[[ROWLDM1:.*]] = llvm.mul %[[ROW1]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET1:.*]] = llvm.add %[[ROWLDM1]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS1:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    // CHECK-NEXT:  %[[LOADED1:.*]] = llvm.load %[[ADDRESS1]] : !llvm.ptr<3> -> f32
    // CHECK-NEXT:  %[[LOADEDVAL1:.*]] = llvm.insertelement %[[LOADED1]], %[[LOADEDVAL0]][%[[C1]] : i32] : vector<8xf32>
    // We just check the loading and insertion of two values only, rest of the
    // values need not be checked as they are emitted in a loop just with
    // different parameters.
    // CHECK:       %[[C7:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:       %[[RES:.*]] = llvm.insertelement %{{.*}}, %{{.*}}[%[[C7]] : i32] : vector<8xf32>
    // CHECK-NEXT:  llvm.return %[[RES]] : vector<8xf32>
    return %0 : !gpu.mma_matrix<16x16xf32, "COp">
  }
}

// -----

gpu.module @main {
  // CHECK-LABEL: load_c_op_16_16_16_no_opselect
  func.func @load_c_op_16_16_16_no_opselect()->(!gpu.mma_matrix<16x16xf16, "COp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "COp">
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[BASE:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:       %[[C32_0:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[WARPLOCALTID:.*]] = llvm.sub %{{.*}}, %{{.*}}  : i32
    // CHECK-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT:  %[[WRAPPEDTID:.*]] = llvm.srem %[[WARPLOCALTID]], %[[C16]]  : i32
    // CHECK-NEXT:  %[[LOADEDVALS:.*]] = llvm.mlir.undef : vector<16xf16>
    // CHECK-NEXT:  %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT:  %[[WTIDDIV16:.*]] = llvm.sdiv %[[WARPLOCALTID]], %[[C16]]  : i32
    // CHECK-NEXT:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:  %[[ITER0:.*]] = llvm.mul %[[C0]], %[[C2]]  : i32
    // CHECK-NEXT:  %[[ROW0:.*]] = llvm.add %[[ITER0]], %[[WTIDDIV16]]  : i32
    // CHECK-NEXT:  %[[ROWLDM0:.*]] = llvm.mul %[[ROW0]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET0:.*]] = llvm.add %[[ROWLDM0]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS0:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET0]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED0:.*]] = llvm.load %[[ADDRESS0]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[LOADEDVALS0:.*]] = llvm.insertelement %[[LOADED0]], %[[LOADEDVALS]][%[[ITER0]] : i32] : vector<16xf16>
    // CHECK-NEXT:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:  %[[ITER1:.*]] = llvm.mul %[[C1]], %[[C2]]  : i32
    // CHECK-NEXT:  %[[ROW1:.*]] = llvm.add %[[ITER1]], %[[WTIDDIV16]]  : i32
    // CHECK-NEXT:  %[[ROWLDM1:.*]] = llvm.mul %[[ROW1]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET1:.*]] = llvm.add %[[ROWLDM1]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS1:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[LOADED1:.*]] = llvm.load %[[ADDRESS1]] : !llvm.ptr<3> -> f16
    // CHECK-NEXT:  %[[LOADEDVALS1:.*]] = llvm.insertelement %[[LOADED1]], %[[LOADEDVALS0]][%[[ITER1]] : i32] : vector<16xf16>
    // CHECK:       %[[C15:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK:       %[[RES:.*]] = llvm.insertelement %{{.*}}, %{{.*}}[%{{.*}} : i32] : vector<16xf16>
    // CHECK-NEXT:  llvm.return %[[RES]] : vector<16xf16>
    return %0 : !gpu.mma_matrix<16x16xf16, "COp">
  }
}

// -----

gpu.module @main {
  // OPSEL-LABEL: load_c_op_16_16_16_opselect
  func.func @load_c_op_16_16_16_opselect()->(!gpu.mma_matrix<16x16xf16, "COp">) {
    %wg = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    %0 = gpu.subgroup_mma_load_matrix %wg[%i, %j] {leadDimension = 32 : index} : memref<32x32xf16, 3> -> !gpu.mma_matrix<16x16xf16, "COp">
    // OPSEL:       llvm.mlir.constant(32 : index) : i32
    // OPSEL:       %[[BASE:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // OPSEL:       %[[C32_0:.*]] = llvm.mlir.constant(32 : index) : i32
    // OPSEL:       %[[WARPLOCALTID:.*]] = llvm.sub %{{.*}}, %{{.*}}  : i32
    // OPSEL-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // OPSEL-NEXT:  %[[WRAPPEDTID:.*]] = llvm.srem %[[WARPLOCALTID]], %[[C16]]  : i32
    // OPSEL-NEXT:  %[[LOADEDVALS:.*]] = llvm.mlir.undef : vector<16xf16>
    // OPSEL-NEXT:  %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // OPSEL-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // OPSEL-NEXT:  %[[WTIDDIV16:.*]] = llvm.sdiv %[[WARPLOCALTID]], %[[C16]]  : i32
    // OPSEL-NEXT:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // OPSEL-NEXT:  %[[ITER0:.*]] = llvm.mul %[[C0]], %[[C2]]  : i32
    // OPSEL-NEXT:  %[[ROW0:.*]] = llvm.add %[[ITER0]], %[[WTIDDIV16]]  : i32
    // OPSEL-NEXT:  %[[ROWLDM0:.*]] = llvm.mul %[[ROW0]], %[[C32_0]]  : i32
    // OPSEL-NEXT:  %[[OFFSET0:.*]] = llvm.add %[[ROWLDM0]], %[[WRAPPEDTID]]  : i32
    // OPSEL-NEXT:  %[[ADDRESS0:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET0]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // OPSEL-NEXT:  %[[LOADED0:.*]] = llvm.load %[[ADDRESS0]] : !llvm.ptr<3> -> f16
    // OPSEL-NEXT:  %[[C1C:.*]] = llvm.mlir.constant(1 : i32) : i32
    // OPSEL-NEXT:  %[[VECOFFSET0:.*]] = llvm.add %[[ITER0]], %[[C1C]]  : i32
    // OPSEL-NEXT:  %[[LOADEDVALS0:.*]] = llvm.insertelement %[[LOADED0]], %[[LOADEDVALS]][%[[VECOFFSET0]] : i32] : vector<16xf16>
    // OPSEL-NEXT:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // OPSEL-NEXT:  %[[ITER1:.*]] = llvm.mul %[[C1]], %[[C2]]  : i32
    // OPSEL-NEXT:  %[[ROW1:.*]] = llvm.add %[[ITER1]], %[[WTIDDIV16]]  : i32
    // OPSEL-NEXT:  %[[ROWLDM1:.*]] = llvm.mul %[[ROW1]], %[[C32_0]]  : i32
    // OPSEL-NEXT:  %[[OFFSET1:.*]] = llvm.add %[[ROWLDM1]], %[[WRAPPEDTID]]  : i32
    // OPSEL-NEXT:  %[[ADDRESS1:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // OPSEL-NEXT:  %[[LOADED1:.*]] = llvm.load %[[ADDRESS1]] : !llvm.ptr<3> -> f16
    // OPSEL-NEXT:  %[[C1C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // OPSEL-NEXT:  %[[VECOFFSET1:.*]] = llvm.add %[[ITER1]], %[[C1C1]]  : i32
    // OPSEL-NEXT:  %[[LOADEDVALS1:.*]] = llvm.insertelement %[[LOADED1]], %[[LOADEDVALS0]][%[[VECOFFSET1]] : i32] : vector<16xf16>
    // OPSEL:       %[[C15:.*]] = llvm.mlir.constant(15 : i32) : i32
    // OPSEL:       %[[RES:.*]] = llvm.insertelement %{{.*}}, %{{.*}}[%{{.*}} : i32] : vector<16xf16>
    // OPSEL-NEXT:  llvm.return %[[RES]] : vector<16xf16>
    return %0 : !gpu.mma_matrix<16x16xf16, "COp">
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: compute_op_f32_f16
  // CHECK-SAME: (%[[AOP:.*]]: vector<16xf16>, %[[BOP:.*]]: vector<16xf16>, %[[COP:.*]]: vector<8xf32>)
  func.func @compute_op_f32_f16(%arg0: !gpu.mma_matrix<16x16xf16, "AOp">, %arg1: !gpu.mma_matrix<16x16xf16, "BOp">, %arg2: !gpu.mma_matrix<16x16xf32, "COp">) -> (!gpu.mma_matrix<16x16xf32, "COp">) {
    %0 = gpu.subgroup_mma_compute %arg0, %arg1, %arg2 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
    // CHECK:       %[[RES:.*]] = rocdl.wmma.f32.16x16x16.f16 %[[AOP]], %[[BOP]], %[[COP]] : (vector<16xf16>, vector<16xf16>, vector<8xf32>) -> vector<8xf32>
    // CHECK-NEXT:  llvm.return %[[RES]] : vector<8xf32>
    return %0 : !gpu.mma_matrix<16x16xf32, "COp">
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: compute_op_f16_f16
  // CHECK-SAME: (%[[AOP:.*]]: vector<16xf16>, %[[BOP:.*]]: vector<16xf16>, %[[COP:.*]]: vector<16xf16>)
  // OPSEL-LABEL: compute_op_f16_f16
  // OPSEL-SAME: (%[[AOP:.*]]: vector<16xf16>, %[[BOP:.*]]: vector<16xf16>, %[[COP:.*]]: vector<16xf16>)
  func.func @compute_op_f16_f16(%arg0: !gpu.mma_matrix<16x16xf16, "AOp">, %arg1: !gpu.mma_matrix<16x16xf16, "BOp">, %arg2: !gpu.mma_matrix<16x16xf16, "COp">) -> (!gpu.mma_matrix<16x16xf16, "COp">) {
    %0 = gpu.subgroup_mma_compute %arg0, %arg1, %arg2 : !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp"> -> !gpu.mma_matrix<16x16xf16, "COp">
    // CHECK:       %[[OPSELECT:.*]] = llvm.mlir.constant(false) : i1
    // CHECK-NEXT:  %[[RES:.*]] = rocdl.wmma.f16.16x16x16.f16 %[[AOP]], %[[BOP]], %[[COP]], %[[OPSELECT]] : (vector<16xf16>, vector<16xf16>, vector<16xf16>, i1) -> vector<16xf16>
    // CHECK-NEXT:  llvm.return %[[RES]] : vector<16xf16>
    // OPSEL:       %[[OPSELECT:.*]] = llvm.mlir.constant(true) : i1
    // OPSEL-NEXT:  %[[RES:.*]] = rocdl.wmma.f16.16x16x16.f16 %[[AOP]], %[[BOP]], %[[COP]], %[[OPSELECT]] : (vector<16xf16>, vector<16xf16>, vector<16xf16>, i1) -> vector<16xf16>
    // OPSEL-NEXT:  llvm.return %[[RES]] : vector<16xf16>
    return %0 : !gpu.mma_matrix<16x16xf16, "COp">
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: store_cop_f32
  // CHECK-SAME: (%[[SRC:.*]]: vector<8xf32>)
  func.func @store_cop_f32(%arg0: !gpu.mma_matrix<16x16xf32, "COp">) -> () {
    %wg_1 = memref.alloca() {alignment = 32} : memref<32x32xf32, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    gpu.subgroup_mma_store_matrix %arg0, %wg_1[%i, %j] {leadDimension = 32 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<32x32xf32, 3>
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[BASE:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    // CHECK:       %[[C32_0:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[WARPLOCALTID:.*]] = llvm.sub %{{.*}}, %{{.*}}  : i32
    // CHECK-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT:  %[[WRAPPEDTID:.*]] = llvm.srem %[[WARPLOCALTID]], %[[C16]]  : i32
    // CHECK-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT:  %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:  %[[WTIDDIV16:.*]] = llvm.sdiv %[[WARPLOCALTID]], %[[C16]]  : i32
    // CHECK-NEXT:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:  %[[ITER0:.*]] = llvm.mul %[[C0]], %[[C2]] : i32
    // CHECK-NEXT:  %[[ROW0:.*]] = llvm.add %[[WTIDDIV16]], %[[ITER0]]  : i32
    // CHECK-NEXT:  %[[ROWLDM0:.*]] = llvm.mul %[[ROW0]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET0:.*]] = llvm.add %[[ROWLDM0]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS0:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET0]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    // CHECK-NEXT:  %[[ELE0:.*]] = llvm.extractelement %[[SRC]][%[[C0]] : i32] : vector<8xf32>
    // CHECK-NEXT:  llvm.store %[[ELE0]], %[[ADDRESS0]] : f32, !llvm.ptr<3>
    // CHECK-NEXT:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:  %[[ITER1:.*]] = llvm.mul %[[C1]], %[[C2]] : i32
    // CHECK-NEXT:  %[[ROW1:.*]] = llvm.add %[[WTIDDIV16]], %[[ITER1]]  : i32
    // CHECK-NEXT:  %[[ROWLDM1:.*]] = llvm.mul %[[ROW1]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET1:.*]] = llvm.add %[[ROWLDM1]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS1:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f32
    // CHECK-NEXT:  %[[ELE1:.*]] = llvm.extractelement %[[SRC]][%[[C1]] : i32] : vector<8xf32>
    // CHECK-NEXT:  llvm.store %[[ELE1]], %[[ADDRESS1]] : f32, !llvm.ptr<3>
    // CHECK:       %[[C7:.*]] = llvm.mlir.constant(7 : i32) : i32
    // CHECK:       %[[ELE7:.*]] = llvm.extractelement %[[SRC]][%[[C7]] : i32] : vector<8xf32>
    // CHECK-NEXT:  llvm.store %[[ELE7]], %{{.*}} : f32, !llvm.ptr<3>
    return
  }
}

// -----

gpu.module @test_module {
  // CHECK-LABEL: store_cop_f16_no_opsel
  // CHECK-SAME: (%[[SRC:.*]]: vector<16xf16>)
  func.func @store_cop_f16_no_opsel(%arg0: !gpu.mma_matrix<16x16xf16, "COp">) -> () {
    %wg_1 = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    gpu.subgroup_mma_store_matrix %arg0, %wg_1[%i, %j] {leadDimension = 32 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
    // CHECK:       llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[BASE:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK:       %[[C32_0:.*]] = llvm.mlir.constant(32 : index) : i32
    // CHECK:       %[[WARPLOCALTID:.*]] = llvm.sub %{{.*}}, %{{.*}}  : i32
    // CHECK-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT:  %[[WRAPPEDTID:.*]] = llvm.srem %[[WARPLOCALTID]], %[[C16]]  : i32
    // CHECK-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // CHECK-NEXT:  %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // CHECK-NEXT:  %[[WTIDDIV16:.*]] = llvm.sdiv %[[WARPLOCALTID]], %[[C16]]  : i32
    // CHECK-NEXT:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK-NEXT:  %[[ITER0:.*]] = llvm.mul %[[C0]], %[[C2]] : i32
    // CHECK-NEXT:  %[[ROW0:.*]] = llvm.add %[[WTIDDIV16]], %[[ITER0]]  : i32
    // CHECK-NEXT:  %[[ROWLDM0:.*]] = llvm.mul %[[ROW0]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET0:.*]] = llvm.add %[[ROWLDM0]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS0:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET0]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[ELE0:.*]] = llvm.extractelement %[[SRC]][%[[ITER0]] : i32] : vector<16xf16>
    // CHECK-NEXT:  llvm.store %[[ELE0]], %[[ADDRESS0]] : f16, !llvm.ptr<3>
    // CHECK-NEXT:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT:  %[[ITER1:.*]] = llvm.mul %[[C1]], %[[C2]] : i32
    // CHECK-NEXT:  %[[ROW1:.*]] = llvm.add %[[WTIDDIV16]], %[[ITER1]]  : i32
    // CHECK-NEXT:  %[[ROWLDM1:.*]] = llvm.mul %[[ROW1]], %[[C32_0]]  : i32
    // CHECK-NEXT:  %[[OFFSET1:.*]] = llvm.add %[[ROWLDM1]], %[[WRAPPEDTID]]  : i32
    // CHECK-NEXT:  %[[ADDRESS1:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[ELE1:.*]] = llvm.extractelement %[[SRC]][%[[ITER1]] : i32] : vector<16xf16>
    // CHECK-NEXT:  llvm.store %[[ELE1]], %[[ADDRESS1]] : f16, !llvm.ptr<3>
    // CHECK:       %[[C15:.*]] = llvm.mlir.constant(15 : i32) : i32
    // CHECK:       %[[ITER15:.*]] = llvm.mul %[[C15]], %[[C2]] : i32
    // CHECK:       %[[ADDRESS15:.*]] = llvm.getelementptr %[[BASE]][%{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // CHECK-NEXT:  %[[ELE15:.*]] = llvm.extractelement %[[SRC]][%[[ITER15]] : i32] : vector<16xf16>
    // CHECK-NEXT:  llvm.store %[[ELE15]], %[[ADDRESS15]] : f16, !llvm.ptr<3>
    return
  }
}

// -----

gpu.module @test_module {
  // OPSEL-LABEL: store_cop_f16_opsel
  // OPSEL-SAME: (%[[SRC:.*]]: vector<16xf16>)
  func.func @store_cop_f16_opsel(%arg0: !gpu.mma_matrix<16x16xf16, "COp">) -> () {
    %wg_1 = memref.alloca() {alignment = 32} : memref<32x32xf16, 3>
    %i = arith.constant 16 : index
    %j = arith.constant 16 : index
    gpu.subgroup_mma_store_matrix %arg0, %wg_1[%i, %j] {leadDimension = 32 : index} : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
    // OPSEL:       llvm.mlir.constant(32 : index) : i32
    // OPSEL:       %[[BASE:.*]] = llvm.getelementptr %{{.*}} : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // OPSEL:       %[[C32_0:.*]] = llvm.mlir.constant(32 : index) : i32
    // OPSEL:       %[[WARPLOCALTID:.*]] = llvm.sub %{{.*}}, %{{.*}}  : i32
    // OPSEL-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // OPSEL-NEXT:  %[[WRAPPEDTID:.*]] = llvm.srem %[[WARPLOCALTID]], %[[C16]]  : i32
    // OPSEL-NEXT:  %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
    // OPSEL-NEXT:  %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
    // OPSEL-NEXT:  %[[WTIDDIV16:.*]] = llvm.sdiv %[[WARPLOCALTID]], %[[C16]]  : i32
    // OPSEL-NEXT:  %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
    // OPSEL-NEXT:  %[[ITER0:.*]] = llvm.mul %[[C0]], %[[C2]] : i32
    // OPSEL-NEXT:  %[[ROW0:.*]] = llvm.add %[[WTIDDIV16]], %[[ITER0]]  : i32
    // OPSEL-NEXT:  %[[ROWLDM0:.*]] = llvm.mul %[[ROW0]], %[[C32_0]]  : i32
    // OPSEL-NEXT:  %[[OFFSET0:.*]] = llvm.add %[[ROWLDM0]], %[[WRAPPEDTID]]  : i32
    // OPSEL-NEXT:  %[[ADDRESS0:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET0]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // OPSEL-NEXT:  %[[C01:.*]] = llvm.mlir.constant(1 : i32) : i32
    // OPSEL-NEXT:  %[[INX0:.*]] = llvm.add %[[ITER0]], %[[C01]]  : i32
    // OPSEL-NEXT:  %[[ELE0:.*]] = llvm.extractelement %[[SRC]][%[[INX0]] : i32] : vector<16xf16>
    // OPSEL-NEXT:  llvm.store %[[ELE0]], %[[ADDRESS0]] : f16, !llvm.ptr<3>
    // OPSEL-NEXT:  %[[C1:.*]] = llvm.mlir.constant(1 : i32) : i32
    // OPSEL-NEXT:  %[[ITER1:.*]] = llvm.mul %[[C1]], %[[C2]] : i32
    // OPSEL-NEXT:  %[[ROW1:.*]] = llvm.add %[[WTIDDIV16]], %[[ITER1]]  : i32
    // OPSEL-NEXT:  %[[ROWLDM1:.*]] = llvm.mul %[[ROW1]], %[[C32_0]]  : i32
    // OPSEL-NEXT:  %[[OFFSET1:.*]] = llvm.add %[[ROWLDM1]], %[[WRAPPEDTID]]  : i32
    // OPSEL-NEXT:  %[[ADDRESS1:.*]] = llvm.getelementptr %[[BASE]][%[[OFFSET1]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // OPSEL-NEXT:  %[[C11:.*]] = llvm.mlir.constant(1 : i32) : i32
    // OPSEL-NEXT:  %[[INX1:.*]] = llvm.add %[[ITER1]], %[[C11]]  : i32
    // OPSEL-NEXT:  %[[ELE1:.*]] = llvm.extractelement %[[SRC]][%[[INX1]] : i32] : vector<16xf16>
    // OPSEL-NEXT:  llvm.store %[[ELE1]], %[[ADDRESS1]] : f16, !llvm.ptr<3>
    // OPSEL:       %[[C15:.*]] = llvm.mlir.constant(15 : i32) : i32
    // OPSEL-NEXT:  %[[ITER15:.*]] = llvm.mul %[[C15]], %[[C2]] : i32
    // OPSEL:       %[[ADDRESS15:.*]] = llvm.getelementptr %[[BASE]][%{{.*}}] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, f16
    // OPSEL-NEXT:  %[[C151:.*]] = llvm.mlir.constant(1 : i32) : i32
    // OPSEL-NEXT:  %[[INX15:.*]] = llvm.add %[[ITER15]], %[[C151]]  : i32
    // OPSEL-NEXT:  %[[ELE15:.*]] = llvm.extractelement %[[SRC]][%[[INX15]] : i32] : vector<16xf16>
    // OPSEL-NEXT:  llvm.store %[[ELE15]], %[[ADDRESS15]] : f16, !llvm.ptr<3>
    return
  }
}
