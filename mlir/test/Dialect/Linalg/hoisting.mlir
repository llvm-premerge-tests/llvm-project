// RUN: mlir-opt  -test-transform-dialect-interpreter -canonicalize --split-input-file --allow-unregistered-dialect %s | FileCheck %s

// CHECK-LABEL: func @hoist_vector_transfer_pairs(
//  CHECK-SAME:   %[[MEMREF0:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF1:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF2:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF3:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF4:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF5:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[VAL:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[LB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[UB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[STEP:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[CMP:[a-zA-Z0-9]*]]: i1
func.func @hoist_vector_transfer_pairs(
    %memref0: memref<?x?xf32>, %memref1: memref<?x?xf32>, %memref2: memref<?x?xf32>,
    %memref3: memref<?x?xf32>, %memref4: memref<?x?xf32>, %memref5: memref<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index, %cmp: i1) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<1xf32>
// CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) -> (vector<1xf32>) {
// CHECK:   vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<2xf32>
// CHECK:   scf.for %[[J:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) -> (vector<1xf32>, vector<2xf32>) {
// CHECK:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<3xf32>
// CHECK:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<4xf32>
// CHECK:     "some_crippling_use"(%[[MEMREF4]]) : (memref<?x?xf32>) -> ()
// CHECK:     vector.transfer_read %{{.*}} : memref<?x?xf32>, vector<5xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<1xf32>) -> vector<1xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%[[MEMREF2]], %{{.*}}) : (memref<?x?xf32>, vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<5xf32>) -> vector<5xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<3xf32>, memref<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<4xf32>, memref<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<5xf32>, memref<?x?xf32>
// CHECK:     "some_crippling_use"(%[[MEMREF3]]) : (memref<?x?xf32>) -> ()
// CHECK:     scf.yield {{.*}} : vector<1xf32>, vector<2xf32>
// CHECK:   }
// CHECK:   vector.transfer_write %{{.*}} : vector<2xf32>, memref<?x?xf32>
// CHECK:   "unrelated_use"(%[[MEMREF0]]) : (memref<?x?xf32>) -> ()
// CHECK:   scf.yield {{.*}} : vector<1xf32>
// CHECK: }
// CHECK: vector.transfer_write %{{.*}} : vector<1xf32>, memref<?x?xf32>
// CHECK: "unrelated_use"(%[[MEMREF1]]) : (memref<?x?xf32>) -> ()
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %r0 = vector.transfer_read %memref1[%c0, %c0], %cst: memref<?x?xf32>, vector<1xf32>
      %r1 = vector.transfer_read %memref0[%i, %i], %cst: memref<?x?xf32>, vector<2xf32>
      %r2 = vector.transfer_read %memref2[%c0, %c0], %cst: memref<?x?xf32>, vector<3xf32>
      %r3 = vector.transfer_read %memref3[%c0, %c0], %cst: memref<?x?xf32>, vector<4xf32>
      "some_crippling_use"(%memref4) : (memref<?x?xf32>) -> ()
      %r4 = vector.transfer_read %memref4[%c0, %c0], %cst: memref<?x?xf32>, vector<5xf32>
      %r5 = vector.transfer_read %memref5[%c0, %c0], %cst: memref<?x?xf32>, vector<6xf32>
      "some_crippling_use"(%memref5) : (memref<?x?xf32>) -> ()
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%memref2, %r2) : (memref<?x?xf32>, vector<3xf32>) -> vector<3xf32>
      %u3 = "some_use"(%r3) : (vector<4xf32>) -> vector<4xf32>
      %u4 = "some_use"(%r4) : (vector<5xf32>) -> vector<5xf32>
      %u5 = "some_use"(%r5) : (vector<6xf32>) -> vector<6xf32>
      vector.transfer_write %u0, %memref1[%c0, %c0] : vector<1xf32>, memref<?x?xf32>
      vector.transfer_write %u1, %memref0[%i, %i] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u2, %memref2[%c0, %c0] : vector<3xf32>, memref<?x?xf32>
      vector.transfer_write %u3, %memref3[%c0, %c0] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u4, %memref4[%c0, %c0] : vector<5xf32>, memref<?x?xf32>
      vector.transfer_write %u5, %memref5[%c0, %c0] : vector<6xf32>, memref<?x?xf32>
      "some_crippling_use"(%memref3) : (memref<?x?xf32>) -> ()
    }
    "unrelated_use"(%memref0) : (memref<?x?xf32>) -> ()
  }
  "unrelated_use"(%memref1) : (memref<?x?xf32>) -> ()
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_disjoint(
//  CHECK-SAME:   %[[MEMREF0:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF1:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF2:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[MEMREF3:[a-zA-Z0-9]*]]: memref<?x?xf32>,
//  CHECK-SAME:   %[[VAL:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[LB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[UB:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[STEP:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[RANDOM:[a-zA-Z0-9]*]]: index,
//  CHECK-SAME:   %[[CMP:[a-zA-Z0-9]*]]: i1
func.func @hoist_vector_transfer_pairs_disjoint(
    %memref0: memref<?x?xf32>, %memref1: memref<?x?xf32>,
    %memref2: memref<?x?xf32>, %memref3: memref<?x?xf32>, %val: index, %lb : index, %ub : index,
    %step: index, %random_index : index, %cmp: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %[[MEMREF2]]{{.*}} : memref<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[MEMREF2]]{{.*}} : memref<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[MEMREF3]]{{.*}} : memref<?x?xf32>, vector<4xf32>
// CHECK: vector.transfer_read %[[MEMREF3]]{{.*}} : memref<?x?xf32>, vector<4xf32>
// CHECK: scf.for %[[I:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) ->
//  CHECK-SAME: (vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:   scf.for %[[J:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args({{.*}}) ->
//  CHECK-SAME: (vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:     vector.transfer_read %[[MEMREF1]]{{.*}} : memref<?x?xf32>, vector<2xf32>
// CHECK:     vector.transfer_read %[[MEMREF1]]{{.*}} : memref<?x?xf32>, vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     vector.transfer_write %{{.*}}, %[[MEMREF1]]{{.*}} : vector<2xf32>, memref<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}}, %[[MEMREF1]]{{.*}} : vector<2xf32>, memref<?x?xf32>
// CHECK:     scf.yield {{.*}} : vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK:   }
// CHECK:   scf.yield {{.*}} : vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK: }
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF3]]{{.*}} : vector<4xf32>, memref<?x?xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF3]]{{.*}} : vector<4xf32>, memref<?x?xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF2]]{{.*}} : vector<3xf32>, memref<?x?xf32>
// CHECK: vector.transfer_write %{{.*}}, %[[MEMREF2]]{{.*}} : vector<3xf32>, memref<?x?xf32>
  scf.for %i = %lb to %ub step %step {
    scf.for %j = %lb to %ub step %step {
      %r00 = vector.transfer_read %memref1[%c0, %c0], %cst: memref<?x?xf32>, vector<2xf32>
      %r01 = vector.transfer_read %memref1[%c0, %c1], %cst: memref<?x?xf32>, vector<2xf32>
      %r20 = vector.transfer_read %memref2[%c0, %c0], %cst: memref<?x?xf32>, vector<3xf32>
      %r21 = vector.transfer_read %memref2[%c0, %c3], %cst: memref<?x?xf32>, vector<3xf32>
      %r30 = vector.transfer_read %memref3[%c0, %random_index], %cst: memref<?x?xf32>, vector<4xf32>
      %r31 = vector.transfer_read %memref3[%c1, %random_index], %cst: memref<?x?xf32>, vector<4xf32>
      %r10 = vector.transfer_read %memref0[%i, %i], %cst: memref<?x?xf32>, vector<2xf32>
      %r11 = vector.transfer_read %memref0[%random_index, %random_index], %cst: memref<?x?xf32>, vector<2xf32>
      %u00 = "some_use"(%r00) : (vector<2xf32>) -> vector<2xf32>
      %u01 = "some_use"(%r01) : (vector<2xf32>) -> vector<2xf32>
      %u20 = "some_use"(%r20) : (vector<3xf32>) -> vector<3xf32>
      %u21 = "some_use"(%r21) : (vector<3xf32>) -> vector<3xf32>
      %u30 = "some_use"(%r30) : (vector<4xf32>) -> vector<4xf32>
      %u31 = "some_use"(%r31) : (vector<4xf32>) -> vector<4xf32>
      %u10 = "some_use"(%r10) : (vector<2xf32>) -> vector<2xf32>
      %u11 = "some_use"(%r11) : (vector<2xf32>) -> vector<2xf32>
      vector.transfer_write %u00, %memref1[%c0, %c0] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u01, %memref1[%c0, %c1] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u20, %memref2[%c0, %c0] : vector<3xf32>, memref<?x?xf32>
      vector.transfer_write %u21, %memref2[%c0, %c3] : vector<3xf32>, memref<?x?xf32>
      vector.transfer_write %u30, %memref3[%c0, %random_index] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u31, %memref3[%c1, %random_index] : vector<4xf32>, memref<?x?xf32>
      vector.transfer_write %u10, %memref0[%i, %i] : vector<2xf32>, memref<?x?xf32>
      vector.transfer_write %u11, %memref0[%random_index, %random_index] : vector<2xf32>, memref<?x?xf32>
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_in_affine_loops(
//  CHECK-SAME:   %[[MEMREF0:[a-zA-Z0-9]+]]: memref<64x64xi32>,
//  CHECK-SAME:   %[[MEMREF1:[a-zA-Z0-9]+]]: memref<64x64xi32>,
//  CHECK-SAME:   %[[MEMREF2:[a-zA-Z0-9]+]]: memref<64x64xi32>) {
//       CHECK:   %[[C0:.*]] = arith.constant 0 : i32
//       CHECK:   affine.for %[[I:.*]] = 0 to 64 {
//       CHECK:     affine.for %[[J:.*]] = 0 to 64 step 16 {
//       CHECK:       %[[R0:.*]] = vector.transfer_read %[[MEMREF2]][%[[I]], %[[J]]], %[[C0]] : memref<64x64xi32>, vector<16xi32>
//       CHECK:       %[[R:.*]] = affine.for %[[K:.*]] = 0 to 64 iter_args(%[[ACC:.*]] = %[[R0]]) -> (vector<16xi32>) {
//       CHECK:         %[[AV:.*]] = vector.transfer_read %[[MEMREF0]][%[[I]], %[[K]]], %[[C0]] {{.*}}: memref<64x64xi32>, vector<16xi32>
//       CHECK:         %[[BV:.*]] = vector.transfer_read %[[MEMREF1]][%[[K]], %[[J]]], %[[C0]] {{.*}}: memref<64x64xi32>, vector<16xi32>
//       CHECK:         %[[T0:.*]] = arith.muli %[[AV]], %[[BV]] : vector<16xi32>
//       CHECK:         %[[T1:.*]] = arith.addi %[[ACC]], %[[T0]] : vector<16xi32>
//       CHECK:         affine.yield %[[T1]] : vector<16xi32>
//       CHECK:       }
//       CHECK:       vector.transfer_write %[[R]], %[[MEMREF2]][%[[I]], %[[J]]] : vector<16xi32>, memref<64x64xi32>
//       CHECK:     }
//       CHECK:   }
func.func @hoist_vector_transfer_pairs_in_affine_loops(%memref0: memref<64x64xi32>, %memref1: memref<64x64xi32>, %memref2: memref<64x64xi32>) {
  %c0_i32 = arith.constant 0 : i32
  affine.for %arg3 = 0 to 64 {
    affine.for %arg4 = 0 to 64 step 16 {
      affine.for %arg5 = 0 to 64 {
        %0 = vector.transfer_read %memref0[%arg3, %arg5], %c0_i32 {permutation_map = affine_map<(d0, d1) -> (0)>} : memref<64x64xi32>, vector<16xi32>
        %1 = vector.transfer_read %memref1[%arg5, %arg4], %c0_i32 : memref<64x64xi32>, vector<16xi32>
        %2 = vector.transfer_read %memref2[%arg3, %arg4], %c0_i32 : memref<64x64xi32>, vector<16xi32>
        %3 = arith.muli %0, %1 : vector<16xi32>
        %4 = arith.addi %2, %3 : vector<16xi32>
        vector.transfer_write %4, %memref2[%arg3, %arg4] : vector<16xi32>, memref<64x64xi32>
      }
    }
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_tensor
func.func @hoist_vector_transfer_pairs_tensor(
    %tensor0: tensor<?x?xf32>, %tensor1: tensor<?x?xf32>, %tensor2: tensor<?x?xf32>,
    %tensor3: tensor<?x?xf32>, %tensor4: tensor<?x?xf32>, %tensor5: tensor<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index) ->
    (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
     tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<1xf32>
// CHECK: scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>) {
// CHECK:   vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:   scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<2xf32>, vector<1xf32>) {
// CHECK:     vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<4xf32>
// CHECK:     "some_crippling_use"(%{{.*}}) : (tensor<?x?xf32>) -> ()
// CHECK:     vector.transfer_read %{{.*}} : tensor<?x?xf32>, vector<5xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<1xf32>) -> vector<1xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (tensor<?x?xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<5xf32>) -> vector<5xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<3xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}} : vector<5xf32>, tensor<?x?xf32>
// CHECK:     "some_crippling_use"(%{{.*}}) : (tensor<?x?xf32>) -> ()
// CHECK:     scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<2xf32>, vector<1xf32>
// CHECK:   }
// CHECK:   vector.transfer_write %{{.*}} : vector<2xf32>, tensor<?x?xf32>
// CHECK:   scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>
// CHECK: }
// CHECK: vector.transfer_write %{{.*}} : vector<1xf32>, tensor<?x?xf32>
  %0:6 = scf.for %i = %lb to %ub step %step
  iter_args(%arg0 = %tensor0, %arg1 = %tensor1, %arg2 = %tensor2,
            %arg3 = %tensor3,  %arg4 = %tensor4, %arg5 = %tensor5)
  -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
     tensor<?x?xf32>, tensor<?x?xf32>)  {
    %1:6 = scf.for %j = %lb to %ub step %step
    iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %arg2,
              %arg9 = %arg3,  %arg10 = %arg4, %arg11 = %arg5)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
       tensor<?x?xf32>, tensor<?x?xf32>)  {
      %r0 = vector.transfer_read %arg7[%c0, %c0], %cst: tensor<?x?xf32>, vector<1xf32>
      %r1 = vector.transfer_read %arg6[%i, %i], %cst: tensor<?x?xf32>, vector<2xf32>
      %r3 = vector.transfer_read %arg9[%c0, %c0], %cst: tensor<?x?xf32>, vector<4xf32>
      "some_crippling_use"(%arg10) : (tensor<?x?xf32>) -> ()
      %r4 = vector.transfer_read %arg10[%c0, %c0], %cst: tensor<?x?xf32>, vector<5xf32>
      %r5 = vector.transfer_read %arg11[%c0, %c0], %cst: tensor<?x?xf32>, vector<6xf32>
      "some_crippling_use"(%arg11) : (tensor<?x?xf32>) -> ()
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%arg8) : (tensor<?x?xf32>) -> vector<3xf32>
      %u3 = "some_use"(%r3) : (vector<4xf32>) -> vector<4xf32>
      %u4 = "some_use"(%r4) : (vector<5xf32>) -> vector<5xf32>
      %u5 = "some_use"(%r5) : (vector<6xf32>) -> vector<6xf32>
      %w1 = vector.transfer_write %u0, %arg7[%c0, %c0] : vector<1xf32>, tensor<?x?xf32>
      %w0 = vector.transfer_write %u1, %arg6[%i, %i] : vector<2xf32>, tensor<?x?xf32>
      %w2 = vector.transfer_write %u2, %arg8[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>
      %w3 = vector.transfer_write %u3, %arg9[%c0, %c0] : vector<4xf32>, tensor<?x?xf32>
      %w4 = vector.transfer_write %u4, %arg10[%c0, %c0] : vector<5xf32>, tensor<?x?xf32>
      %w5 = vector.transfer_write %u5, %arg11[%c0, %c0] : vector<6xf32>, tensor<?x?xf32>
      "some_crippling_use"(%w3) : (tensor<?x?xf32>) -> ()
      scf.yield %w0, %w1, %w2, %w3, %w4, %w5 :
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
        tensor<?x?xf32>, tensor<?x?xf32>
      }
      scf.yield %1#0,  %1#1, %1#2, %1#3, %1#4, %1#5 :
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
        tensor<?x?xf32>, tensor<?x?xf32>
  }
  return %0#0,  %0#1, %0#2, %0#3, %0#4,  %0#5 :
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>,
        tensor<?x?xf32>, tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_tensor_subsets %0
    : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_disjoint_tensor(
//  CHECK-SAME:   %[[TENSOR0:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR1:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR2:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR3:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
func.func @hoist_vector_transfer_pairs_disjoint_tensor(
    %tensor0: tensor<?x?xf32>, %tensor1: tensor<?x?xf32>,
    %tensor2: tensor<?x?xf32>, %tensor3: tensor<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index,
    %random_index : index) ->
    (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.0 : f32

// CHECK: vector.transfer_read %[[TENSOR2]]{{.*}} : tensor<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[TENSOR2]]{{.*}} : tensor<?x?xf32>, vector<3xf32>
// CHECK: vector.transfer_read %[[TENSOR3]]{{.*}} : tensor<?x?xf32>, vector<4xf32>
// CHECK: vector.transfer_read %[[TENSOR3]]{{.*}} : tensor<?x?xf32>, vector<4xf32>
// CHECK: %[[R:.*]]:6 = scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:   scf.for {{.*}} iter_args({{.*}}) ->
// CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>) {
// CHECK:     vector.transfer_read %[[TENSOR1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:     vector.transfer_read %[[TENSOR1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<3xf32>) -> vector<3xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     "some_use"(%{{.*}}) : (vector<2xf32>) -> vector<2xf32>
// CHECK:     vector.transfer_write %{{.*}}, %{{.*}}{{.*}} : vector<2xf32>, tensor<?x?xf32>
// CHECK:     vector.transfer_write %{{.*}}, %{{.*}}{{.*}} : vector<2xf32>, tensor<?x?xf32>
// CHECK:     scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK:   }
// CHECK:   scf.yield {{.*}} :
// CHECK-SAME: tensor<?x?xf32>, tensor<?x?xf32>, vector<3xf32>, vector<3xf32>, vector<4xf32>, vector<4xf32>
// CHECK: }
// CHECK: %[[TENSOR4:.*]] = vector.transfer_write %[[R]]#5, %[[TENSOR3]]{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK:                   vector.transfer_write %[[R]]#4, %[[TENSOR4]]{{.*}} : vector<4xf32>, tensor<?x?xf32>
// CHECK: %[[TENSOR5:.*]] = vector.transfer_write %[[R]]#3, %[[TENSOR2]]{{.*}} : vector<3xf32>, tensor<?x?xf32>
// CHECK:                   vector.transfer_write %[[R]]#2, %[[TENSOR5]]{{.*}} : vector<3xf32>, tensor<?x?xf32>
  %0:4 = scf.for %i = %lb to %ub step %step
  iter_args(%arg0 = %tensor0, %arg1 = %tensor1, %arg2 = %tensor2,
            %arg3 = %tensor3)
  -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
    %1:4 = scf.for %j = %lb to %ub step %step
    iter_args(%arg4 = %arg0, %arg5 = %arg1, %arg6 = %arg2,
              %arg7 = %arg3)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) {
      %r00 = vector.transfer_read %arg5[%c0, %c0], %cst: tensor<?x?xf32>, vector<2xf32>
      %r01 = vector.transfer_read %arg5[%c0, %c1], %cst: tensor<?x?xf32>, vector<2xf32>
      %r20 = vector.transfer_read %arg6[%c0, %c0], %cst: tensor<?x?xf32>, vector<3xf32>
      %r21 = vector.transfer_read %arg6[%c0, %c3], %cst: tensor<?x?xf32>, vector<3xf32>
      %r30 = vector.transfer_read %arg7[%c0, %random_index], %cst: tensor<?x?xf32>, vector<4xf32>
      %r31 = vector.transfer_read %arg7[%c1, %random_index], %cst: tensor<?x?xf32>, vector<4xf32>
      %r10 = vector.transfer_read %arg4[%i, %i], %cst: tensor<?x?xf32>, vector<2xf32>
      %r11 = vector.transfer_read %arg4[%random_index, %random_index], %cst: tensor<?x?xf32>, vector<2xf32>
      %u00 = "some_use"(%r00) : (vector<2xf32>) -> vector<2xf32>
      %u01 = "some_use"(%r01) : (vector<2xf32>) -> vector<2xf32>
      %u20 = "some_use"(%r20) : (vector<3xf32>) -> vector<3xf32>
      %u21 = "some_use"(%r21) : (vector<3xf32>) -> vector<3xf32>
      %u30 = "some_use"(%r30) : (vector<4xf32>) -> vector<4xf32>
      %u31 = "some_use"(%r31) : (vector<4xf32>) -> vector<4xf32>
      %u10 = "some_use"(%r10) : (vector<2xf32>) -> vector<2xf32>
      %u11 = "some_use"(%r11) : (vector<2xf32>) -> vector<2xf32>
      %w10 = vector.transfer_write %u00, %arg5[%c0, %c0] : vector<2xf32>, tensor<?x?xf32>
      %w11 = vector.transfer_write %u01, %w10[%c0, %c1] : vector<2xf32>, tensor<?x?xf32>
      %w20 = vector.transfer_write %u20, %arg6[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>
      %w21 = vector.transfer_write %u21, %w20[%c0, %c3] : vector<3xf32>, tensor<?x?xf32>
      %w30 = vector.transfer_write %u30, %arg7[%c0, %random_index] : vector<4xf32>, tensor<?x?xf32>
      %w31 = vector.transfer_write %u31, %w30[%c1, %random_index] : vector<4xf32>, tensor<?x?xf32>
      %w00 = vector.transfer_write %u10, %arg4[%i, %i] : vector<2xf32>, tensor<?x?xf32>
      %w01 = vector.transfer_write %u11, %w00[%random_index, %random_index] : vector<2xf32>, tensor<?x?xf32>
      scf.yield %w01, %w11, %w21, %w31 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    }
    scf.yield %1#0,  %1#1, %1#2, %1#3 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
  }
  return %0#0,  %0#1, %0#2, %0#3 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_tensor_subsets %0
    : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_tensor_and_slices
//  CHECK-SAME:   %[[TENSOR0:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR1:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR2:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR3:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR4:[a-zA-Z0-9]*]]: tensor<?x?xf32>,
//  CHECK-SAME:   %[[TENSOR5:[a-zA-Z0-9]*]]: tensor<?x?xf32>
func.func @hoist_vector_transfer_pairs_tensor_and_slices(
    %tensor0: tensor<?x?xf32>, %tensor1: tensor<?x?xf32>, %tensor2: tensor<?x?xf32>,
    %tensor3: tensor<?x?xf32>, %tensor4: tensor<?x?xf32>, %tensor5: tensor<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index) ->
    (
      tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>//, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    ) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

  //      CHECK: scf.for %[[I:.*]] = {{.*}} iter_args(
  // CHECK-SAME:   %[[TENSOR0_ARG:[0-9a-zA-Z]+]] = %[[TENSOR0]],
  // CHECK-SAME:   %[[TENSOR1_ARG:[0-9a-zA-Z]+]] = %[[TENSOR1]],
  // CHECK-SAME:   %[[TENSOR2_ARG:[0-9a-zA-Z]+]] = %[[TENSOR2]]
  // CHECK-SAME: ) ->
  // CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
  %0:3 = scf.for %i = %lb to %ub step %step
  iter_args(%arg0 = %tensor0, %arg1 = %tensor1, %arg2 = %tensor2)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)  {

    // Hoisted
    // CHECK:   %[[ST0:.*]] = tensor.extract_slice %[[TENSOR0_ARG]][%[[I]], %[[I]]]{{.*}}: tensor<?x?xf32> to tensor<?x?xf32>
    // CHECK:   %[[V0:.*]] = vector.transfer_read %[[ST0]]{{.*}} : tensor<?x?xf32>, vector<1xf32>

    //      CHECK:   %[[R:.*]]:3 = scf.for %[[J:.*]] = {{.*}} iter_args(
    // CHECK-SAME:   %[[TENSOR1_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR1_ARG]]
    // CHECK-SAME:   %[[TENSOR2_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR2_ARG]]
    // CHECK-SAME:   %[[V0_ARG_L2:[0-9a-zA-Z]+]] = %[[V0]]
    // CHECK-SAME: ) ->
    // CHECK-SAME: (tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>
    %1:3 = scf.for %j = %lb to %ub step %step
    iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %arg2)
    -> (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>)  {
      // Hoists.
      %st0 = tensor.extract_slice %arg6[%i, %i][%step, %step][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %r0 = vector.transfer_read %st0[%c0, %c0], %cst: tensor<?x?xf32>, vector<1xf32>

      // CHECK:     %[[ST1:.*]] = tensor.extract_slice %[[TENSOR1_ARG_L2]][%[[J]],{{.*}}: tensor<?x?xf32> to tensor<?x?xf32>
      // CHECK:     %[[V1:.*]] = vector.transfer_read %[[ST1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
      // Does not hoist (slice depends on %j)
      %st1 = tensor.extract_slice %arg7[%j, %c0][%step, %step][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %r1 = vector.transfer_read %st1[%c0, %c0], %cst: tensor<?x?xf32>, vector<2xf32>

      // CHECK:     %[[ST2:.*]] = tensor.extract_slice %[[TENSOR2_ARG_L2]][%[[I]],{{.*}}: tensor<?x?xf32> to tensor<?x?xf32>
      // CHECK:     %[[V2:.*]] = vector.transfer_read %[[ST2]]{{.*}} : tensor<?x?xf32>, vector<3xf32>
      // Does not hoist, 2 slice %arg8.
      %st2 = tensor.extract_slice %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %r2 = vector.transfer_read %st2[%c0, %c0], %cst: tensor<?x?xf32>, vector<3xf32>

      // CHECK:     %[[U0:.*]] = "some_use"(%[[V0_ARG_L2]]) : (vector<1xf32>) -> vector<1xf32>
      // CHECK:     %[[U1:.*]] = "some_use"(%[[V1]]) : (vector<2xf32>) -> vector<2xf32>
      // CHECK:     %[[U2:.*]] = "some_use"(%[[V2]]) : (vector<3xf32>) -> vector<3xf32>
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%r2) : (vector<3xf32>) -> vector<3xf32>

      // Hoists
      %w0 = vector.transfer_write %u0, %st0[%c0, %c0] : vector<1xf32>, tensor<?x?xf32>

      // CHECK-DAG:     %[[STI1:.*]] = vector.transfer_write %[[U1]], %{{.*}} : vector<2xf32>, tensor<?x?xf32>
      // Does not hoist (associated slice depends on %j).
      %w1 = vector.transfer_write %u1, %st1[%i, %i] : vector<2xf32>, tensor<?x?xf32>

      // CHECK-DAG:     %[[STI2:.*]] = vector.transfer_write %[[U2]], %{{.*}} : vector<3xf32>, tensor<?x?xf32>
      // Does not hoist, 2 slice / insert_slice for %arg8.
      %w2 = vector.transfer_write %u2, %st2[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>

      // Hoists.
      %sti0 = tensor.insert_slice %w0 into %arg6[%i, %i][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

      // CHECK-DAG:     tensor.insert_slice %[[STI1]] into %[[TENSOR1_ARG_L2]][%[[J]],{{.*}}: tensor<?x?xf32> into tensor<?x?xf32>
      // Does not hoist (depends on %j).
      %sti1 = tensor.insert_slice %w1 into %arg7[%j, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

      // CHECK-DAG:     tensor.insert_slice %[[STI2]] into %[[TENSOR2_ARG_L2]][%[[I]],{{.*}}: tensor<?x?xf32> into tensor<?x?xf32>
      // Does not hoist, 2 slice / insert_slice for %arg8.
      %sti2 = tensor.insert_slice %w2 into %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
      // Extract with a different stride to make sure we cannot fold this extract with the above insert.
      %st22 = tensor.extract_slice %sti2[%i, %c0][%step, %step][2, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %sti22 = tensor.insert_slice %st22 into %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>

      // CHECK:     scf.yield {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>, vector<1xf32>
      // CHECK:   }
      scf.yield %sti0, %sti1, %sti22:
        tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    }

    // Hoisted
    // CHECK:   %[[STI0:.*]] = vector.transfer_write %[[R]]#2, %[[ST0]]{{.*}} : vector<1xf32>, tensor<?x?xf32>
    // CHECK:   tensor.insert_slice %[[STI0]] into %[[TENSOR0_ARG]][%[[I]], %[[I]]]{{.*}} : tensor<?x?xf32> into tensor<?x?xf32>

    // CHECK:   scf.yield {{.*}} : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
    scf.yield %1#0, %1#1, %1#2 :
      tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>

    // CHECK: }
  }
  return %0#0, %0#1, %0#2 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_tensor_subsets %0
    : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_write_pairs_disjoint_tensor(
//  CHECK-SAME:   %[[T:.*]]: tensor<?x?xf32>,
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[R0:.*]] = vector.transfer_read %[[T]][%[[C0]], %[[C0]]], %{{.*}} : tensor<?x?xf32>, vector<2xf32>
//   CHECK-DAG:   %[[R1:.*]] = vector.transfer_read %[[T]][%[[C0]], %[[C3]]], %{{.*}} : tensor<?x?xf32>, vector<2xf32>
//       CHECK:   %[[F:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[R3:.*]] = %[[R1:.*]], %[[R2:.*]] = %[[R0]]) -> (vector<2xf32>, vector<2xf32>) {
//       CHECK:     %[[R4:.*]] = "some_use"(%[[R2]]) : (vector<2xf32>) -> vector<2xf32>
//       CHECK:     %[[R5:.*]] = "some_use"(%[[R3]]) : (vector<2xf32>) -> vector<2xf32>
//       CHECK:     scf.yield %[[R5]], %[[R4]] : vector<2xf32>, vector<2xf32>
//       CHECK:   }
//       CHECK:   %[[W0:.*]] = vector.transfer_write %[[F]]#1, %[[T]][%[[C0]], %[[C0]]] : vector<2xf32>, tensor<?x?xf32>
//       CHECK:   %[[W1:.*]] = vector.transfer_write %[[F]]#0, %[[W0]][%[[C0]], %[[C3]]] : vector<2xf32>, tensor<?x?xf32>
//       CHECK:  return %[[W1]] : tensor<?x?xf32>
func.func @hoist_vector_transfer_write_pairs_disjoint_tensor(
    %tensor: tensor<?x?xf32>,
    %val: index, %lb : index, %ub : index, %step: index) ->
    (tensor<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %cst = arith.constant 0.0 : f32
  %1 = scf.for %j = %lb to %ub step %step iter_args(%arg5 = %tensor)
    -> (tensor<?x?xf32>) {
    %r00 = vector.transfer_read %arg5[%c0, %c0], %cst: tensor<?x?xf32>, vector<2xf32>
    %u00 = "some_use"(%r00) : (vector<2xf32>) -> vector<2xf32>
    %w10 = vector.transfer_write %u00, %arg5[%c0, %c0] : vector<2xf32>, tensor<?x?xf32>

    // Hoist by properly bypassing the disjoint write %w10.
    %r01 = vector.transfer_read %w10[%c0, %c3], %cst: tensor<?x?xf32>, vector<2xf32>
    %u01 = "some_use"(%r01) : (vector<2xf32>) -> vector<2xf32>
    %w11 = vector.transfer_write %u01, %w10[%c0, %c3] : vector<2xf32>, tensor<?x?xf32>
    scf.yield %w11 : tensor<?x?xf32>
  }
  return %1 : tensor<?x?xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_tensor_subsets %0
    : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL: func @hoist_vector_transfer_pairs_tensor_and_slices_static_large_tensor
//  CHECK-SAME:   %[[TENSOR0:[a-zA-Z0-9]*]]: tensor<100x100xf32>,
//  CHECK-SAME:   %[[TENSOR1:[a-zA-Z0-9]*]]: tensor<200x200xf32>,
//  CHECK-SAME:   %[[TENSOR2:[a-zA-Z0-9]*]]: tensor<300x300xf32>
func.func @hoist_vector_transfer_pairs_tensor_and_slices_static_large_tensor(
    %tensor0: tensor<100x100xf32>, %tensor1: tensor<200x200xf32>, %tensor2: tensor<300x300xf32>,
    %val: index, %lb : index, %ub : index, %step: index) ->
    (
      tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>
    ) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32

  //      CHECK: scf.for %[[I:.*]] = {{.*}} iter_args(
  // CHECK-SAME:   %[[TENSOR0_ARG:[0-9a-zA-Z]+]] = %[[TENSOR0]],
  // CHECK-SAME:   %[[TENSOR1_ARG:[0-9a-zA-Z]+]] = %[[TENSOR1]],
  // CHECK-SAME:   %[[TENSOR2_ARG:[0-9a-zA-Z]+]] = %[[TENSOR2]]
  // CHECK-SAME: ) ->
  // CHECK-SAME: (tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>
  %0:3 = scf.for %i = %lb to %ub step %step
  iter_args(%arg0 = %tensor0, %arg1 = %tensor1, %arg2 = %tensor2)
    -> (tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>)  {

    // Hoisted
    // CHECK:   %[[ST0:.*]] = tensor.extract_slice %[[TENSOR0_ARG]][%[[I]], %[[I]]]{{.*}}: tensor<100x100xf32> to tensor<?x?xf32>
    // CHECK:   %[[V0:.*]] = vector.transfer_read %[[ST0]]{{.*}} : tensor<?x?xf32>, vector<1xf32>

    //      CHECK:   %[[R:.*]]:3 = scf.for %[[J:.*]] = {{.*}} iter_args(
    // CHECK-SAME:   %[[TENSOR1_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR1_ARG]]
    // CHECK-SAME:   %[[TENSOR2_ARG_L2:[0-9a-zA-Z]+]] = %[[TENSOR2_ARG]]
    // CHECK-SAME:   %[[V0_ARG_L2:[0-9a-zA-Z]+]] = %[[V0]]
    // CHECK-SAME: ) ->
    // CHECK-SAME: (tensor<200x200xf32>, tensor<300x300xf32>, vector<1xf32>
    %1:3 = scf.for %j = %lb to %ub step %step
    iter_args(%arg6 = %arg0, %arg7 = %arg1, %arg8 = %arg2)
    -> (tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>)  {
      // Hoists.
      %st0 = tensor.extract_slice %arg6[%i, %i][%step, %step][1, 1] : tensor<100x100xf32> to tensor<?x?xf32>
      %r0 = vector.transfer_read %st0[%c0, %c0], %cst: tensor<?x?xf32>, vector<1xf32>

      // CHECK:     %[[ST1:.*]] = tensor.extract_slice %[[TENSOR1_ARG_L2]][%[[J]],{{.*}}: tensor<200x200xf32> to tensor<?x?xf32>
      // CHECK:     %[[V1:.*]] = vector.transfer_read %[[ST1]]{{.*}} : tensor<?x?xf32>, vector<2xf32>
      // Does not hoist (slice depends on %j)
      %st1 = tensor.extract_slice %arg7[%j, %c0][%step, %step][1, 1] : tensor<200x200xf32> to tensor<?x?xf32>
      %r1 = vector.transfer_read %st1[%c0, %c0], %cst: tensor<?x?xf32>, vector<2xf32>

      // CHECK:     %[[ST2:.*]] = tensor.extract_slice %[[TENSOR2_ARG_L2]][%[[I]],{{.*}}: tensor<300x300xf32> to tensor<?x?xf32>
      // CHECK:     %[[V2:.*]] = vector.transfer_read %[[ST2]]{{.*}} : tensor<?x?xf32>, vector<3xf32>
      // Does not hoist, 2 slice %arg8.
      %st2 = tensor.extract_slice %arg8[%i, %c0][%step, %step][1, 1] : tensor<300x300xf32> to tensor<?x?xf32>
      %r2 = vector.transfer_read %st2[%c0, %c0], %cst: tensor<?x?xf32>, vector<3xf32>

      // CHECK:     %[[U0:.*]] = "some_use"(%[[V0_ARG_L2]]) : (vector<1xf32>) -> vector<1xf32>
      // CHECK:     %[[U1:.*]] = "some_use"(%[[V1]]) : (vector<2xf32>) -> vector<2xf32>
      // CHECK:     %[[U2:.*]] = "some_use"(%[[V2]]) : (vector<3xf32>) -> vector<3xf32>
      %u0 = "some_use"(%r0) : (vector<1xf32>) -> vector<1xf32>
      %u1 = "some_use"(%r1) : (vector<2xf32>) -> vector<2xf32>
      %u2 = "some_use"(%r2) : (vector<3xf32>) -> vector<3xf32>

      // Hoists
      %w0 = vector.transfer_write %u0, %st0[%c0, %c0] : vector<1xf32>, tensor<?x?xf32>

      // CHECK-DAG:     %[[STI1:.*]] = vector.transfer_write %[[U1]], %{{.*}} : vector<2xf32>, tensor<?x?xf32>
      // Does not hoist (associated slice depends on %j).
      %w1 = vector.transfer_write %u1, %st1[%i, %i] : vector<2xf32>, tensor<?x?xf32>

      // CHECK-DAG:     %[[STI2:.*]] = vector.transfer_write %[[U2]], %{{.*}} : vector<3xf32>, tensor<?x?xf32>
      // Does not hoist, 2 slice / insert_slice for %arg8.
      %w2 = vector.transfer_write %u2, %st2[%c0, %c0] : vector<3xf32>, tensor<?x?xf32>

      // Hoists.
      %sti0 = tensor.insert_slice %w0 into %arg6[%i, %i][%step, %step][1, 1] : tensor<?x?xf32> into tensor<100x100xf32>

      // CHECK-DAG:     tensor.insert_slice %[[STI1]] into %[[TENSOR1_ARG_L2]][%[[J]],{{.*}}: tensor<?x?xf32> into tensor<200x200xf32>
      // Does not hoist (depends on %j).
      %sti1 = tensor.insert_slice %w1 into %arg7[%j, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<200x200xf32>

      // CHECK-DAG:     tensor.insert_slice %[[STI2]] into %[[TENSOR2_ARG_L2]][%[[I]],{{.*}}: tensor<?x?xf32> into tensor<300x300xf32>
      // Does not hoist, 2 slice / insert_slice for %arg8.
      %sti2 = tensor.insert_slice %w2 into %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<300x300xf32>
      // Extract with a different stride to make sure we cannot fold this extract with the above insert.
      %st22 = tensor.extract_slice %sti2[%i, %c0][%step, %step][2, 1] : tensor<300x300xf32> to tensor<?x?xf32>
      %sti22 = tensor.insert_slice %st22 into %arg8[%i, %c0][%step, %step][1, 1] : tensor<?x?xf32> into tensor<300x300xf32>

      // CHECK:     scf.yield {{.*}} : tensor<200x200xf32>, tensor<300x300xf32>, vector<1xf32>
      // CHECK:   }
      scf.yield %sti0, %sti1, %sti22:
        tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>
    }

    // Hoisted
    // CHECK:   %[[STI0:.*]] = vector.transfer_write %[[R]]#2, %[[ST0]]{{.*}} : vector<1xf32>, tensor<?x?xf32>
    // CHECK:   tensor.insert_slice %[[STI0]] into %[[TENSOR0_ARG]][%[[I]], %[[I]]]{{.*}} : tensor<?x?xf32> into tensor<100x100xf32>

    // CHECK:   scf.yield {{.*}} : tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>
    scf.yield %1#0, %1#1, %1#2 :
      tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>

    // CHECK: }
  }
  return %0#0, %0#1, %0#2 : tensor<100x100xf32>, tensor<200x200xf32>, tensor<300x300xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_tensor_subsets %0
    : (!transform.any_op) -> ()
}

// -----

// CHECK-LABEL:  func.func @hoist_vector_transfer_read(
// CHECK-DAG:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:      %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:      %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:      %[[CST:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:          %[[ALLOC:.+]] = memref.alloc() : memref<32x64xf32>
// CHECK:          %[[ALLOC_0:.+]] = memref.alloc() : memref<32x128xf32>
// CHECK:          %[[CAST:.+]] = memref.cast %[[ALLOC_0]] : memref<32x128xf32> to memref<32x128xf32, strided<[128, 1],
// CHECK-SAME:       offset: ?>>
// CHECK:          %[[D0:.+]] = vector.transfer_read %[[ALLOC]][%[[C0]], %[[C0]]], %[[CST]] {in_bounds = [true, true]} :
// CHECK-SAME:       memref<32x64xf32>, vector<32x64xf32>
// CHECK:          scf.for %[[ARG0:.+]] = %[[C0]] to %[[C1024]] step %[[C128]] {
// CHECK:            %[[D1:.+]] = vector.transfer_read %[[ALLOC_0]][%[[C0]], %[[C0]]], %[[CST]] {in_bounds = [true, true]}
// CHECK-SAME:         : memref<32x128xf32>, vector<32x128xf32>
// CHECK:            "some_use"(%[[D0]], %[[D1]], %[[CAST]]) : (vector<32x64xf32>, vector<32x128xf32>, memref<32x128xf32,
// CHECK-SAME:         strided<[128, 1], offset: ?>>) -> ()
// CHECK:          }
// CHECK:          memref.dealloc %[[ALLOC]] : memref<32x64xf32>
// CHECK:          return
func.func @hoist_vector_transfer_read() {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1024 = arith.constant 1024 : index
  %cst_2 = arith.constant 0.000000e+00 : f32
  %memref0 = memref.alloc() : memref<32x64xf32>
  %memref2 = memref.alloc() : memref<32x128xf32>
  %subview2 = memref.subview %memref2[%c0, %c0] [32, 128] [1, 1]: memref<32x128xf32> to memref<32x128xf32, strided<[128, 1], offset: ?>>
  scf.for %arg0 = %c0 to %c1024 step %c128 {
    %2 = vector.transfer_read %memref2[%c0, %c0], %cst_2 {in_bounds = [true, true]} : memref<32x128xf32>, vector<32x128xf32>
    %3 = vector.transfer_read %memref0[%c0, %c0], %cst_2 {in_bounds = [true, true]} : memref<32x64xf32>, vector<32x64xf32>
    "some_use"(%3, %2, %subview2) : (vector<32x64xf32>, vector<32x128xf32>, memref<32x128xf32, strided<[128, 1], offset: ?>>) -> ()
  }
  memref.dealloc %memref0 : memref<32x64xf32>
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

// The transfers in this test case cannot be hoisted and replaced by a vector
// iter_arg because they do not match.

// CHECK-LABEL:  func.func @non_matching_transfers(
//       CHECK:    scf.for {{.*}} {
//       CHECK:      vector.transfer_read
//       CHECK:      vector.transfer_write
//       CHECK:    }
func.func @non_matching_transfers(%m: memref<6x1x7x32xf32>) {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c128 = arith.constant 128 : index
  %cst = arith.constant dense<5.5> : vector<6x7x32xf32>
  %cst_0 = arith.constant 0.0 : f32
  scf.for %iv = %c0 to %c1024 step %c128 {
    %read = vector.transfer_read %m[%c0, %c0, %c0, %c0], %cst_0 {in_bounds = [true, true, true], permutation_map = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>} : memref<6x1x7x32xf32>, vector<6x7x32xf32>
    %added = arith.addf %read, %cst : vector<6x7x32xf32>
    %bc = vector.broadcast %added : vector<6x7x32xf32> to vector<1x6x7x32xf32>
    %tr = vector.transpose %bc, [1, 0, 2, 3] : vector<1x6x7x32xf32> to vector<6x1x7x32xf32>
    vector.transfer_write %tr, %m[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<6x1x7x32xf32>, memref<6x1x7x32xf32>
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_redundant_vector_transfers %0
    : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @hoist_unrolled_vector_for_mma(%0: memref<3456x2048xf16>, %1: memref<2048x1024xf16>, %2: memref<3456x1024xf32>, %workgroup_id_x : index) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xf32>
  %c64 = arith.constant 64 : index
  %c2048 = arith.constant 2048 : index
  %3 = gpu.thread_id  x
  %4 = gpu.thread_id  y
  %5 = affine.apply affine_map<()[s0, s1] -> (s1 * 32 + (s0 floordiv 8) * 128)>()[%workgroup_id_x, %4]
  %6 = affine.apply affine_map<()[s0, s1] -> (s0 * 128 + s1 * 32 - (s0 floordiv 8) * 1024)>()[%workgroup_id_x, %3]
  %7 = scf.for %arg0 = %c0 to %c2048 step %c64 iter_args(%arg1 = %cst_0) -> (vector<32x32xf32>) {
    %26 = vector.transfer_read %0[%5, %arg0], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %27 = affine.apply affine_map<(d0) -> (d0 + 16)>(%arg0)
    %28 = vector.transfer_read %0[%5, %27], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %29 = affine.apply affine_map<(d0) -> (d0 + 32)>(%arg0)
    %30 = vector.transfer_read %0[%5, %29], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %31 = affine.apply affine_map<(d0) -> (d0 + 48)>(%arg0)
    %32 = vector.transfer_read %0[%5, %31], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %33 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
    %34 = vector.transfer_read %0[%33, %arg0], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %35 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
    %36 = affine.apply affine_map<(d0) -> (d0 + 16)>(%arg0)
    %37 = vector.transfer_read %0[%35, %36], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %38 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
    %39 = affine.apply affine_map<(d0) -> (d0 + 32)>(%arg0)
    %40 = vector.transfer_read %0[%38, %39], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %41 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
    %42 = affine.apply affine_map<(d0) -> (d0 + 48)>(%arg0)
    %43 = vector.transfer_read %0[%41, %42], %cst {in_bounds = [true, true]} : memref<3456x2048xf16>, vector<16x16xf16>
    %44 = vector.transfer_read %1[%arg0, %6], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %45 = affine.apply affine_map<(d0) -> (d0 + 16)>(%arg0)
    %46 = vector.transfer_read %1[%45, %6], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %47 = affine.apply affine_map<(d0) -> (d0 + 32)>(%arg0)
    %48 = vector.transfer_read %1[%47, %6], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %49 = affine.apply affine_map<(d0) -> (d0 + 48)>(%arg0)
    %50 = vector.transfer_read %1[%49, %6], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %51 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
    %52 = vector.transfer_read %1[%arg0, %51], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %53 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
    %54 = affine.apply affine_map<(d0) -> (d0 + 16)>(%arg0)
    %55 = vector.transfer_read %1[%54, %53], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %56 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
    %57 = affine.apply affine_map<(d0) -> (d0 + 32)>(%arg0)
    %58 = vector.transfer_read %1[%57, %56], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %59 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
    %60 = affine.apply affine_map<(d0) -> (d0 + 48)>(%arg0)
    %61 = vector.transfer_read %1[%60, %59], %cst {in_bounds = [true, true], permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : memref<2048x1024xf16>, vector<16x16xf16>
    %62 = vector.extract_strided_slice %44 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %63 = vector.extract_strided_slice %arg1 {offsets = [0, 0], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %64 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %62, %63 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %65 = vector.extract_strided_slice %44 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %66 = vector.extract_strided_slice %arg1 {offsets = [0, 8], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %67 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %65, %66 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %68 = vector.extract_strided_slice %52 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %69 = vector.extract_strided_slice %arg1 {offsets = [0, 16], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %70 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %68, %69 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %71 = vector.extract_strided_slice %52 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %72 = vector.extract_strided_slice %arg1 {offsets = [0, 24], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %73 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %26, %71, %72 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %74 = vector.extract_strided_slice %44 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %75 = vector.extract_strided_slice %arg1 {offsets = [16, 0], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %76 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %34, %74, %75 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %77 = vector.extract_strided_slice %44 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %78 = vector.extract_strided_slice %arg1 {offsets = [16, 8], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %79 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %34, %77, %78 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %80 = vector.extract_strided_slice %52 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %81 = vector.extract_strided_slice %arg1 {offsets = [16, 16], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %82 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %34, %80, %81 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %83 = vector.extract_strided_slice %52 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %84 = vector.extract_strided_slice %arg1 {offsets = [16, 24], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
    %85 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %34, %83, %84 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %86 = vector.extract_strided_slice %46 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %87 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %28, %86, %64 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %88 = vector.extract_strided_slice %46 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %89 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %28, %88, %67 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %90 = vector.extract_strided_slice %55 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %91 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %28, %90, %70 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %92 = vector.extract_strided_slice %55 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %93 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %28, %92, %73 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %94 = vector.extract_strided_slice %46 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %95 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %37, %94, %76 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %96 = vector.extract_strided_slice %46 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %97 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %37, %96, %79 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %98 = vector.extract_strided_slice %55 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %99 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %37, %98, %82 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %100 = vector.extract_strided_slice %55 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %101 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %37, %100, %85 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %102 = vector.extract_strided_slice %48 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %103 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %30, %102, %87 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %104 = vector.extract_strided_slice %48 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %105 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %30, %104, %89 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %106 = vector.extract_strided_slice %58 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %107 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %30, %106, %91 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %108 = vector.extract_strided_slice %58 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %109 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %30, %108, %93 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %110 = vector.extract_strided_slice %48 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %111 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %110, %95 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %112 = vector.extract_strided_slice %48 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %113 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %112, %97 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %114 = vector.extract_strided_slice %58 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %115 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %114, %99 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %116 = vector.extract_strided_slice %58 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %117 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %40, %116, %101 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %118 = vector.extract_strided_slice %50 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %119 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %32, %118, %103 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %120 = vector.extract_strided_slice %50 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %121 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %32, %120, %105 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %122 = vector.extract_strided_slice %61 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %123 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %32, %122, %107 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %124 = vector.extract_strided_slice %61 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %125 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %32, %124, %109 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %126 = vector.extract_strided_slice %50 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %127 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %126, %111 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %128 = vector.extract_strided_slice %50 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %129 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %128, %113 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %130 = vector.extract_strided_slice %61 {offsets = [0, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %131 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %130, %115 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %132 = vector.extract_strided_slice %61 {offsets = [8, 0], sizes = [8, 16], strides = [1, 1]} : vector<16x16xf16> to vector<8x16xf16>
    %133 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %43, %132, %117 : vector<16x16xf16>, vector<8x16xf16> into vector<16x8xf32>
    %134 = vector.insert_strided_slice %119, %cst_0 {offsets = [0, 0], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %135 = vector.insert_strided_slice %121, %134 {offsets = [0, 8], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %136 = vector.insert_strided_slice %123, %135 {offsets = [0, 16], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %137 = vector.insert_strided_slice %125, %136 {offsets = [0, 24], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %138 = vector.insert_strided_slice %127, %137 {offsets = [16, 0], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %139 = vector.insert_strided_slice %129, %138 {offsets = [16, 8], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %140 = vector.insert_strided_slice %131, %139 {offsets = [16, 16], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    %141 = vector.insert_strided_slice %133, %140 {offsets = [16, 24], strides = [1, 1]} : vector<16x8xf32> into vector<32x32xf32>
    scf.yield %141 : vector<32x32xf32>
  }
  %8 = vector.extract_strided_slice %7 {offsets = [0, 0], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  vector.transfer_write %8, %2[%5, %6] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %9 = vector.extract_strided_slice %7 {offsets = [0, 8], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %10 = affine.apply affine_map<(d0) -> (d0 + 8)>(%6)
  vector.transfer_write %9, %2[%5, %10] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %11 = vector.extract_strided_slice %7 {offsets = [0, 16], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %12 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
  vector.transfer_write %11, %2[%5, %12] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %13 = vector.extract_strided_slice %7 {offsets = [0, 24], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %14 = affine.apply affine_map<(d0) -> (d0 + 24)>(%6)
  vector.transfer_write %13, %2[%5, %14] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %15 = vector.extract_strided_slice %7 {offsets = [16, 0], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %16 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
  vector.transfer_write %15, %2[%16, %6] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %17 = vector.extract_strided_slice %7 {offsets = [16, 8], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %18 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
  %19 = affine.apply affine_map<(d0) -> (d0 + 8)>(%6)
  vector.transfer_write %17, %2[%18, %19] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %20 = vector.extract_strided_slice %7 {offsets = [16, 16], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %21 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
  %22 = affine.apply affine_map<(d0) -> (d0 + 16)>(%6)
  vector.transfer_write %20, %2[%21, %22] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  %23 = vector.extract_strided_slice %7 {offsets = [16, 24], sizes = [16, 8], strides = [1, 1]} : vector<32x32xf32> to vector<16x8xf32>
  %24 = affine.apply affine_map<(d0) -> (d0 + 16)>(%5)
  %25 = affine.apply affine_map<(d0) -> (d0 + 24)>(%6)
  vector.transfer_write %23, %2[%24, %25] {in_bounds = [true, true]} : vector<16x8xf32>, memref<3456x1024xf32>
  return
}
// CHECK-LABEL: func.func @hoist_unrolled_vector_for_mma
// CHECK:         %[[INIT:.+]] = arith.constant dense<0.000000e+00> : vector<16x8xf32>
// CHECK:         %[[RES:.+]]:8 = scf.for {{.+}} iter_args(%[[ARG0:.+]] = %[[INIT]]
// CHECK-NOT:       vector.extract_strided_slice %[[ARG0]]
// vector.insert_strided_slice ops are folded to their consumers.
// CHECK-NOT:     vector.insert_strided_slice
// CHECK:        vector.transfer_write %[[RES]]#0
// CHECK:        vector.transfer_write %[[RES]]#1
// CHECK:        vector.transfer_write %[[RES]]#2
// CHECK:        vector.transfer_write %[[RES]]#3
// CHECK:        vector.transfer_write %[[RES]]#4
// CHECK:        vector.transfer_write %[[RES]]#5
// CHECK:        vector.transfer_write %[[RES]]#6
// CHECK:        vector.transfer_write %[[RES]]#7

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1
    : (!transform.any_op) -> !transform.any_op
  transform.structured.hoist_unrolled_vector_extract_insert_slice %0
    : (!transform.any_op) -> ()
}
