// RUN: mlir-opt %s --test-transform-dialect-interpreter | FileCheck %s

/// This tests that shape casts of scalable vectors (with one trailing scalable dim)
/// can be correctly lowered to vector.scalable.insert/extract.

// CHECK-LABEL: i32_vector_shape_cast_trailing_3d_scalable_to_1d
// CHECK-SAME: %[[arg0:.*]]: vector<2x1x[4]xi32>
func.func @i32_vector_shape_cast_trailing_3d_scalable_to_1d(%arg0: vector<2x1x[4]xi32>) -> vector<[8]xi32>
{
  // CHECK-NEXT: %[[cst:.*]] = arith.constant dense<0> : vector<[8]xi32>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.extract %[[arg0]][0, 0] : vector<2x1x[4]xi32>
  // CHECK-NEXT: %[[res0:.*]] = vector.scalable.insert %[[subvec0]], %[[cst]][0] : vector<[4]xi32> into vector<[8]xi32>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.extract %[[arg0]][1, 0] : vector<2x1x[4]xi32>
  // CHECK-NEXT: %[[res1:.*]] = vector.scalable.insert %[[subvec1]], %[[res0]][4] : vector<[4]xi32> into vector<[8]xi32>
  %flat = vector.shape_cast %arg0 : vector<2x1x[4]xi32> to vector<[8]xi32>
  // CHECK-NEXT: return %[[res1]] : vector<[8]xi32>
  return %flat : vector<[8]xi32>
}

// -----

// CHECK-LABEL: i32_vector_shape_cast_1d_scalable_to_3d
// CHECK-SAME: %[[arg0:.*]]: vector<[8]xi32>
func.func @i32_vector_shape_cast_1d_scalable_to_3d(%arg0: vector<[8]xi32>) -> vector<2x1x[4]xi32> {
  // CHECK-NEXT: %[[cst:.*]] = arith.constant dense<0> : vector<2x1x[4]xi32>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.scalable.extract %[[arg0]][0] : vector<[4]xi32> from vector<[8]xi32>
  // CHECK-NEXT: %[[res0:.*]] = vector.insert %[[subvec0]], %[[cst]] [0, 0] : vector<[4]xi32> into vector<2x1x[4]xi32>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.scalable.extract %[[arg0]][4] : vector<[4]xi32> from vector<[8]xi32>
  // CHECK-NEXT: %[[res1:.*]] = vector.insert %[[subvec1]], %[[res0]] [1, 0] : vector<[4]xi32> into vector<2x1x[4]xi32>
  %unflat = vector.shape_cast %arg0 : vector<[8]xi32> to vector<2x1x[4]xi32>
  // CHECK-NEXT: return %[[res1]] : vector<2x1x[4]xi32>
  return %unflat : vector<2x1x[4]xi32>
}

// -----

// CHECK-LABEL: i8_vector_shape_cast_trailing_2d_scalable_to_1d
// CHECK-SAME: %[[arg0:.*]]: vector<4x[8]xi8>
func.func @i8_vector_shape_cast_trailing_2d_scalable_to_1d(%arg0: vector<4x[8]xi8>) -> vector<[32]xi8> {
  // CHECK-NEXT: %[[cst:.*]] = arith.constant dense<0> : vector<[32]xi8>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.extract %[[arg0]][0] : vector<4x[8]xi8>
  // CHECK-NEXT: %[[res0:.*]] = vector.scalable.insert %[[subvec0]], %[[cst]][0] : vector<[8]xi8> into vector<[32]xi8>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.extract %[[arg0]][1] : vector<4x[8]xi8>
  // CHECK-NEXT: %[[res1:.*]] = vector.scalable.insert %[[subvec1]], %[[res0]][8] : vector<[8]xi8> into vector<[32]xi8>
  // CHECK-NEXT: %[[subvec2:.*]] = vector.extract %[[arg0]][2] : vector<4x[8]xi8>
  // CHECK-NEXT: %[[res2:.*]] = vector.scalable.insert %[[subvec2]], %[[res1]][16] : vector<[8]xi8> into vector<[32]xi8>
  // CHECK-NEXT: %[[subvec3:.*]] = vector.extract %[[arg0]][3] : vector<4x[8]xi8>
  // CHECK-NEXT: %[[res3:.*]] = vector.scalable.insert %[[subvec3]], %[[res2]][24] : vector<[8]xi8> into vector<[32]xi8>
  %flat = vector.shape_cast %arg0 : vector<4x[8]xi8> to vector<[32]xi8>
  // CHECK-NEXT: return %[[res3]] : vector<[32]xi8>
  return %flat : vector<[32]xi8>
}

// -----

// CHECK-LABEL: i8_vector_shape_cast_1d_scalable_to_2d
// CHECK-SAME: %[[arg0:.*]]: vector<[32]xi8>
func.func @i8_vector_shape_cast_1d_scalable_to_2d(%arg0: vector<[32]xi8>) -> vector<4x[8]xi8> {
  // CHECK-NEXT: %[[cst:.*]] = arith.constant dense<0> : vector<4x[8]xi8>
  // CHECK-NEXT: %[[subvec0:.*]] = vector.scalable.extract %arg0[0] : vector<[8]xi8> from vector<[32]xi8>
  // CHECK-NEXT: %[[res0:.*]] = vector.insert %[[subvec0]], %[[cst]] [0] : vector<[8]xi8> into vector<4x[8]xi8>
  // CHECK-NEXT: %[[subvec1:.*]] = vector.scalable.extract %[[arg0]][8] : vector<[8]xi8> from vector<[32]xi8>
  // CHECK-NEXT: %[[res1:.*]] = vector.insert %[[subvec1]], %[[res0]] [1] : vector<[8]xi8> into vector<4x[8]xi8>
  // CHECK-NEXT: %[[subvec2:.*]] = vector.scalable.extract %[[arg0]][16] : vector<[8]xi8> from vector<[32]xi8>
  // CHECK-NEXT: %[[res2:.*]] = vector.insert %[[subvec2]], %[[res1]] [2] : vector<[8]xi8> into vector<4x[8]xi8>
  // CHECK-NEXT: %[[subvec3:.*]] = vector.scalable.extract %[[arg0]][24] : vector<[8]xi8> from vector<[32]xi8>
  // CHECK-NEXT: %[[res3:.*]] = vector.insert %[[subvec3]], %[[res2]] [3] : vector<[8]xi8> into vector<4x[8]xi8>
  %unflat = vector.shape_cast %arg0 : vector<[32]xi8> to vector<4x[8]xi8>
  // CHECK-NEXT: return %[[res3]] : vector<4x[8]xi8>
  return %unflat : vector<4x[8]xi8>
}

transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %f = transform.structured.match ops{["func.func"]} in %module_op
    : (!transform.any_op) -> !transform.any_op

  transform.apply_patterns to %f {
    transform.apply_patterns.vector.lower_shape_cast
  } : !transform.any_op
}
