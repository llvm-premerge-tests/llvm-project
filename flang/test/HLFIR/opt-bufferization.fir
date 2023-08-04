// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// simplified example
func.func @simple(%arg: !fir.ref<!fir.array<42xi32>>) {
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return
}
// CHECK-LABEL:   func.func @simple(
// CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           fir.do_loop %[[VAL_6:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_1]] unordered {
// CHECK:             %[[VAL_7:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_10:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_9]] to %[[VAL_10]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// check we support reads that don't alias the transformed array
func.func @read_no_alias(%arg: !fir.ref<!fir.array<42xi32>>, %arg1: !fir.ref<!fir.array<42xi32>>) {
  %c42 = arith.constant 42 : index
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %other:2 = hlfir.declare %arg1(%shape) {uniq_name = "other"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %other_ref = hlfir.designate %other#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %other_val = fir.load %other_ref : !fir.ref<i32>
    %sub = arith.subi %val, %other_val : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return
}
// CHECK-LABEL:   func.func @read_no_alias(
// CHECK-SAME:                             %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>,
// CHECK-SAME:                             %[[VAL_1:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_3]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_6:.*]]:2 = hlfir.declare %[[VAL_1]](%[[VAL_4]]) {uniq_name = "other"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           fir.do_loop %[[VAL_7:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_2]] unordered {
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = hlfir.designate %[[VAL_6]]#0 (%[[VAL_7]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_10:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<i32>
// CHECK:             %[[VAL_12:.*]] = arith.subi %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:             %[[VAL_13:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             hlfir.assign %[[VAL_12]] to %[[VAL_13]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:           }
// CHECK:           return
// CHECK:         }


// check we don't transform when there is another use of the elemental expr
func.func @two_uses(%arg: !fir.ref<!fir.array<42xi32>>) -> i32 {
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  %bad = hlfir.apply %elemental, %c42 : (!hlfir.expr<42xi32>, index) -> i32
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return %bad : i32
}
// CHECK-LABEL:   func.func @two_uses(
// CHECK-SAME:                        %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) -> i32 {
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index):
// CHECK:             %[[VAL_7:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_2]] : i32
// CHECK:             hlfir.yield_element %[[VAL_9]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_10:.*]] to %[[VAL_4]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
// CHECK:           %[[VAL_11:.*]] = hlfir.apply %[[VAL_10]], %[[VAL_1]] : (!hlfir.expr<42xi32>, index) -> i32
// CHECK:           hlfir.destroy %[[VAL_10]] : !hlfir.expr<42xi32>
// CHECK:           return %[[VAL_11]] : i32
// CHECK:         }

// two dimensional array
func.func @two_dimensional(%arg: !fir.ref<!fir.array<42x42xi32>>) {
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42, %c42 : (index, index) -> !fir.shape<2>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42x42xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<42x42xi32>>, !fir.ref<!fir.array<42x42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<42x42xi32> {
  ^bb0(%i: index, %j: index):
    %ref = hlfir.designate %array#0 (%i, %j) : (!fir.ref<!fir.array<42x42xi32>>, index, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42x42xi32>, !fir.ref<!fir.array<42x42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42x42xi32>
  return
}
// CHECK-LABEL:   func.func @two_dimensional(
// CHECK-SAME:                               %[[VAL_0:.*]]: !fir.ref<!fir.array<42x42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]], %[[VAL_2]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42x42xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<42x42xi32>>, !fir.ref<!fir.array<42x42xi32>>)
// CHECK:           fir.do_loop %[[VAL_6:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_1]] unordered {
// CHECK:             fir.do_loop %[[VAL_7:.*]] = %[[VAL_1]] to %[[VAL_2]] step %[[VAL_1]] unordered {
// CHECK:               %[[VAL_8:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]], %[[VAL_6]])  : (!fir.ref<!fir.array<42x42xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:               %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_3]] : i32
// CHECK:               %[[VAL_11:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]], %[[VAL_6]])  : (!fir.ref<!fir.array<42x42xi32>>, index, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_10]] to %[[VAL_11]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// don't transform when elements are accessessed out of order (e.g. transposed)
func.func @transposed(%arg: !fir.ref<!fir.array<42x42xi32>>) {
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42, %c42 : (index, index) -> !fir.shape<2>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42x42xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<42x42xi32>>, !fir.ref<!fir.array<42x42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<2>) -> !hlfir.expr<42x42xi32> {
  ^bb0(%i: index, %j: index):
    %ref = hlfir.designate %array#0 (%j, %i) : (!fir.ref<!fir.array<42x42xi32>>, index, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42x42xi32>, !fir.ref<!fir.array<42x42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42x42xi32>
  return
}
// CHECK-LABEL:   func.func @transposed(
// CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<!fir.array<42x42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]], %[[VAL_1]] : (index, index) -> !fir.shape<2>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42x42xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<42x42xi32>>, !fir.ref<!fir.array<42x42xi32>>)
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<2>) -> !hlfir.expr<42x42xi32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index, %[[VAL_7:.*]]: index):
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_7]], %[[VAL_6]])  : (!fir.ref<!fir.array<42x42xi32>>, index, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_2]] : i32
// CHECK:             hlfir.yield_element %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_11:.*]] to %[[VAL_4]]#0 : !hlfir.expr<42x42xi32>, !fir.ref<!fir.array<42x42xi32>>
// CHECK:           hlfir.destroy %[[VAL_11]] : !hlfir.expr<42x42xi32>
// CHECK:           return
// CHECK:         }

// don't transform when there's an operation with unknown effects
func.func @unknown(%arg: !fir.ref<!fir.array<42xi32>>) {
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    %res = fir.call @impure(%sub) : (i32) -> i32
    hlfir.yield_element %res : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return
}
// CHECK-LABEL:   func.func @unknown(
// CHECK-SAME:                       %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_3]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_3]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index):
// CHECK:             %[[VAL_7:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_10:.*]] = fir.call @impure(%[[VAL_9]]) : (i32) -> i32
// CHECK:             hlfir.yield_element %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_11:.*]] to %[[VAL_4]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
// CHECK:           hlfir.destroy %[[VAL_11]] : !hlfir.expr<42xi32>
// CHECK:           return
// CHECK:         }

// don't transform when there's an operation with write effects
func.func @write(%arg: !fir.ref<!fir.array<42xi32>>, %arg1: !fir.ref<!fir.array<42xi32>>) {
  %alloc = fir.alloca i32
  %c42 = arith.constant 42 : index
  %c1_i32 = arith.constant 1 : i32
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %array2:2 = hlfir.declare %arg1(%shape) {uniq_name = "array2"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    hlfir.assign %array2#0 to %array#0 : !fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %sub = arith.subi %val, %c1_i32 : i32
    fir.store %sub to %alloc : !fir.ref<i32>
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return
}
// CHECK-LABEL:   func.func @write(
// CHECK-SAME:                     %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>,
// CHECK-SAME:                     %[[ARG_1:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_3:.*]] = fir.alloca i32
// CHECK:           %[[VAL_4:.*]] = fir.shape %[[VAL_2]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_4]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_5B:.*]]:2 = hlfir.declare %[[ARG_1]](%[[VAL_4]]) {uniq_name = "array2"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_6:.*]] = hlfir.elemental %[[VAL_4]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
// CHECK:           ^bb0(%[[VAL_7:.*]]: index):
// CHECK:             hlfir.assign %[[VAL_5B]]#0 to %[[VAL_5]]#0 : !fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_5]]#0 (%[[VAL_7]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_1]] : i32
// CHECK:             fir.store %[[VAL_10]] to %[[VAL_3]] : !fir.ref<i32>
// CHECK:             hlfir.yield_element %[[VAL_10]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_11:.*]] to %[[VAL_5]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
// CHECK:           hlfir.destroy %[[VAL_11]] : !hlfir.expr<42xi32>
// CHECK:           return
// CHECK:         }

// don't transform when there is an aliasing read
func.func @readAlias(%arg: !fir.ref<!fir.array<42xi32>>) {
  %c42 = arith.constant 42 : index
  %shape = fir.shape %c42 : (index) -> !fir.shape<1>
  %array:2 = hlfir.declare %arg(%shape) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %arrayDup:2 = hlfir.declare %arg(%shape) {uniq_name = "arrayDup"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
  %elemental = hlfir.elemental %shape unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
  ^bb0(%i: index):
    %ref = hlfir.designate %array#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %refDup = hlfir.designate %arrayDup#0 (%i) : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
    %val = fir.load %ref : !fir.ref<i32>
    %valDup = fir.load %refDup : !fir.ref<i32>
    %sub = arith.subi %val, %valDup : i32
    hlfir.yield_element %sub : i32
  }
  hlfir.assign %elemental to %array#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
  hlfir.destroy %elemental : !hlfir.expr<42xi32>
  return
}
// CHECK-LABEL:   func.func @readAlias(
// CHECK-SAME:                         %[[VAL_0:.*]]: !fir.ref<!fir.array<42xi32>>) {
// CHECK:           %[[VAL_1:.*]] = arith.constant 42 : index
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_2]]) {uniq_name = "array"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_0]](%[[VAL_2]]) {uniq_name = "arrayDup"} : (!fir.ref<!fir.array<42xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<42xi32>>, !fir.ref<!fir.array<42xi32>>)
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_2]] unordered : (!fir.shape<1>) -> !hlfir.expr<42xi32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index):
// CHECK:             %[[VAL_7:.*]] = hlfir.designate %[[VAL_3]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_8:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<42xi32>>, index) -> !fir.ref<i32>
// CHECK:             %[[VAL_9:.*]] = fir.load %[[VAL_7]] : !fir.ref<i32>
// CHECK:             %[[VAL_10:.*]] = fir.load %[[VAL_8]] : !fir.ref<i32>
// CHECK:             %[[VAL_11:.*]] = arith.subi %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:             hlfir.yield_element %[[VAL_11]] : i32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_12:.*]] to %[[VAL_3]]#0 : !hlfir.expr<42xi32>, !fir.ref<!fir.array<42xi32>>
// CHECK:           hlfir.destroy %[[VAL_12]] : !hlfir.expr<42xi32>
// CHECK:           return
// CHECK:         }

fir.global @_QMmEblock : !fir.array<9x9x9xi32> {
  %0 = fir.undefined !fir.array<9x9x9xi32>
  fir.has_value %0 : !fir.array<9x9x9xi32>
}
fir.global @_QMmECr constant : i32 {
  %c9_i32 = arith.constant 9 : i32
  fir.has_value %c9_i32 : i32
}

// does it work for the intended case?
func.func @_QMmPrepro(%arg0: !fir.ref<i32> {fir.bindc_name = "imin"}, %arg1: !fir.ref<i32> {fir.bindc_name = "imax"}, %arg2: !fir.ref<i32> {fir.bindc_name = "row"}) {
  %c10_i32 = arith.constant 10 : i32
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c9 = arith.constant 9 : index
  %0 = fir.address_of(@_QMmEblock) : !fir.ref<!fir.array<9x9x9xi32>>
  %1 = fir.shape %c9, %c9, %c9 : (index, index, index) -> !fir.shape<3>
  %2:2 = hlfir.declare %0(%1) {uniq_name = "_QMmEblock"} : (!fir.ref<!fir.array<9x9x9xi32>>, !fir.shape<3>) -> (!fir.ref<!fir.array<9x9x9xi32>>, !fir.ref<!fir.array<9x9x9xi32>>)
  %3 = fir.address_of(@_QMmECr) : !fir.ref<i32>
  %4:2 = hlfir.declare %3 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMmECr"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca i32 {bindc_name = "i1", uniq_name = "_QMmFreproEi1"}
  %6:2 = hlfir.declare %5 {uniq_name = "_QMmFreproEi1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %7:2 = hlfir.declare %arg1 {uniq_name = "_QMmFreproEimax"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %8:2 = hlfir.declare %arg0 {uniq_name = "_QMmFreproEimin"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %9:2 = hlfir.declare %arg2 {uniq_name = "_QMmFreproErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %10 = fir.load %8#0 : !fir.ref<i32>
  %11 = fir.convert %10 : (i32) -> index
  %12 = fir.load %7#0 : !fir.ref<i32>
  %13 = fir.convert %12 : (i32) -> index
  %14 = fir.convert %11 : (index) -> i32
  %15:2 = fir.do_loop %arg3 = %11 to %13 step %c1 iter_args(%arg4 = %14) -> (index, i32) {
    fir.store %arg4 to %6#1 : !fir.ref<i32>
    %16 = fir.load %9#0 : !fir.ref<i32>
    %17 = fir.convert %16 : (i32) -> i64
    %18 = fir.load %6#0 : !fir.ref<i32>
    %19 = fir.convert %18 : (i32) -> i64
    %20 = fir.shape %c8 : (index) -> !fir.shape<1>
    %21 = hlfir.designate %2#0 (%17, %c2:%c9:%c1, %19)  shape %20 : (!fir.ref<!fir.array<9x9x9xi32>>, i64, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<8xi32>>
    %22 = hlfir.elemental %20 unordered : (!fir.shape<1>) -> !hlfir.expr<8xi32> {
    ^bb0(%arg5: index):
      %27 = hlfir.designate %21 (%arg5)  : (!fir.box<!fir.array<8xi32>>, index) -> !fir.ref<i32>
      %28 = fir.load %27 : !fir.ref<i32>
      %29 = arith.subi %28, %c10_i32 : i32
      hlfir.yield_element %29 : i32
    }
    hlfir.assign %22 to %21 : !hlfir.expr<8xi32>, !fir.box<!fir.array<8xi32>>
    hlfir.destroy %22 : !hlfir.expr<8xi32>
    %23 = arith.addi %arg3, %c1 : index
    %24 = fir.convert %c1 : (index) -> i32
    %25 = fir.load %6#1 : !fir.ref<i32>
    %26 = arith.addi %25, %24 : i32
    fir.result %23, %26 : index, i32
  }
  fir.store %15#1 to %6#1 : !fir.ref<i32>
  return
}
// CHECK-LABEL:   func.func @_QMmPrepro(
// CHECK-SAME:                          %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "imin"},
// CHECK-SAME:                          %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "imax"},
// CHECK-SAME:                          %[[VAL_2:.*]]: !fir.ref<i32> {fir.bindc_name = "row"}) {
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 9 : index
// CHECK:           %[[VAL_8:.*]] = fir.address_of(@_QMmEblock) : !fir.ref<!fir.array<9x9x9xi32>>
// CHECK:           %[[VAL_9:.*]] = fir.shape %[[VAL_7]], %[[VAL_7]], %[[VAL_7]] : (index, index, index) -> !fir.shape<3>
// CHECK:           %[[VAL_10:.*]]:2 = hlfir.declare %[[VAL_8]](%[[VAL_9]]) {uniq_name = "_QMmEblock"} : (!fir.ref<!fir.array<9x9x9xi32>>, !fir.shape<3>) -> (!fir.ref<!fir.array<9x9x9xi32>>, !fir.ref<!fir.array<9x9x9xi32>>)
// CHECK:           %[[VAL_11:.*]] = fir.address_of(@_QMmECr) : !fir.ref<i32>
// CHECK:           %[[VAL_12:.*]]:2 = hlfir.declare %[[VAL_11]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QMmECr"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_13:.*]] = fir.alloca i32 {bindc_name = "i1", uniq_name = "_QMmFreproEi1"}
// CHECK:           %[[VAL_14:.*]]:2 = hlfir.declare %[[VAL_13]] {uniq_name = "_QMmFreproEi1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_15:.*]]:2 = hlfir.declare %[[VAL_1]] {uniq_name = "_QMmFreproEimax"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_16:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QMmFreproEimin"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_17:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QMmFreproErow"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
// CHECK:           %[[VAL_18:.*]] = fir.load %[[VAL_16]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_19:.*]] = fir.convert %[[VAL_18]] : (i32) -> index
// CHECK:           %[[VAL_20:.*]] = fir.load %[[VAL_15]]#0 : !fir.ref<i32>
// CHECK:           %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (i32) -> index
// CHECK:           %[[VAL_22:.*]] = fir.convert %[[VAL_19]] : (index) -> i32
// CHECK:           %[[VAL_23:.*]]:2 = fir.do_loop %[[VAL_24:.*]] = %[[VAL_19]] to %[[VAL_21]] step %[[VAL_6]] iter_args(%[[VAL_25:.*]] = %[[VAL_22]]) -> (index, i32) {
// CHECK:             fir.store %[[VAL_25]] to %[[VAL_14]]#1 : !fir.ref<i32>
// CHECK:             %[[VAL_26:.*]] = fir.load %[[VAL_17]]#0 : !fir.ref<i32>
// CHECK:             %[[VAL_27:.*]] = fir.convert %[[VAL_26]] : (i32) -> i64
// CHECK:             %[[VAL_28:.*]] = fir.load %[[VAL_14]]#0 : !fir.ref<i32>
// CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_28]] : (i32) -> i64
// CHECK:             %[[VAL_30:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:             %[[VAL_31:.*]] = hlfir.designate %[[VAL_10]]#0 (%[[VAL_27]], %[[VAL_5]]:%[[VAL_7]]:%[[VAL_6]], %[[VAL_29]])  shape %[[VAL_30]] : (!fir.ref<!fir.array<9x9x9xi32>>, i64, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<8xi32>>
// CHECK:             fir.do_loop %[[VAL_32:.*]] = %[[VAL_6]] to %[[VAL_4]] step %[[VAL_6]] unordered {
// CHECK:               %[[VAL_33:.*]] = hlfir.designate %[[VAL_31]] (%[[VAL_32]])  : (!fir.box<!fir.array<8xi32>>, index) -> !fir.ref<i32>
// CHECK:               %[[VAL_34:.*]] = fir.load %[[VAL_33]] : !fir.ref<i32>
// CHECK:               %[[VAL_35:.*]] = arith.subi %[[VAL_34]], %[[VAL_3]] : i32
// CHECK:               %[[VAL_36:.*]] = hlfir.designate %[[VAL_31]] (%[[VAL_32]])  : (!fir.box<!fir.array<8xi32>>, index) -> !fir.ref<i32>
// CHECK:               hlfir.assign %[[VAL_35]] to %[[VAL_36]] temporary_lhs : i32, !fir.ref<i32>
// CHECK:             }
// CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_24]], %[[VAL_6]] : index
// CHECK:             %[[VAL_38:.*]] = fir.convert %[[VAL_6]] : (index) -> i32
// CHECK:             %[[VAL_39:.*]] = fir.load %[[VAL_14]]#1 : !fir.ref<i32>
// CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_38]] : i32
// CHECK:             fir.result %[[VAL_37]], %[[VAL_40]] : index, i32
// CHECK:           }
// CHECK:           fir.store %[[VAL_41:.*]]#1 to %[[VAL_14]]#1 : !fir.ref<i32>
// CHECK:           return
// CHECK:         }
