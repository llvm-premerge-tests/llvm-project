// RUN: fir-opt %s --test-side-effects --verify-diagnostics

func.func @concat(%arg0: !fir.ref<!fir.char<1,10>>, %arg1: !fir.ref<!fir.char<1, 20>>) {
// expected-remark@+1 {{operation has no memory effects}}
  %c30 = arith.constant 30 : index
// expected-remark@+2 {{found an instance of 'allocate' on resource '<Default>'}}
// expected-remark@+1 {{found an instance of 'read' on a value, on resource '<Default>'}}
  %0 = hlfir.concat %arg0, %arg1 len %c30 : (!fir.ref<!fir.char<1,10>>, !fir.ref<!fir.char<1,20>>, index) -> (!hlfir.expr<!fir.char<1,30>>)
// expected-remark@+1 {{operation has no memory effects}}
  return
}

// TODO: testing for other operations
