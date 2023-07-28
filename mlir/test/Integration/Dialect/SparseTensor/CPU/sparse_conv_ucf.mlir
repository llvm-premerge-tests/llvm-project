// DEFINE: %{option} = "enable-runtime-library=false"
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = TENSOR0="%mlir_src_dir/test/Integration/data/ResNet50/0.8/bottleneck_1_block_group1_1_1.tns" \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}

!Filename = !llvm.ptr<i8>

#SS = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed" ]
}>

#SSSD = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed", "compressed", "dense" ]
}>

module {
  func.func private @getTensorFilename(index) -> (!Filename)

  func.func @alloc_4d_filled_f32(%s1 : index, %s2 : index, %s3 : index, %s4 : index, %f : f32) -> tensor<?x?x?x?xf32> {
    %buf = bufferization.alloc_tensor(%s1, %s2, %s3, %s4) : tensor<?x?x?x?xf32>
    %ret = linalg.fill ins(%f : f32) outs(%buf : tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %ret : tensor<?x?x?x?xf32>
  }

  func.func @conv_2d(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?x?x?x?xf32, #SSSD>, %arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    %ret = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>,
                                      strides = dense<1> : tensor<2xi64>}
      ins (%arg0, %arg1: tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32, #SSSD>)
      outs (%arg2: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %ret : tensor<?x?x?x?xf32>
  }

  func.func @runBenchmark(%benchmark : index, %N : index, %H : index, %W : index, %R : index, %S : index,
  %STR : index, %PAD : index, %C : index, %M : index) {
    // Compute output shape
    %i1 = arith.constant 1 : index
    %i2 = arith.constant 2 : index
    %Pad2 = arith.muli %PAD, %i2 : index
    %HPad = arith.addi %H, %Pad2 : index
    %WPad = arith.addi %W, %Pad2 : index
    %HPadMinusR = arith.subi %HPad, %R : index
    %HPadMinusRDivStr = arith.divui  %HPadMinusR, %STR : index
    %P = arith.addi %HPadMinusRDivStr, %i1 : index
    %WPadMinusS = arith.subi %WPad, %S : index
    %WPadMinusSDivStr = arith.divui %WPadMinusS, %STR : index
    %Q = arith.addi %WPadMinusSDivStr, %i1: index

    // Construct filter of size MxCxRxS.
    %fileName = call @getTensorFilename(%benchmark) : (index) -> (!Filename)
    %readFile = sparse_tensor.new %fileName : !Filename to tensor<?x?xf32, #SS>
    // TODO(anlunx): Use tensor.reshape insteand of expand_shape.
    %expanded = tensor.expand_shape %readFile [[0], [1, 2, 3]] : tensor<?x?xf32, #SS> into tensor<?x?x1x1xf32, #SSSD>
    %filter = sparse_tensor.convert %expanded : tensor<?x?x1x1xf32, #SSSD> to tensor<?x?x?x?xf32, #SSSD>
    %noe = sparse_tensor.number_of_entries %filter : tensor<?x?x?x?xf32, #SSSD>
    vector.print %noe : index

    // Construct input.
    %input_elem = arith.constant 1.0 : f32
    %input = call @alloc_4d_filled_f32(%N, %C, %H, %W, %input_elem) :(index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)
    // Construct output.
    %output_elem = arith.constant 0.0 : f32
    %output = call @alloc_4d_filled_f32(%N, %M, %P, %Q, %output_elem) :(index, index, index, index, f32) -> (tensor<?x?x?x?xf32>)

    %ret = call @conv_2d(%input, %filter, %output) : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32, #SSSD>, tensor<?x?x?x?xf32>) -> (tensor<?x?x?x?xf32>)
    %i0 = arith.constant 0 : index
    %df = arith.constant -1.0 : f32
    %v1 = vector.transfer_read %ret[%i0, %i0, %i0, %i0], %df: tensor<?x?x?x?xf32>, vector<10xf32>
    vector.print %v1 : vector<10xf32>

    bufferization.dealloc_tensor %readFile: tensor<?x?xf32, #SS>
    bufferization.dealloc_tensor %filter: tensor<?x?x?x?xf32, #SSSD>
    bufferization.dealloc_tensor %input : tensor<?x?x?x?xf32>
    bufferization.dealloc_tensor %output : tensor<?x?x?x?xf32>
    return
  }

  func.func @entry() {
    %c0 = arith.constant 0 : index
    %N = arith.constant 1 : index
    %H = arith.constant 56 : index
    %W = arith.constant 56 : index
    %R = arith.constant 1 : index
    %S = arith.constant 1 : index
    %STR = arith.constant 1 : index
    %PAD = arith.constant 0 : index
    %C = arith.constant 256 : index
    %M = arith.constant 64 : index

    // Check number_of_entries in sparse filter.
    // CHECK: 3279 
    // Check output
    // CHECK: ( 74, 74, 74, 74, 74, 74, 74, 74, 74, 74 )
    call @runBenchmark(%c0, %N, %H, %W, %R, %S, %STR, %PAD, %C, %M) :
      (index, index, index, index, index, index, index, index , index, index) -> ()

    return
  }
}