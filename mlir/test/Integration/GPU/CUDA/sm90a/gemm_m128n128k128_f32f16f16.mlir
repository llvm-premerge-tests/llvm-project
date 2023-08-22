// RUN: mlir-opt %s --convert-nvgpu-to-nvvm -gpu-kernel-outlining \
// RUN:     -convert-scf-to-cf -convert-nvvm-to-llvm \
// RUN:     -convert-vector-to-llvm \
// RUN:     -convert-math-to-llvm \
// RUN:     -expand-strided-metadata \
// RUN:     -lower-affine \
// RUN:     -convert-index-to-llvm=index-bitwidth=32 \
// RUN:     -convert-arith-to-llvm \
// RUN:     -finalize-memref-to-llvm \
// RUN:     -convert-func-to-llvm \
// RUN:     -canonicalize \
// RUN: | mlir-opt -pass-pipeline='builtin.module(gpu.module(strip-debuginfo,convert-gpu-to-nvvm,convert-nvgpu-to-nvvm{use-opaque-pointers=1},lower-affine,convert-scf-to-cf,convert-vector-to-llvm,convert-math-to-llvm,expand-strided-metadata,lower-affine,convert-index-to-llvm{index-bitwidth=32},convert-arith-to-llvm,reconcile-unrealized-casts,gpu-to-cubin{chip=sm_90 features=+ptx80 dump-ptx}))' \
// RUN: 2&>1 | FileCheck %s --check-prefixes=CHECK-PTX

// CHECK-PTX: mbarrier.init.shared.b64
// CHECK-PTX: mbarrier.arrive.expect_tx.shared.b64
// CHECK-PTX: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
// CHECK-PTX: cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
// CHECK-PTX: mbarrier.arrive.expect_tx.shared.b64
// CHECK-PTX: mbarrier.try_wait.parity.shared.b64
// CHECK-PTX: wgmma.fence.sync.aligned;
// CHECK-PTX: wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
// CHECK-PTX: wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
// CHECK-PTX: wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
// CHECK-PTX: wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
// CHECK-PTX: wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
// CHECK-PTX: wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
// CHECK-PTX: wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
// CHECK-PTX: wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
// CHECK-PTX: wgmma.commit_group.sync.aligned
// CHECK-PTX: wgmma.wait_group.sync.aligned 1


// # GEMM Sequential
// ------------------------------------------
// for(i...128)
//  for(j...128)
//   for(k...128)
//     D = A * B
// # Parallel H100 GEMM (1 CTA, 1 warpgroup(128 threads))
// ------------------------------------------
// wgmma.mma_async shape<m = 64, n = 128, k = 16>
// wgmma.mma_async shape<m = 64, n = 128, k = 16>
// wgmma.mma_async shape<m = 64, n = 128, k = 16>
// wgmma.mma_async shape<m = 64, n = 128, k = 16>
// wgmma.mma_async shape<m = 64, n = 128, k = 16>
// wgmma.mma_async shape<m = 64, n = 128, k = 16>
// wgmma.mma_async shape<m = 64, n = 128, k = 16>
// wgmma.mma_async shape<m = 64, n = 128, k = 16>
  
!barrierType = !nvgpu.mbarrier.barrier<memorySpace = #gpu.address_space<workgroup>>
!tokenType = !nvgpu.mbarrier.token

// TMA device descriptor
!lhs = memref<128x64xf16>
!rhs = memref<64x128xf16>

!shmemlhs = memref<128x64xf16,3>
!shmemrhs = memref<64x128xf16,3>

!lhsTensorMap = !nvgpu.tensormap.descriptor<tensor = !shmemlhs, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>
!rhsTensorMap = !nvgpu.tensormap.descriptor<tensor = !shmemrhs, swizzle = swizzle_128b, l2promo=none, oob=zero, interleave=none>

!accMatrix = !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, 
  f32, f32, f32, f32, f32, f32)>
  
module @mymod{
memref.global "private" @bufferLhsGlobal : !shmemlhs
memref.global "private" @bufferRhsGlobal : !shmemrhs

func.func @main() {
  %c64 = arith.constant 64 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %f2 = arith.constant 2.0 : f16
  %f3 = arith.constant 3.0 : f16

  // Step 1. Allocate matrices by managed memory for sake of simplicity.
  %lhs = memref.alloc() : !lhs
  %rhs = memref.alloc() : !rhs
  
  // Step 2. Intialize the Input matrix with ones.
  scf.for %i = %c0 to %c64 step %c1 {
    scf.for %j = %c0 to %c128 step %c1 {
      memref.store %f3, %rhs[%i, %j] : !rhs
      memref.store %f2, %lhs[%j, %i] : !lhs
    }
  }
  
  // Step 3. Copy data to the device.
  %0 = gpu.wait async
  %memref, %asyncToken = gpu.alloc async [%0] () : !lhs
  %memref_1, %asyncToken_2 = gpu.alloc async [%0] () : !rhs
  %1 = gpu.memcpy async [%0] %memref, %lhs : !lhs, !lhs
  %2 = gpu.memcpy async [%0] %memref_1, %rhs : !rhs, !rhs

  // Step 4. Create 2 TMA Descriptors for input matrices
  %lhs_unranked = memref.cast %memref :!lhs  to memref<*xf16>
  %rhs_unranked = memref.cast %memref_1 :!rhs  to memref<*xf16>

  %lhsTensorMap = nvgpu.tma.create.descriptor %lhs_unranked box[%c128, %c64] : memref<*xf16> -> !lhsTensorMap
  %rhsTensorMap = nvgpu.tma.create.descriptor %rhs_unranked box[%c128, %c64] : memref<*xf16> -> !rhsTensorMap

  // Step 5. Launch the kernel.
  gpu.launch blocks(%bx, %by, %bz) in (%grid_x = %c1, %grid_y = %c1, %grid_z = %c1)
           threads(%tx, %ty, %tz) in (%block_x = %c128, %block_y = %c1, %block_z = %c1) {
      %num_threads = gpu.block_dim x
      %ic0 = arith.constant 0 : index
      %ic7 = arith.constant 7 : index
      %ic1 = arith.constant 1 : index
      %ic2 = arith.constant 2 : index
      %ic45 = arith.constant 45 : index
      %ic63 = arith.constant 63 : index
      %ic15 = arith.constant 15 : index
      %ic127 = arith.constant 127 : index
      %tidx = gpu.thread_id x

      // Step 1. Get shared memory pointers. This could be dynamic shared memory, but for now I use static
      %lhsShmem = memref.get_global @bufferLhsGlobal : !shmemlhs
      %rhsShmem = memref.get_global @bufferRhsGlobal : !shmemrhs

      // Step 2. Create a barrier type. This is i64 value in shared memory.
      %barrier = nvgpu.mbarrier.create -> !barrierType

      // Step 3. Initialize the barrier
      nvgpu.mbarrier.init %barrier, %num_threads : !barrierType
      gpu.barrier
      // Step 3.1 threadIdx.x == 0 does TMA request
      %cnd = arith.cmpi eq, %tidx, %ic0 : index    
      scf.if %cnd {
        %x1 = memref.dim %lhsShmem, %c0 : !shmemlhs
        %x2 = memref.dim %lhsShmem, %c1 : !shmemlhs
        %x3 = memref.dim %rhsShmem, %c0 : !shmemrhs
        %x4 = memref.dim %rhsShmem, %c1 : !shmemrhs
        %x5 = arith.muli %x1, %x2 : index
        %x6 = arith.muli %x3, %x4 : index
        %x7 = arith.addi %x5, %x6 : index
        %txcount = arith.muli %x7, %ic2 : index
        gpu.printf "[GPU] TMA SIZE %d\n" %txcount : index

        %lhs0 = memref.load %lhsShmem[%ic0, %ic0] : !shmemlhs
        %rhs0 = memref.load %rhsShmem[%ic0, %ic0] : !shmemrhs
        %lhs032 = arith.extf %lhs0: f16 to f32
        %rhs032 = arith.extf %rhs0: f16 to f32
        gpu.printf "[GPU] Before TMA Shmem lhs[0][0] \t %f\n" %lhs032 : f32
        gpu.printf "[GPU] Before TMA Shmem rhs[0][0] \t %f\n" %rhs032 : f32
        nvgpu.tma.async.load %lhsTensorMap[%ic0, %ic0], %barrier to %lhsShmem : !lhsTensorMap, !barrierType -> !shmemlhs
        nvgpu.tma.async.load %rhsTensorMap[%ic0, %ic0], %barrier to %rhsShmem : !rhsTensorMap, !barrierType -> !shmemrhs
        nvgpu.mbarrier.arrive.expect_tx %barrier, %txcount : !barrierType
        scf.yield 
      } else {
        nvgpu.mbarrier.arrive.expect_tx %barrier, %ic0 : !barrierType
        scf.yield 
      }
      
      %phase = arith.constant 0 : index
      %ticks = arith.constant 10000000 : index
      nvgpu.mbarrier.try_wait.parity %barrier, %phase, %ticks : !barrierType
      
      %descA = nvgpu.wgmma.generate.descriptor %lhsShmem, %lhsTensorMap : !shmemlhs, !lhsTensorMap
      %descB = nvgpu.wgmma.generate.descriptor %rhsShmem, %rhsTensorMap : !shmemrhs, !rhsTensorMap

      // Step 4 Sanity check of TMA
      scf.if %cnd {
        %lhs0 = memref.load %lhsShmem[%ic7, %ic7] : !shmemlhs        
        %rhs0 = memref.load %rhsShmem[%ic7, %ic1] : !shmemrhs
        %lhs032 = arith.extf %lhs0: f16 to f32
        %rhs032 = arith.extf %rhs0: f16 to f32
        gpu.printf "[GPU] TMA Loaded shmem lhs[0][0] \t %f\n" %lhs032 : f32
        gpu.printf "[GPU] TMA Loaded shmem rhs[0][0] \t %f\n" %rhs032 : f32
        gpu.printf "WGMMA DescA : 0x%llx\n" %descA : i64
        gpu.printf "WGMMA DescB : 0x%llx\n" %descB : i64
      }
      // Step 6. GEMM 128x128x128
      %d2 = arith.constant 2 : i64
      %d4 = arith.constant 4 : i64
      %d6 = arith.constant 6 : i64
      %d8 = arith.constant 8 : i64
      %d512 = arith.constant 512 : i64
      %d514 = arith.constant 514 : i64
      %d516 = arith.constant 516 : i64
      %d518 = arith.constant 518 : i64

      %scaleD = arith.constant 1 : i32 // D = A*B (no accumulate itself)
      nvvm.wgmma.fence.aligned
      %wgmma_result_00 = llvm.mlir.undef : !accMatrix
      %wgmma_result_01 = nvvm.wgmma.mma_async %descA, %descB, 
                  #nvvm.shape<m = 64, n = 128, k = 16>, 
                  D [%wgmma_result_00, #nvvm.wgmma_scale_out<one>],
                  A [<f16>, #nvvm.wgmma_scale_in<one>, <row>], 
                  B [<f16>, #nvvm.wgmma_scale_in<one>, <row>] : !accMatrix -> !accMatrix
      %descA1 = arith.addi %descA, %d2 : i64
      %descB1 = arith.addi %descB, %d2 : i64
      %wgmma_result_02 = nvvm.wgmma.mma_async %descA1, %descB1, 
                  #nvvm.shape<m = 64, n = 128, k = 16>, 
                  D [%wgmma_result_01, #nvvm.wgmma_scale_out<one>],
                  A [<f16>, #nvvm.wgmma_scale_in<one>, <row>], 
                  B [<f16>, #nvvm.wgmma_scale_in<one>, <row>] : !accMatrix -> !accMatrix
      %descA2 = arith.addi %descA, %d4 : i64
      %descB2 = arith.addi %descB, %d4 : i64
      %wgmma_result_03 = nvvm.wgmma.mma_async %descA2, %descB2, 
                  #nvvm.shape<m = 64, n = 128, k = 16>, 
                  D [%wgmma_result_02, #nvvm.wgmma_scale_out<one>],
                  A [<f16>, #nvvm.wgmma_scale_in<one>, <row>], 
                  B [<f16>, #nvvm.wgmma_scale_in<one>, <row>] : !accMatrix -> !accMatrix
      %descA3 = arith.addi %descA, %d6 : i64
      %descB3 = arith.addi %descB, %d6 : i64
      %wgmma_result_04 = nvvm.wgmma.mma_async %descA3, %descB3, 
                  #nvvm.shape<m = 64, n = 128, k = 16>, 
                  D [%wgmma_result_03, #nvvm.wgmma_scale_out<one>],
                  A [<f16>, #nvvm.wgmma_scale_in<one>, <row>], 
                  B [<f16>, #nvvm.wgmma_scale_in<one>, <row>] : !accMatrix -> !accMatrix
      
      %wgmma_result_10 = llvm.mlir.undef : !accMatrix
      %descA4 = arith.addi %descA, %d512 : i64
      %wgmma_result_11 = nvvm.wgmma.mma_async %descA4, %descB, 
                  #nvvm.shape<m = 64, n = 128, k = 16>, 
                  D [%wgmma_result_10, #nvvm.wgmma_scale_out<one>],
                  A [<f16>, #nvvm.wgmma_scale_in<one>, <row>], 
                  B [<f16>, #nvvm.wgmma_scale_in<one>, <row>] : !accMatrix -> !accMatrix
      %descA5 = arith.addi %descA, %d514 : i64
      %descB5 = arith.addi %descB, %d2 : i64
      %wgmma_result_12 = nvvm.wgmma.mma_async %descA5, %descB5, 
                  #nvvm.shape<m = 64, n = 128, k = 16>, 
                  D [%wgmma_result_11, #nvvm.wgmma_scale_out<one>],
                  A [<f16>, #nvvm.wgmma_scale_in<one>, <row>], 
                  B [<f16>, #nvvm.wgmma_scale_in<one>, <row>] : !accMatrix -> !accMatrix
      %descA6 = arith.addi %descA, %d516 : i64
      %descB6 = arith.addi %descB, %d4 : i64
      %wgmma_result_13 = nvvm.wgmma.mma_async %descA6, %descB6, 
                  #nvvm.shape<m = 64, n = 128, k = 16>, 
                  D [%wgmma_result_12, #nvvm.wgmma_scale_out<one>],
                  A [<f16>, #nvvm.wgmma_scale_in<one>, <row>], 
                  B [<f16>, #nvvm.wgmma_scale_in<one>, <row>] : !accMatrix -> !accMatrix
      %descA7 = arith.addi %descA, %d518 : i64
      %descB7 = arith.addi %descB, %d8 : i64
      %wgmma_result_14 = nvvm.wgmma.mma_async %descA7, %descB7, 
                  #nvvm.shape<m = 64, n = 128, k = 16>, 
                  D [%wgmma_result_13, #nvvm.wgmma_scale_out<one>],
                  A [<f16>, #nvvm.wgmma_scale_in<one>, <row>], 
                  B [<f16>, #nvvm.wgmma_scale_in<one>, <row>] : !accMatrix -> !accMatrix

      nvvm.wgmma.commit.group.sync.aligned
      nvvm.wgmma.wait.group.sync.aligned 1
  
      // Step 7. Sanity check WGMMA (2nd and 3rd Warp)
      %cnd125 = arith.cmpi uge, %tidx, %ic63 : index          
      %r0 = llvm.extractvalue %wgmma_result_04[0] : !accMatrix
      %r1 = llvm.extractvalue %wgmma_result_04[1] : !accMatrix
      %r2 = llvm.extractvalue %wgmma_result_04[2] : !accMatrix
      %r3 = llvm.extractvalue %wgmma_result_14[0] : !accMatrix
      %r4 = llvm.extractvalue %wgmma_result_14[1] : !accMatrix
      %r5 = llvm.extractvalue %wgmma_result_14[2] : !accMatrix
      gpu.printf "[GPU][thread=%3d] WGMMA [0]=%4.1f [1]=%4.1f [2]=%4.1f [3]=%4.1f [4]=%4.1f [5]=%4.1f\n" %tidx, %r0, %r1, %r2, %r3, %r4, %r5 : index, f32, f32, f32, f32, f32, f32
      gpu.terminator
  }
  return 
}
}
