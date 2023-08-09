module {
  llvm.func @init_mbarrier_arrive_expect_tx(%arg0: !llvm.ptr<3>, %arg1: i32) {
    llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.shared.b64 _, [$0], $1;", "r,r" %arg0, %arg1 : (!llvm.ptr<3>, i32) -> ()
    llvm.return
  }
  llvm.func @init_mbarrier_arrive_expect_tx_generic(%arg0: !llvm.ptr, %arg1: i32) {
    llvm.inline_asm has_side_effects asm_dialect = att "mbarrier.arrive.expect_tx.b64 _, [$0], $1;", "l,r" %arg0, %arg1 : (!llvm.ptr, i32) -> ()
    llvm.return
  }
  llvm.func @init_mbarrier_try_wait_shared(%arg0: !llvm.ptr<3>, %arg1: i32, %arg2: i32) {
    llvm.inline_asm has_side_effects asm_dialect = att "{\0A\09.reg .pred       P1; \0A\09LAB_WAIT: \0A\09mbarrier.try_wait.parity.shared.b64 P1, [$0], $1, $2; \0A\09@P1 bra.uni DONE; \0A\09bra.uni     LAB_WAIT; \0A\09DONE: \0A\09}", "r,r,r" %arg0, %arg2, %arg1 : (!llvm.ptr<3>, i32, i32) -> ()
    llvm.return
  }
  llvm.func @init_mbarrier_try_wait(%arg0: !llvm.ptr, %arg1: i32, %arg2: i32) {
    llvm.inline_asm has_side_effects asm_dialect = att "{\0A\09.reg .pred       P1; \0A\09LAB_WAIT: \0A\09mbarrier.try_wait.parity.b64 P1, [$0], $1, $2; \0A\09@P1 bra.uni DONE; \0A\09bra.uni     LAB_WAIT; \0A\09DONE: \0A\09}", "l,r,r" %arg0, %arg2, %arg1 : (!llvm.ptr, i32, i32) -> ()
    llvm.return
  }
  func.func @async_cp(%arg0: !llvm.ptr<3>, %arg1: !llvm.ptr<1>) {
    nvvm.cp.async.shared.global %arg0, %arg1, 16, cache =  ca : !llvm.ptr<3>, !llvm.ptr<1>
    nvvm.cp.async.shared.global %arg0, %arg1, 16, cache =  cg : !llvm.ptr<3>, !llvm.ptr<1>
    return
  }
  func.func @async_cp_zfill(%arg0: !llvm.ptr<3>, %arg1: !llvm.ptr<1>, %arg2: i32) {
    %0 = llvm.mlir.constant(16 : i32) : i32
    llvm.inline_asm has_side_effects asm_dialect = att "cp.async.cg.shared.global [$0], [$1], $2, $3;\0A", "r,l,n,r" %arg0, %arg1, %0, %arg2 : (!llvm.ptr<3>, !llvm.ptr<1>, i32, i32) -> ()
    %1 = llvm.mlir.constant(4 : i32) : i32
    llvm.inline_asm has_side_effects asm_dialect = att "cp.async.ca.shared.global [$0], [$1], $2, $3;\0A", "r,l,n,r" %arg0, %arg1, %1, %arg2 : (!llvm.ptr<3>, !llvm.ptr<1>, i32, i32) -> ()
    return
  }
  func.func @tma_load_1d(%arg0: !llvm.ptr, %arg1: !llvm.ptr<3>, %arg2: !llvm.ptr<3>, %arg3: i32) {
    llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$3} ], [$2];", "r,l,r,r" %arg1, %arg0, %arg2, %arg3 : (!llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>, i32) -> ()
    return
  }
  func.func @tma_load_2d(%arg0: !llvm.ptr, %arg1: !llvm.ptr<3>, %arg2: !llvm.ptr<3>, %arg3: i32, %arg4: i32) {
    llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4} ], [$2];", "r,l,r,r,r" %arg1, %arg0, %arg2, %arg3, %arg4 : (!llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>, i32, i32) -> ()
    return
  }
  func.func @tma_load_3d(%arg0: !llvm.ptr, %arg1: !llvm.ptr<3>, %arg2: !llvm.ptr<3>, %arg3: i32, %arg4: i32, %arg5: i32) {
    llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5} ], [$2];", "r,l,r,r,r,r" %arg1, %arg0, %arg2, %arg3, %arg4, %arg5 : (!llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>, i32, i32, i32) -> ()
    return
  }
  func.func @tma_load_4d(%arg0: !llvm.ptr, %arg1: !llvm.ptr<3>, %arg2: !llvm.ptr<3>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5, $6} ], [$2];", "r,l,r,r,r,r,r" %arg1, %arg0, %arg2, %arg3, %arg4, %arg5, %arg6 : (!llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>, i32, i32, i32, i32) -> ()
    return
  }
  func.func @tma_load_5d(%arg0: !llvm.ptr, %arg1: !llvm.ptr<3>, %arg2: !llvm.ptr<3>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
    llvm.inline_asm has_side_effects asm_dialect = att "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1, {$3, $4, $5, $6, $7} ], [$2];", "r,l,r,r,r,r,r,r" %arg1, %arg0, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 : (!llvm.ptr<3>, !llvm.ptr, !llvm.ptr<3>, i32, i32, i32, i32, i32) -> ()
    return
  }
  func.func @wgmma_execute() {
    llvm.inline_asm has_side_effects asm_dialect = att "wgmma.fence.sync.aligned;", ""  : () -> ()
    llvm.inline_asm has_side_effects asm_dialect = att "wgmma.commit_group.sync.aligned;", ""  : () -> ()
    %0 = llvm.mlir.constant(0 : i32) : i32
    llvm.inline_asm has_side_effects asm_dialect = att "wgmma.wait_group.sync.aligned $0;", "n" %0 : (i32) -> ()
    llvm.inline_asm has_side_effects asm_dialect = att "wgmma.fence.sync.aligned;", ""  : () -> ()
    llvm.inline_asm has_side_effects asm_dialect = att "wgmma.commit_group.sync.aligned;", ""  : () -> ()
    %1 = llvm.mlir.constant(1 : i32) : i32
    llvm.inline_asm has_side_effects asm_dialect = att "wgmma.wait_group.sync.aligned $0;", "n" %1 : (i32) -> ()
    return
  }
}


// -----
module {
  func.func @wgmma_f32_f16_f16(%arg0: i64, %arg1: i64) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> {
    %0 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.mlir.constant(-1 : i32) : i32
    %3 = llvm.mlir.constant(-1 : i32) : i32
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(1 : i32) : i32
    %6 = llvm.extractvalue %0[0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %7 = llvm.extractvalue %0[1] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %8 = llvm.extractvalue %0[2] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %9 = llvm.extractvalue %0[3] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %10 = llvm.extractvalue %0[4] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %11 = llvm.extractvalue %0[5] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %12 = llvm.extractvalue %0[6] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %13 = llvm.extractvalue %0[7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %14 = llvm.extractvalue %0[8] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %15 = llvm.extractvalue %0[9] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %16 = llvm.extractvalue %0[10] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %17 = llvm.extractvalue %0[11] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %18 = llvm.extractvalue %0[12] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %19 = llvm.extractvalue %0[13] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %20 = llvm.extractvalue %0[14] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %21 = llvm.extractvalue %0[15] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %22 = llvm.inline_asm has_side_effects asm_dialect = att "{\0A.reg .pred p;\0Asetp.ne.b32 p, $18, 0;\0Awgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, $16, $17, p, $19,  $20, $21,  $22;\0A}\0A", "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,l,l,n,n,n,n,n" %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %arg0, %arg1, %1, %2, %3, %4, %5 : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, i64, i64, i32, i32, i32, i32, i32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    %c2_i64 = arith.constant 2 : i64
    %23 = arith.addi %arg0, %c2_i64 : i64
    %24 = arith.addi %arg1, %c2_i64 : i64
    %25 = llvm.mlir.constant(0 : i32) : i32
    %26 = llvm.mlir.constant(-1 : i32) : i32
    %27 = llvm.mlir.constant(-1 : i32) : i32
    %28 = llvm.mlir.constant(1 : i32) : i32
    %29 = llvm.mlir.constant(1 : i32) : i32
    %30 = llvm.extractvalue %22[0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %31 = llvm.extractvalue %22[1] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %32 = llvm.extractvalue %22[2] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %33 = llvm.extractvalue %22[3] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %34 = llvm.extractvalue %22[4] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %35 = llvm.extractvalue %22[5] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %36 = llvm.extractvalue %22[6] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %37 = llvm.extractvalue %22[7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %38 = llvm.extractvalue %22[8] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %39 = llvm.extractvalue %22[9] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %40 = llvm.extractvalue %22[10] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %41 = llvm.extractvalue %22[11] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %42 = llvm.extractvalue %22[12] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %43 = llvm.extractvalue %22[13] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %44 = llvm.extractvalue %22[14] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %45 = llvm.extractvalue %22[15] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %46 = llvm.inline_asm has_side_effects asm_dialect = att "{\0A.reg .pred p;\0Asetp.ne.b32 p, $18, 0;\0Awgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15}, $16, $17, p, $19,  $20, $21,  $22;\0A}\0A", "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,l,l,n,n,n,n,n" %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %23, %24, %25, %26, %27, %28, %29 : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, i64, i64, i32, i32, i32, i32, i32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    return %46 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  }
}


// -----
module {
  func.func @wgmma_s32_s8_s8_satfinite(%arg0: i64, %arg1: i64) -> !llvm.struct<(i32, i32, i32, i32)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.extractvalue %0[0] : !llvm.struct<(i32, i32, i32, i32)> 
    %3 = llvm.extractvalue %0[1] : !llvm.struct<(i32, i32, i32, i32)> 
    %4 = llvm.extractvalue %0[2] : !llvm.struct<(i32, i32, i32, i32)> 
    %5 = llvm.extractvalue %0[3] : !llvm.struct<(i32, i32, i32, i32)> 
    %6 = llvm.inline_asm has_side_effects asm_dialect = att "{\0A.reg .pred p;\0Asetp.ne.b32 p, $6, 0;\0Awgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite {$0, $1, $2, $3}, $4, $5, p;\0A}\0A", "=r,=r,=r,=r,0,1,2,3,l,l,n" %2, %3, %4, %5, %arg0, %arg1, %1 : (i32, i32, i32, i32, i64, i64, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.extractvalue %6[0] : !llvm.struct<(i32, i32, i32, i32)> 
    %9 = llvm.extractvalue %6[1] : !llvm.struct<(i32, i32, i32, i32)> 
    %10 = llvm.extractvalue %6[2] : !llvm.struct<(i32, i32, i32, i32)> 
    %11 = llvm.extractvalue %6[3] : !llvm.struct<(i32, i32, i32, i32)> 
    %12 = llvm.inline_asm has_side_effects asm_dialect = att "{\0A.reg .pred p;\0Asetp.ne.b32 p, $6, 0;\0Awgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite {$0, $1, $2, $3}, $4, $5, p;\0A}\0A", "=r,=r,=r,=r,0,1,2,3,l,l,n" %8, %9, %10, %11, %arg0, %arg1, %7 : (i32, i32, i32, i32, i64, i64, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.extractvalue %12[0] : !llvm.struct<(i32, i32, i32, i32)> 
    %15 = llvm.extractvalue %12[1] : !llvm.struct<(i32, i32, i32, i32)> 
    %16 = llvm.extractvalue %12[2] : !llvm.struct<(i32, i32, i32, i32)> 
    %17 = llvm.extractvalue %12[3] : !llvm.struct<(i32, i32, i32, i32)> 
    %18 = llvm.inline_asm has_side_effects asm_dialect = att "{\0A.reg .pred p;\0Asetp.ne.b32 p, $6, 0;\0Awgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite {$0, $1, $2, $3}, $4, $5, p;\0A}\0A", "=r,=r,=r,=r,0,1,2,3,l,l,n" %14, %15, %16, %17, %arg0, %arg1, %13 : (i32, i32, i32, i32, i64, i64, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    return %18 : !llvm.struct<(i32, i32, i32, i32)>
  }
  func.func @wgmma_s32_u8_u8(%arg0: i64, %arg1: i64) -> !llvm.struct<(i32, i32, i32, i32)> {
    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32, i32, i32)>
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.extractvalue %0[0] : !llvm.struct<(i32, i32, i32, i32)> 
    %3 = llvm.extractvalue %0[1] : !llvm.struct<(i32, i32, i32, i32)> 
    %4 = llvm.extractvalue %0[2] : !llvm.struct<(i32, i32, i32, i32)> 
    %5 = llvm.extractvalue %0[3] : !llvm.struct<(i32, i32, i32, i32)> 
    %6 = llvm.inline_asm has_side_effects asm_dialect = att "{\0A.reg .pred p;\0Asetp.ne.b32 p, $6, 0;\0Awgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 {$0, $1, $2, $3}, $4, $5, p;\0A}\0A", "=r,=r,=r,=r,0,1,2,3,l,l,n" %2, %3, %4, %5, %arg0, %arg1, %1 : (i32, i32, i32, i32, i64, i64, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    %7 = llvm.mlir.constant(1 : i32) : i32
    %8 = llvm.extractvalue %6[0] : !llvm.struct<(i32, i32, i32, i32)> 
    %9 = llvm.extractvalue %6[1] : !llvm.struct<(i32, i32, i32, i32)> 
    %10 = llvm.extractvalue %6[2] : !llvm.struct<(i32, i32, i32, i32)> 
    %11 = llvm.extractvalue %6[3] : !llvm.struct<(i32, i32, i32, i32)> 
    %12 = llvm.inline_asm has_side_effects asm_dialect = att "{\0A.reg .pred p;\0Asetp.ne.b32 p, $6, 0;\0Awgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 {$0, $1, $2, $3}, $4, $5, p;\0A}\0A", "=r,=r,=r,=r,0,1,2,3,l,l,n" %8, %9, %10, %11, %arg0, %arg1, %7 : (i32, i32, i32, i32, i64, i64, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    %13 = llvm.mlir.constant(1 : i32) : i32
    %14 = llvm.extractvalue %12[0] : !llvm.struct<(i32, i32, i32, i32)> 
    %15 = llvm.extractvalue %12[1] : !llvm.struct<(i32, i32, i32, i32)> 
    %16 = llvm.extractvalue %12[2] : !llvm.struct<(i32, i32, i32, i32)> 
    %17 = llvm.extractvalue %12[3] : !llvm.struct<(i32, i32, i32, i32)> 
    %18 = llvm.inline_asm has_side_effects asm_dialect = att "{\0A.reg .pred p;\0Asetp.ne.b32 p, $6, 0;\0Awgmma.mma_async.sync.aligned.m64n8k32.s32.u8.u8 {$0, $1, $2, $3}, $4, $5, p;\0A}\0A", "=r,=r,=r,=r,0,1,2,3,l,l,n" %14, %15, %16, %17, %arg0, %arg1, %13 : (i32, i32, i32, i32, i64, i64, i32) -> !llvm.struct<(i32, i32, i32, i32)>
    return %18 : !llvm.struct<(i32, i32, i32, i32)>
  }
}


// -----
module {
  func.func @wgmma_f32_tf32_tf32(%arg0: i64, %arg1: i64) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> {
    %0 = llvm.mlir.undef : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    %1 = llvm.mlir.constant(1 : i32) : i32
    %2 = llvm.mlir.constant(1 : i32) : i32
    %3 = llvm.mlir.constant(1 : i32) : i32
    %4 = llvm.extractvalue %0[0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %5 = llvm.extractvalue %0[1] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %6 = llvm.extractvalue %0[2] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %7 = llvm.extractvalue %0[3] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %8 = llvm.extractvalue %0[4] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %9 = llvm.extractvalue %0[5] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %10 = llvm.extractvalue %0[6] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %11 = llvm.extractvalue %0[7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %12 = llvm.extractvalue %0[8] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %13 = llvm.extractvalue %0[9] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %14 = llvm.extractvalue %0[10] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %15 = llvm.extractvalue %0[11] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %16 = llvm.extractvalue %0[12] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %17 = llvm.extractvalue %0[13] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %18 = llvm.extractvalue %0[14] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %19 = llvm.extractvalue %0[15] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %20 = llvm.extractvalue %0[16] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %21 = llvm.extractvalue %0[17] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %22 = llvm.extractvalue %0[18] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %23 = llvm.extractvalue %0[19] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %24 = llvm.extractvalue %0[20] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %25 = llvm.extractvalue %0[21] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %26 = llvm.extractvalue %0[22] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %27 = llvm.extractvalue %0[23] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %28 = llvm.extractvalue %0[24] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %29 = llvm.extractvalue %0[25] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %30 = llvm.extractvalue %0[26] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %31 = llvm.extractvalue %0[27] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %32 = llvm.extractvalue %0[28] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %33 = llvm.extractvalue %0[29] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %34 = llvm.extractvalue %0[30] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %35 = llvm.extractvalue %0[31] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %36 = llvm.inline_asm has_side_effects asm_dialect = att "{\0A.reg .pred p;\0Asetp.ne.b32 p, $34, 0;\0Awgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $32, $33, p, $35,  $36;\0A}\0A", "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,n,n,n" %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %arg0, %arg1, %1, %2, %3 : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, i64, i64, i32, i32, i32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    %37 = llvm.mlir.constant(1 : i32) : i32
    %38 = llvm.mlir.constant(1 : i32) : i32
    %39 = llvm.mlir.constant(1 : i32) : i32
    %40 = llvm.extractvalue %36[0] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %41 = llvm.extractvalue %36[1] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %42 = llvm.extractvalue %36[2] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %43 = llvm.extractvalue %36[3] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %44 = llvm.extractvalue %36[4] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %45 = llvm.extractvalue %36[5] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %46 = llvm.extractvalue %36[6] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %47 = llvm.extractvalue %36[7] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %48 = llvm.extractvalue %36[8] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %49 = llvm.extractvalue %36[9] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %50 = llvm.extractvalue %36[10] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %51 = llvm.extractvalue %36[11] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %52 = llvm.extractvalue %36[12] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %53 = llvm.extractvalue %36[13] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %54 = llvm.extractvalue %36[14] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %55 = llvm.extractvalue %36[15] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %56 = llvm.extractvalue %36[16] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %57 = llvm.extractvalue %36[17] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %58 = llvm.extractvalue %36[18] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %59 = llvm.extractvalue %36[19] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %60 = llvm.extractvalue %36[20] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %61 = llvm.extractvalue %36[21] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %62 = llvm.extractvalue %36[22] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %63 = llvm.extractvalue %36[23] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %64 = llvm.extractvalue %36[24] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %65 = llvm.extractvalue %36[25] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %66 = llvm.extractvalue %36[26] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %67 = llvm.extractvalue %36[27] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %68 = llvm.extractvalue %36[28] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %69 = llvm.extractvalue %36[29] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %70 = llvm.extractvalue %36[30] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %71 = llvm.extractvalue %36[31] : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)> 
    %72 = llvm.inline_asm has_side_effects asm_dialect = att "{\0A.reg .pred p;\0Asetp.ne.b32 p, $34, 0;\0Awgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 {$0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28, $29, $30, $31}, $32, $33, p, $35,  $36;\0A}\0A", "=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,=f,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,l,l,n,n,n" %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %arg0, %arg1, %37, %38, %39 : (f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, i64, i64, i32, i32, i32) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
    return %72 : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  }
}

