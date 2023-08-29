# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import gpu

from ext_test_helper import print_gen_init_arg_names


def run(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            sequence = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                [],
                transform.AnyOpType.get(),
            )
            with InsertionPoint(sequence.body):
                f(sequence.bodyTarget)
                transform.YieldOp()
        print(module)
    return f


@run
def testMapForallToBlocksCompact(target):
    gpu.MapForallToBlocks(target)
    # CHECK-LABEL: TEST: testMapForallToBlocksCompact
    # CHECK: = transform.gpu.map_forall_to_blocks
    # CHECK-NOT: grid_dims
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op
    # CHECK-NOT: grid_dims


@run
def testMapForallToBlocksTyped(target):
    gpu.MapForallToBlocks(transform.OperationType.get("test.dummy"), target)
    # CHECK-LABEL: TEST: testMapForallToBlocksTyped
    # CHECK: = transform.gpu.map_forall_to_blocks
    # CHECK-SAME: (!transform.any_op) -> !transform.op<"test.dummy">


@run
def testMapForallToBlocksGridDims(target):
    gpu.MapForallToBlocks(target, grid_dims=[4, 2])
    # CHECK-LABEL: TEST: testMapForallToBlocksGridDims
    # CHECK: = transform.gpu.map_forall_to_blocks
    # CHECK-SAME: grid_dims = [4, 2]
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op


@run
def testMapNestedForallToThreadsCompact(target):
    gpu.MapNestedForallToThreads(target)
    # CHECK-LABEL: TEST: testMapNestedForallToThreadsCompact
    # CHECK: transform.gpu.map_nested_forall_to_threads
    # CHECK-SAME: block_dims = []
    # CHECK-SAME: (!transform.any_op) -> !transform.any_op


@run
def testMapNestedForallToThreadsTyped(target):
    gpu.MapNestedForallToThreads(transform.OperationType.get("test.dummy"), target)
    # CHECK-LABEL: TEST: testMapNestedForallToThreadsTyped
    # CHECK: transform.gpu.map_nested_forall_to_threads
    # CHECK-SAME: block_dims = []
    # CHECK-SAME: (!transform.any_op) -> !transform.op<"test.dummy">


@run
def testMapNestedForallToThreadsAttributes(target):
    gpu.MapNestedForallToThreads(
        target, block_dims=[4, 2], warp_size=64, sync_after_distribute=False
    )
    # CHECK-LABEL: TEST: testMapNestedForallToThreadsAttributes
    # CHECK: transform.gpu.map_nested_forall_to_threads
    # CHECK-SAME: block_dims = [4, 2]
    # CHECK-SAME: sync_after_distribute = false
    # CHECK-SAME: warp_size = 64


@run
def testMapNestedForallToThreadsGenArgs(_):
    print_gen_init_arg_names(gpu.MapNestedForallToThreads)
    # CHECK-LABEL: TEST: testMapNestedForallToThreadsGenArgs
    # CHECK: ['self', 'result', 'target', 'block_dims', 'sync_after_distribute', 'warp_size', 'loc', 'ip']
