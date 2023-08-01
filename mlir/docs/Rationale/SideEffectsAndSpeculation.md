# Side Effects & Speculation

This document outlines how MLIR models side effects and how speculation works in
MLIR.

This rationale only applies to operations used in
[CFG regions](../LangRef.md#control-flow-and-ssacfg-regions). Side effect
modeling in [graph regions](../LangRef.md#graph-regions) is TBD.

[TOC]

## Overview

Many MLIR operations don't exhibit any behavior other than consuming and
producing SSA values. These operations can be reordered with other operations as
long as they obey SSA dominance requirements and can be eliminated or even
introduced (e.g. for
[rematerialization](https://en.wikipedia.org/wiki/Rematerialization)) as needed.

However, a subset of MLIR operations have implicit behavior than isn't reflected
in their SSA data-flow semantics. These operations need special handing, and
cannot be reordered, eliminated or introduced without additional analysis.

This doc introduces a categorization of these operations and shows how these
operations are modeled in MLIR.

## Categorization

Operations with implicit behaviors can be broadly categorized as follows:

1. Operations with memory effects. These operations read from and write to some
   mutable system resource, e.g. the heap, the stack, HW registers, the console.
   They may also interact with the heap in other ways, like by allocating and
   freeing memory. E.g. standard memory reads and writes, `printf` (which can be
   modeled as "writing" to the console and reading from the input buffers).
1. Operations with undefined behavior. These operations are not defined on
   certain inputs or in some situations -- we do not specify what happens when
   such illegal inputs are passed, and instead say that behavior is undefined
   and can assume it does not happen. In practice, in such cases these ops may
   do anything from producing garbage results to crashing the program or
   corrupting memory. E.g. integer division which has UB when dividing by zero,
   loading from a pointer that has been freed.
1. Operations that don't terminate. E.g. an `scf.while` where the condition is
   always true.
1. Operations with non-local control flow. These operations may pop their
   current frame of execution and return directly to an older frame. E.g.
   `longjmp`, operations that throw exceptions.

Finally, a given operation may have a combination of the above implicit
behaviors. The combination of implicit behaviors during the execution of the
operation may be ordered. We use 'stage' to label the order of implicit
behaviors during the execution of 'op'. Implicit behaviors with a lower stage
number happen earlier than those with a higher stage number.

## Modeling

Modeling these behaviors has to walk a fine line -- we need to empower more
complicated passes to reason about the nuances of such behaviors while
simultaneously not overburdening simple passes that only need a coarse grained
"can this op be freely moved" query.

MLIR has two op interfaces to represent these implicit behaviors:

1. The
   [`MemoryEffectsOpInterface` op interface](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.td#L26)
   is used to track memory effects.
1. The
   [`ConditionallySpeculatable` op interface](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Interfaces/SideEffectInterfaces.td#L105)
   is used to track undefined behavior and infinite loops.

Both of these are op interfaces which means operations can dynamically
introspect themselves (e.g. by checking input types or attributes) to infer what
memory effects they have and whether they are speculatable.

We don't have proper modeling yet to fully capture non-local control flow
semantics.

When adding a new op, ask:

1. Does it read from or write to the heap or stack? It should probably implement
   `MemoryEffectsOpInterface`.
1. Does these side effects ordered? It should probably set the stage of
   side effects to make analysis more accurate.
1. Does These side effects act on every single value of resource? If not, it
   should set the effectOnFullRegion field with false.
1. Does it have side effects that must be preserved, like a volatile store or a
   syscall? It should probably implement `MemoryEffectsOpInterface` and model
   the effect as a read from or write to an abstract `Resource`. Please start an
   RFC if your operation has a novel side effect that cannot be adequately
   captured by `MemoryEffectsOpInterface`.
1. Is it well defined in all inputs or does it assume certain runtime
   restrictions on its inputs, e.g. the pointer operand must point to valid
   memory? It should probably implement `ConditionallySpeculatable`.
1. Can it infinitely loop on certain inputs? It should probably implement
   `ConditionallySpeculatable`.
1. Does it have non-local control flow (e.g. `longjmp`)? We don't have proper
   modeling for these yet, patches welcome!
1. Is your operation free of side effects and can be freely hoisted, introduced
   and eliminated? It should probably be marked `Pure`. (TODO: revisit this name
   since it has overloaded meanings in C++.)

## Examples

This section describes a few very simple examples that help understand how to
add side effect correctly.

### SIMD compute operation

If we have a SIMD backend dialect and have a "simd.abs" operation, which read
all value in source memref, calculate its absolute value and write to target
memref.

```mlir
  func.func @abs(%source : memref<10xf32>, %target : memref<10xf32>) {
    simd.abs(%source, %target) : memref<10xf32> to memref<10xf32>
    return
  }
```

The abs operation reads every single value from the source resource and
then writes these values to every single value in the target resource.
Therefore, we need to specify a read side effect for the source and a write side
effect for the target. The read side effect occurs before the write side effect,
so we need to mark the read stage as earlier than the write stage. Additionally,
we need to mark these side effects as acting on every single value in the
resource.

A typical approach is as follows:
``` mlir
  def AbsOp : SIMD_Op<"abs", [...] {
    ...

    let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the source memref",
                             [MemReadAt<0>]>:$source,
                         Arg<AnyRankedOrUnrankedMemRef, "the target memref",
                             [MemWriteAt<1>]>:$target);

    ...
  }
```

In the above example, we added the side effect [MemReadAt<0>] to the source,
indicating abs operation reads every single value from source in stage 0.
[MemReadAt<0>] is a shorthand notation for [MemReadAt<0, true>]. We added the
side effect [MemWriteAt<0>] to the target, indicating abs operation writes on
every single value inside the target on stage 1(after read from source).

### Load like operation

Memref.load is a typical load like operation:
```mlir
  func.func @foo(%input : memref<10xf32>, %index : index) -> f32 {
    %result = memref.load  %input[index] : memref<10xf32>
    return %result : f32
  }
```

The load like operation read one value from input memref and return it.
Therefore, we needs to specify a read side effect for input memref, and mark
not every single value is used.

A typical approach is as follows:
``` mlir
  def LoadOp : MemRef_Op<"load", [...] {
    ...

    let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                             [MemReadAt<0, false>]>:$memref,
                         Variadic<Index>:$indices,
                         DefaultValuedOptionalAttr<BoolAttr, "false">:$nontemporal);

    ...
  }
```

In the above example, we added the side effect [MemReadAt<0, false>] to the
source, indicating load operation read parts of value from memref at stage 0.
