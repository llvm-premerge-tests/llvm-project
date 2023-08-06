==============================================================
C++ Standard Parallelism Offload Support: Compiler And Runtime
==============================================================

.. contents::
   :local:

Introduction
============

This document describes the implementation of support for offloading the
execution of standard C++ algorithms to accelerators that can be targeted via
HIP. Furthermore, it enumerates restrictions on user defined code, as well as
the interactions with runtimes.

Algorithm Offload: What, Why, Where
===================================

C++17 introduced overloads
`for most algorithms in the standard library <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0024r2.html>`_
which allow the user to specify a desired
`execution policy <https://en.cppreference.com/w/cpp/algorithm#Execution_policies>`_.
The `parallel_unsequenced_policy <https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t>`_
maps relatively well to the execution model of many accelerators, such as GPUs.
This, coupled with the ubiquity of GPU accelerated algorithm libraries that
implement most / all corresponding libraries in the standard library
(e.g. `rocThrust <https://github.com/ROCmSoftwarePlatform/rocThrust>`_), makes
it feasible to provide seamless accelerator offload for supported algorithms,
when an accelerated version exists. Thus, it becomes possible to easily access
the computational resources of an accelerator, via a well specified, familiar,
algorithmic interface, without having to delve into low-level hardware specific
details. Putting it all together:

- **What**: standard library algorithms, when invoked with the
  ``parallel_unsequenced_policy``
- **Why**: democratise accelerator programming, without loss of user familiarity
- **Where**: any and all accelerators that can be targeted by Clang/LLVM via HIP

Small Example
=============

Given the following C++ code, which assumes the ``std`` namespace is included:

.. code-block:: C++

   bool has_the_answer(const vector<int>& v) {
     return find(execution::par_unseq, cbegin(v), cend(v), 42) != cend(v);
   }

if Clang is invoked with the ``-stdpar --offload-target=foo`` flags, the call to
``find`` will be offloaded to an accelerator that is part of the ``foo`` target
family. If either ``foo`` or its runtime environment do not support transparent
on-demand paging (such as e.g. that provided in Linux via
`HMM <https://docs.kernel.org/mm/hmm.html>`_), it is necessary to also include
the ``--stdpar-interpose-alloc`` flag. If the accelerator specific algorithm
library ``foo`` uses doesn't have an implementation of a particular algorithm,
execution seamlessly falls back to the host CPU. It is legal to specify multiple
``--offload-target``s. All the flags we introduce, as well as a thorough view of
various restrictions and their implications will be provided below.

Implementation - General View
=============================

We built support for Algorithm Offload support atop the pre-existing HIP
infrastructure. More specifically, when one requests offload via ``-stdpar``,
compilation is switched to HIP compilation, as if ``-x hip`` was specified.
Similarly, linking is also switched to HIP linking, as if ``--hip-link`` was
specified. Note that these are implicit, and one should not assume that any
interop with HIP specific language constructs is available e.g. ``__device__``
annotations are neither necessary nor guaranteed to work.

Since there are no language restriction mechanisms in place, it is necessary to
relax HIP language specific semantic checks performed by the FE; they would
identify otherwise valid, offloadable code, as invalid HIP code. Given that we
know that the user intended only for certain algorithms to be offloaded, and
encoded this by specifying the ``parallel_unsequenced_policy``, we rely on a
pass over IR to clean up any and all code that was not "meant" for offload. If
requested, allocation interposition is also handled via a separate pass over IR.

To interface with the client HIP runtime, and to forward offloaded algorithm
invocations to the corresponding accelerator specific library implementation, an
implementation detail forwarding header is implicitly included by the driver,
when compiling with ``-stdpar``. In what follows, we will delve into each
component that contributes to implementing Algorithm Offload support.

Implementation - Driver
=======================

We augment the ``clang`` driver with the following flags:

- ``-stdpar`` / ``--stdpar`` enables algorithm offload, which depending on
  phase, has the following effects:

  - when compiling:

    - ``-x hip`` gets prepended to enable HIP support;
    - the ``ROCmToolchain`` component checks for the ``stdpar_lib.hpp``
      forwarding header,
      `rocThrust <https://rocm.docs.amd.com/projects/rocThrust/en/latest/>`_ and
      `rocPrim <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/>`_ in
      their canonical locations, which can be overriden via flags found below;
      if all are found, the forwarding header gets implicitly included,
      otherwise an error listing the missing component is generated;
    - the ``LangOpts.HIPStdPar`` member is set.

  - when linking:

    - ``--hip-link`` and ``-frtlib-add-rpath`` gets appended to enable HIP
      support.

- ``-stdpar-interpose-alloc`` / ``--stdpar-interpose-alloc`` enables the
  interposition of standard allocation / deallocation functions with accelerator
  aware equivalents; the ``LangOpts.HIPStdParInterposeAlloc`` member is set;
- ``--stdpar-path=`` specifies a non-canonical path for the forwarding header;
  it must point to the folder where the header is located and not to the header
  itself;
- ``--stdpar-thrust-path=`` specifies a non-canonical path for
  `rocThrust <https://rocm.docs.amd.com/projects/rocThrust/en/latest/>`_; it
  must point to the folder where the library is installed / built under a
  ``/thrust`` subfolder;
- ``--stdpar-prim-path=`` specifies a non-canonical path for
  `rocPrim <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/>`_; it must
  point to the folder where the library is installed / built under a
  ``/rocprim`` subfolder;

The `--offload-arch <https://llvm.org/docs/AMDGPUUsage.html#amdgpu-processors>`_
flag can be used to specify the accelerator for which offload code is to be
generated.

Implementation - Front-End
==========================

When ``LangOpts.HIPStdPar`` is set, we relax some of the HIP language specific
``Sema`` checks to account for the fact that we want to consume pure unannotated
C++ code:

1. ``__device__`` / ``__host__ __device__`` functions (which would originate in
   the accelerator specific algorithm library) are allowed to call implicitly
   ``__host__`` functions;
2. ``__global__`` functions (which would originate in the accelerator specific
   algorithm library) are allowed to call implicitly ``__host__`` functions;
3. resolving ``__builtin`` availability is deferred, because it is possible that
   a ``__builtin`` that is unavailable on the target accelerator is not
   reachable from any offloaded algorithm, and thus will be safely removed in
   the middle-end;
4. ASM parsing / checking is deferred, because it is possible that an ASM block
   that e.g. uses some constraints that are incompatible with the target
   accelerator is not reachable from any offloaded algorithm, and thus will be
   safely removed in the middle-end.

``CodeGen`` is similarly relaxed, with implicitly ``__host__`` functions being
emitted as well.

Implementation - Middle-End
===========================

We add two ``opt`` passes:

1. ``StdParAcceleratorCodeSelectionPass``

   - For all kernels in a ``Module``, compute reachability, where a function
     ``F`` is reachable from a kernel ``K`` if and only if there exists a direct
     call-chain rooted in ``F`` that includes ``K``;
   - Remove all functions that are not reachable from kernels;
   - This pass is only run when compiling for the accelerator.

The first pass assumes that the only code that the user intended to offload was
that which was directly or transitively invocable as part of an algorithm
execution. It also assumes that an accelerator aware algorithm implementation
would rely on accelerator specific special functions (kernels), and that these
effectively constitute the only roots for accelerator execution graphs. Both of
these assumptions are based on observing how widespread accelerators,
such as GPUs, work.

1. ``StdParAllocationInterpositionPass``

   - Iterate through all functions in a ``Module``, and replace standard
     allocation / deallocation functions with accelerator-aware equivalents,
     based on a pre-established table; the list of functions that can be
     interposed is available
     `here <https://github.com/ROCmSoftwarePlatform/roc-stdpar#allocation--deallocation-interposition-status>`_;
   - This is only run when compiling for the host.

The second pass is optional.

Implementation - Forwarding Header
==================================

The forwarding header implements two pieces of functionality:

1. It forwards algorithms to a target accelerator, which is done by relying on
   C++ language rules around overloading:

   - overloads taking an explicit argument of type
     ``parallel_unsequenced_policy`` are introduced into the ``std`` namespace;
   - these will get preferentially selected versus the master template;
   - the body forwards to the equivalent algorithm from the accelerator specific
     library

2. It provides allocation / deallocation functions that are equivalent to the
   standard ones, but obtain memory by invoking
   `hipMallocManaged <https://rocm.docs.amd.com/projects/HIP/en/latest/.doxygen/docBin/html/group___memory_m.html#gab8cfa0e292193fa37e0cc2e4911fa90a>`_
   and release it via `hipFree <https://rocm.docs.amd.com/projects/HIP/en/latest/.doxygen/docBin/html/group___memory.html#ga740d08da65cae1441ba32f8fedb863d1>`_.

Restrictions
============

We define two modes in which runtime execution can occur:

1. **HMM Mode** - this assumes that the
   `HMM <https://docs.kernel.org/mm/hmm.html>`_ subsystem of the Linux kernel
   is used to provide transparent on-demand paging i.e. memory obtained from a
   system / OS allocator such as via a call to ``malloc`` or ``operator new`` is
   directly accessible to the accelerator and it follows the C++ memory model;
2. **Interposition Mode** - this is a fallback mode for cases where transparent
   on-demand paging is unavailable (e.g. in the Windows OS), which means that
   memory must be allocated via an accelerator aware mechanism, and system
   allocated memory is inaccessible for the accelerator.

The following restrictions imposed on user code apply to both modes:

1. Pointers to function, and all associated features, such as e.g. dynamic
   polymorphism, cannot be used (directly or transitively) by the user provided
   callable passed to an algorithm invocation;
2. Global / namespace scope / ``static`` / ``thread`` storage duration variables
   cannot be used (directly or transitively) in name by the user provided
   callable;

   - When executing in **HMM Mode** they can be used in address e.g.:

     .. code-block:: C++

        namespace { int foo = 42; }

        bool never(const vector<int>& v) {
          return any_of(execution::par_unseq, cbegin(v), cend(v), [](auto&& x) {
            return x == foo;
          });
        }

        bool only_in_hmm_mode(const vector<int>& v) {
          return any_of(execution::par_unseq, cbegin(v), cend(v),
                        [p = &foo](auto&& x) { return x == *p; });
        }

3. Only algorithms that are invoked with the ``parallel_unsequenced_policy`` are
   candidates for offload;
4. Only algorithms that are invoked with iterator arguments that model
   `random_access_iterator <https://en.cppreference.com/w/cpp/iterator/random_access_iterator>`_
   are candidates for offload;
5. `Exceptions <https://en.cppreference.com/w/cpp/language/exceptions>`_ cannot
   be used by the user provided callable;
6. Dynamic memory allocation (e.g. ``operator new``) cannot be used by the user
   provided callable;
7. Selective offload is not possible i.e. it is not possible to indicate that
   only some algorithms invoked with the ``parallel_unsequenced_policy`` are to
   be executed on the accelerator.

In addition to the above, using **Interposition Mode** imposes the following
additional restrictions:

1. All code that is expected to interoperate has to be recompiled with the
   ``--stdpar-interpose-alloc`` flag i.e. it is not safe to compose libraries
   that have been independently compiled;
2. automatic storage duration (i.e. stack allocated) variables cannot be used
   (directly or transitively) by the user provided callable e.g.

   .. code-block:: c++

      bool never(const vector<int>& v, int n) {
        return any_of(execution::par_unseq, cbegin(v), cend(v),
                      [p = &n](auto&& x) { return x == *p; });
      }

Current Support
===============

At the moment, C++ Standard Parallelism Offload is only available for AMD GPUs,
when the `ROCm <https://rocm.docs.amd.com/en/latest/>`_ stack is used, on the
Linux operating system. Whilst the design outlined above is generic and target
independent, only the above combination has been validated. In the future, as
other accelerators that can be targeted via HIP are validated, and if they
choose to implement a forwarding header (or contribute to the existing one),
support will be extended.

Focusing on AMD GPU targets, support is synthesised in the following table

.. list-table::
   :header-rows: 1

   * - `Processor <https://llvm.org/docs/AMDGPUUsage.html#amdgpu-processors>`_
     - HMM Mode
     - Interposition Mode
   * - GCN GFX9 (Vega)
     - YES
     - YES
   * - GCN GFX10.1 (RDNA 1)
     - *NO*
     - YES
   * - GCN GFX10.3 (RDNA 2)
     - *NO*
     - YES
   * - GCN GFX11 (RDNA 3)
     - *NO*
     - YES

The minimum Linux kernel version for running in HMM mode is 6.4.

The forwarding header can be obtained from
`its GitHub repository <https://github.com/ROCmSoftwarePlatform/roc-stdpar>`_.
It will be packaged with a future `ROCm <https://rocm.docs.amd.com/en/latest/>`_
release. Because accelerated algorithms are provided via
`rocThrust <https://rocm.docs.amd.com/projects/rocThrust/en/latest/>`_, a
transitive dependency on
`rocPrim <https://rocm.docs.amd.com/projects/rocPRIM/en/latest/>`_ exists. Both
can be obtained either by installing their associated components of the
`ROCm <https://rocm.docs.amd.com/en/latest/>`_ stack, or from their respective
repositories. The list algorithms that can be offloaded is available
`here <https://github.com/ROCmSoftwarePlatform/roc-stdpar#algorithm-support-status>`_.

HIP Specific Elements
---------------------

Whilst the support for C++ Standard Parallelism Offload is generic, and not
defined in terms of the HIP language or HIP runtime APIs, there are consequences
to using the latter in the implementation. We enumerate a few which are likely
to be relevant to users:

1. There is no defined interop with the
   `HIP kernel language <https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html>`_;
   whilst things like using `__device__` annotations might accidentally "work",
   they are not guaranteed to, and thus cannot be relied upon by user code;
2. Combining explicit HIP compilation with ``--stdpar`` based offloading is not
   allowed or supported in any way.
3. There is no way to target different accelerators via a standard algorithm
   invocation (`this might be addressed in future C++ standards <https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2500r1.html>`_).

Open Questions / Future Developments
====================================

1. The restriction on the use of global / namespace scope / ``static`` /
   ``thread`` storage duration variables in offloaded algorithms will be lifted
   in the future, when running in **HMM Mode**;
2. The restriction on the use of dynamic memory allocation in offloaded
   algorithms will be lifted in the future.
3. The restriction on the use of pointers to function, and associated features
   such as dynamic polymorphism might be lifted in the future, when running in
   **HMM Mode**;
4. Offload support might be extended to cases where the ``parallel_policy`` is
   used for some or all targets.
