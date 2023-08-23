================
C++ Safe Buffers
================

.. contents::
   :local:


Introduction
============

Clang can be used to harden your C++ code against buffer overflows, an otherwise
common security issue with C-based languages.

The solution described in this document is an integrated programming model:
it defines safety guidelines that restrict operations the code is allowed
to perform, it provides tools for updating the code to conform to these
guidelines, and it provides runtime mitigations for the situations when
following guidelines isn't sufficient.

Namely, the solution consists of the following parts:

  - ``-Wunsafe-buffer-usage`` is a family of warnings that warns you when
    a potentially unsafe operation is performed in your code;
  - Pragmas ``unsafe_buffer_usage`` and ``only_safe_buffers`` allow you to
    annotate sections of code as opt-out of or opt-into the programming model,
    which enables incremental adoption and provides "escape hatches"
    when unsafety is necessary;
  - Automatic fixits provided by the warning act as a modernizer to help you
    convert large amounts of old code to conform to the warning;
  - Attribute ``[[unsafe_buffer_usage]]`` lets you annotate custom functions as
    unsafe, while providing a safe alternative that can often be suggested by
    the compiler automatically;
  - LLVM's own C++ standard library implementation, libc++, provides a
    hardened mode where C++ classes such as ``std::vector`` and ``std::span``,
    together with their respective ``iterator`` classes, are protected
    at runtime against buffer overflows. This changes buffer overflows from
    extremely dangerous undefined behavior to a predictable runtime crash;
  - Finally, in order to avoid bugs in newly converted code, the
    Clang static analyzer provides a checker to find misconstructed
    span/view objects.

Note that some of these partial solutions are useful on their own. For example,
hardened libc++ can be used as a "sanitizer" to find buffer overflow bugs in
existing code. The static analyzer checker is also universally useful, and
not tied to this programming model. The warning may be desired independently,
even in plain C code, if you need to isolate, annotate and audit all
buffer-related code in your codebase. That said, some of these guidelines
don't cause your code to mitigate vulnerabilities better unless hardened
containers are in use to carry out runtime checks.


Why Not Just Harden All Pointer Operations?
-------------------------------------------

At a glance, it seems reasonable to ask the compiler to make raw pointer
operations crash on out-of-bounds accesses at runtime. And in case of plain C,
this may be the most practical way to achieve safety. The main problem with
such a solution lies in dealing with the fact that most pointers in the code
don't have bounds information associated with them. Adding such information
to the code in order to allow the compiler to perform bounds checks at runtime
would result either in a system of attributes significantly more advanced than
the current solution, or a significant change in the runtime representation
of all pointers.

A different approach is taken by AddressSanitizer, which is amazing at detecting
some buffer overflows in the testing phase as it can be used for checking
pointer accesses with respect to a global map of all allocations. However,
it cannot be used as a security mitigation in production binaries, so it can
only help you with vulnerabilities within your test coverage, which is often
insufficient for security purposes.

Fortunately, in case of C++, there already is a standard solution for passing
bounds information alongside the pointer, namely the ``std::span``.
With the help of safe standard containers such as ``std::span``, you can achieve
bounds safety by *writing better modern C++ code*. The proposed C++ modernizer
is designed to help you with that.


But, Really, What About Plain C?
--------------------------------

While plain C does not offer a natural way to combine pointers with bounds
information, parts of this solution may prove useful. In particular,
``#pragma clang unsafe_buffer_usage`` and ``#pragma clang only_safe_buffers``
may be used for isolating code that deals with buffers. Such code can be
carefully audited and the rest of the code may be taught to only interact with
buffers through functions provided by the audited code. Additionally, attribute
``[[clang::unsafe_buffer_usage]]`` can be used for annotating functions
that have bounds-safe alternatives.

However, no automatic code modernizer for plain C is not provided,
and the hardened C++ standard library cannot benefit C code, which limits
usefulness of the proposed integrated programming model in environments
where C++ cannot be used.


The Programming Model for C++
=============================

Let's start with what your C++ code would need to look like in order to
reap benefits of the safe buffers programming model.

Because the model focuses on preventing buffer overflows, other pointer-based
data structures that don't deal with buffers (such as linked lists or trees)
aren't affected by it, don't require any code changes, nor would they benefit
from additional security once the rest of the code is transformed
to comply to the model. In particular, problems like use-after-free and
iterator invalidation are out of scope.

For dealing with buffers, the programming model encourages you to make your code
provide correct bounds information near every buffer access. Such information
can be consumed either by the standard library to conduct runtime bounds checks,
or by the code itself with assertions or safe idioms such as range-based
for-loops. In order to keep bounds information correct, the following
guidelines should be followed:

1. When you allocate a buffer, allocate a container instead. For example,
   if you find yourself allocating a fixed-size array, use ``std::array``.
   If you need to allocate a stack or heap buffer of variable length,
   ``std::vector`` can often act as a drop-in replacement. A vector can do
   a lot more than that, but you can always resort to a fill-constructor that
   preallocates the buffer of necessary size, and then never use any resizing
   operations. This gives you a simple safe buffer at no extra cost other than
   the cost of safety. Another good solution is ``std::span`` which you can
   initialize with a heap pointer allocated with ``new[]``; it doesn't assume
   unique ownership so you can copy such span if you want, just like
   a raw pointer, but of course, you'll have to manually ``delete[]`` it
   when appropriate. While it may be desirable to use ``std::shared_ptr<T>``
   for shared access to the buffer, note that the standard library cannot
   provide hardening in such cases.
2. When you provide function or class interfaces that accept or return
   raw buffers, consider using standard "view" classes such as ``std::span`` or
   ``std::string_view``. Spans are particularly useful as they can be implicitly
   constructed from arbitrary containers which keeps the interface flexible.
   Alternatively, you can pass the original container – move it or pass
   by reference or use a smart pointer, but that locks the function down to
   a specific container type, so this is only useful if you want to use
   container-specific methods inside the function.
3. If the interface you're providing needs to preserve compatibility with
   other code already written to use the old raw buffer interface, annotate the
   raw buffer interface function with the ``[[clang::unsafe_buffer_usage]]``
   attribute. This would inform your clients that there exists a safer
   alternative so that they could gracefully convert their code to comply with
   the programming model as well.
4. If you deal with interfaces you cannot change, such as an interface of
   a third-party library you're using, and these interfaces require you to pass
   buffers as raw pointers, make sure you "containerize" these buffers
   *as soon as possible*. For example, if the library returns a raw buffer
   pointer, put it into ``std::span`` in order to immediately write down
   the size of that buffer for the purposes of future bounds checks.
   Then keep such buffers contained this way *for as long as possible*,
   especially when passing across function boundaries. Say, until you need to
   pass it back to that library, you can pass ``std::span`` by value between
   your functions to preserve precise size information naturally.
5. Sometimes you will find yourself implementing a custom container.
   Say, ``std::vector`` or ``std::span`` may turn out to be poorly suitable
   for your needs, and this is fine. In such cases, the same guidelines apply.
   You may have to use the opt-out pragma on the implementation of
   the container – that's exactly their purpose! Additionally, consider
   implementing runtime checks in your container similar to the ones already
   present in hardened libc++, because following the guidelines alone is often
   insufficient without such hardening.
   (TODO: Will automatic fixits be able to suggest custom containers or views?)
   (TODO: Explain how to implement such checks in a custom container?)


Compiler Tooling And Enforcement
================================

Now that we know what the code needs to look like, let's talk about what can
the compiler do for you to help you with converting your code, as well as
making sure that the code stays safe after later changes.

The Warning
-----------

The warning ``-Wunsafe-buffer-usage`` warns on the following operations:

  - Array subscript expression on raw arrays or raw pointers,

      - unless the index is a compile-time constant ``0``,
      - or, in case of arrays, if both the index and the array size is known
        at compile time and the index is within bounds;

  - Increment and decrement of a raw pointer with operators ``++`` and ``--``;
  - Addition or subtraction of a number to/from a raw pointer with operators
    ``+``, ``-``, ``+=``, ``-=``,

      - unless that number is a compile time constant ``0``;
      - subtraction between two pointers is also fine;

  - Passing a pointer through a function parameter annotated with
    attribute ``[[clang::unsafe_buffer_usage]]``,

      - unless the pointer is a compile time constant ``0`` or ``nullptr``
        (possibly a result of simple operations, such as C-style ``NULL``
        that expands to ``((void) 0)`` which can be "folded" to ``0``
        at compile time);
      - a number of C/C++ standard library buffer manipulation functions
        (such as ``std::memcpy()`` or ``std::next()``) are hardcoded to act
        as if they had the attribute.

The warning doesn't warn on single pointer use in general, such as
dereferencing operations like ``*`` or ``->`` or ``[0]``. If such operation
causes a buffer overflow, there's probably another unsafe operation nearby
that the warning does warn about. Pointer-based data structures
such as linked lists or trees are allowed as they don't typically cause
buffer overflows. "Temporal" safety issues that arise from using raw pointers,
such use-after-free, null pointer dereference, dangling pointers,
reference invalidation, are out of scope for this warning.

The warning also doesn't warn every time a pointer is passed into a function,
but only when the function is annotated with the attribute. Because
the attribute can be added to functions by automatic fixits, the warning
and the fixes can propagate across function boundaries. The users are also
encouraged to annotate their unsafe functions manually. But the warning is
not otherwise inter-procedural.


The Pragmas
-----------

In order to aid incremental adoption of the programming model, you are
encouraged to enable/disable the warning on a file-by-file basis. Additionally,
pragmas are provided to annotate sections of the code as opt-in to the model
and activate the warnings:

.. code-block:: c++

  #pragma clang only_safe_buffers begin
    ...
  #pragma clang only_safe_buffers end

or opt-out of the model and deactivate the warnings on a section of code:

.. code-block:: c++

  #pragma clang unsafe_buffer_usage end
    ...
  #pragma clang unsafe_buffer_usage begin

Such pragmas not only enable incremental adoption with much smaller granularity,
but also provide essential "escape hatches" when the programming model
is undesirable for a section of code (such as tight loops in
performance-critical code, or implementation of a custom container). In such
cases, sections of code with unsafe buffer usage deserve being explicitly marked
and easily auditable by security researches.

Even though similar functionality can be achieved with the generic pragma
``#pragma clang diagnostic``, our specialized pragmas are preferable because
they clearly document the property of the code within them. Additionally,
it is problematic to "interleave" different instances of
``#pragma clang diagnostic push`` and ``#pragma clang diagnostic pop``
between each other, as ``pop`` always pops the last ``push`` and there's no way
to pop the one you want:

.. code-block:: c++

  #pragma clang diagnostic push warning "-Wfoo"
  #pragma clang diagnostic push warning "-Wsafe-buffers-usage"
  #pragma clang diagnostic pop //end of -Wsafe-buffers-usage
  #pragma clang diagnostic pop //end of -Wfoo

versus

.. code-block:: c++

  #pragma clang diagnostic push warning "-Wfoo"
  #pragma clang unsafe_buffer_usage begin
  #pragma clang diagnostic pop //end of -Wfoo
  #pragma clang unsafe_buffer_usage end //end of -Wunsafe-buffer-usage


The Attribute
-------------

The attribute ``[[clang::unsafe_buffer_usage]]`` should be placed on functions
that need to be avoided as they may cause buffer overflows. It is designed to
aid automatic fixits which would replace such unsafe functions with safe
alternatives, though it can be used even when the fix can't be automated.

The attribute is warranted even if the only way a function can overflow
the buffer is by violating the function's preconditions. For example, it
would make sense to put the attribute on function ``foo()`` below because
passing an incorrect size parameter would cause a buffer overflow:

.. code-block:: c++

  [[clang::unsafe_buffer_usage]]
  void foo(int *buf, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      buf[i] = i;
    }
  }

The attribute is NOT warranted when the function has runtime protection against
overflows, assuming hardened libc++, assuming that containers constructed
outside the function are well-formed. For example, function ``bar()`` below
doesn't need an attribute, because assuming buf is well-formed (has size that
fits the original buffer it refers to), hardened libc++ protects this function
from overflowing the buffer:

.. code-block:: c++

  void bar(std::span<int> buf) {
    for (size_t i = 0; i < buf.size(); ++i) {
      buf[i] = i;
    }
  }

This corresponds to our safety precaution about keeping buffers "containerized"
in spans for as long as possible. Function ``foo()`` may have internal
consistency, but by accepting a raw buffer it requires the user to unwrap
the span, which is undesirable.

The attribute is warranted when a function accepts a raw buffer only to
immediately put it into a span:

.. code-block:: c++

  [[clang::unsafe_buffer_usage]]
  void baz(int *buf, size_t size) {
    std::span<int> sp{ buf, size };
    for (size_t i = 0; i < sp.size(); ++i) {
      sp[i] = i;
    }
  }

In this case ``baz()`` does not contain any unsafe operations, but the awkward
parameter type causes the caller to unwrap the span unnecessarily.
In such cases the attribute may never be removed.

In particular, the attribute is NOT an "escape hatch". It does not suppress
the warnings about unsafe operations in the function. Addressing warnings
inside the function is still valuable for internal consistency.

Attribute ``[[clang::unsafe_buffer_usage]]`` is similar to attribute
[[deprecated]] but it has important differences:

* Use of a function annotated by such attribute causes ``-Wunsafe-buffer-usage``
  warning to appear, instead of ``-Wdeprecated``, so they can be
  enabled/disabled independently as all four combinations make sense;
* The "replacement" parameter of ``[[deprecated]]``, which allows for automatic
  fixits when the function has a drop-in replacement, becomes significantly more
  powerful and flexible in ``[[clang::unsafe_buffer_usage]]`` where it will allow
  non-trivial automatic fixes.

(TODO: Explain parameters of the attribute, how they aid automatic fixits)

Code Modernization Workflow With Semi-Automatic Fixits
------------------------------------------------------

Every time your code preforms an unsafe operation that causes a
``-Wunsafe-buffer-usage warning`` to appear, the warning may be accompanied
by an automatic fix that changes types of one or more variables associated
with the unsafe operation from raw pointer or array type to safe container type.

For example, the following function contains a local constant-size array.

.. code-block:: c++

  void use_array(int[] x);

  void test_array() {
    int x[5];
    x[3] = 3; // Warning: Array indexing is unsafe operation!
    use_array(x);
  }

The automatic fixit associated with the warning would transform this array
into an ``std::array``:

.. code-block:: c++

  void use_array(int[] x);

  void test_array() {
    std::array<int> x[5];
    x[3] = 3;
    use_array(x.data());
  }

Note that the use site of variable ``x`` inside the call to ``use_array()``
needed an update. The fixit considers every use site of the fixed variable
and updates them if necessary.

In some cases an automatic fix would be problematic, so the warning will simply
highlight unsafe operations for you to consider.

In some cases a partial fix is emitted, where you'll have to fill in a few
placeholders in order to document the bounds information related to the pointer.
This is particularly common when suggesting ``std::span``.

For example, consider function ``foo()`` we've seen before:

.. code-block:: c++

  void foo(int *buf, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      buf[i] = i;
    }
  }

In spirit of our guidelines, the automatic fixit would prefer this function
to accept a span instead of raw buffer, so that the span didn't need to be
unwrapped. Of course, such change alone would break both source compatibility
and binary compatibility. In order to avoid that, the fix will provide
a compatibility overload to preserve the old functionality. The updated code
produced by the fixit will look like this:

.. code-block:: c++

  [[clang::unsafe_buffer_usage]]
  void foo(int *buf, size_t size) {
    foo(std::span<int>{ buf, __placeholder__}, size);
  }

  void foo(std::span<int> buf, size_t size) {
    for (size_t i = 0; i < size; ++i) {
      buf[i] = i;
    }
  }

The following changes were performed automatically:

  - The type of parameter ``buf`` was changed from ``int *`` to
    ``std::span<int>``. Use sites updated if necessary.
  - A compatibility overload was autogenerated with the old prototype, with
    attribute ``[[clang::unsafe_buffer_usage]]`` attached to it to encourage
    the callers to switch to the new function – and possibly update the callers
    automatically!

The following changes need manual intervention:

  - The compatibility overload contains a ``__placeholder__`` which needs
    to be populated manually. In this case ``size`` is a good candidate.
  - Despite accepting a ``std::span`` which carries size information,
    the fixed function still accepts ``size`` separately. It can be removed
    manually, or it can be preserved, if ``size`` and ``buf.size()``
    actually need to be different in your case.

Placeholders fulfill an important purpose as they attract attention to
situations where the buffer's size wasn't properly documented for the purposes
of bounds checks. Variable ``size`` does not *have* to carry the size of the
buffer (or the size of *that* buffer) just becaused it's named "size".
The compiler will avoid making guesses about that.

The fixits emitted by the warning are correct modulo placeholders. Placeholders
are the only reason why fixed code is allowed to fail to compile.
Incorrectly resolving the placeholder is the only reason why fixed code
will demonstrate incorrect runtime behavior compared to the original code.
In an otherwise well-formed program it is always possible (and usually easy)
to resolve the placeholder correctly.

Note that regardless of how ``__placeholder__`` is resolved, it does not allow
you to remove the ``[[clang::unsafe_buffer_usage]]`` annotation. The annotation
will stay forever because that function is now equivalent to function ``baz()``
we've seen before: it contains no unsafe operations, but it only offers internal
consistency. It is still possible to misuse that function by supplying an
invalid ``size`` parameter. It still requires you to unwrap ``std::span`` if you
already have it, only to wrap it back immediately. So the callers should still
be updated to use the new function, and automatic fixits will now be emitted
for the call sites to aid that.

Even if you decide to remove the ``size`` parameter, fixits at call sites
will remain operational. A warning will be emitted if the replacement function's
prototype diverges from the original prototype beyond recognition. In such cases
an attribute can be either updated to give more manual hints to the compiler, or
changed to a different variant that explicitly opts out of automatic fixits.

(TODO: Elaborate on that last point and confirm that we actually want
such behavior.)

(TODO: Cover more examples.)

The Hardened libc++
===================

(TODO: Write this section. Probably just link to its own documentation.)

Clang Static Analyzer Checks
============================

(TODO: Write this section.)
