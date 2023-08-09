.. title:: clang-tidy - bugprone-allocation-bool-conversion

bugprone-allocation-bool-conversion
===================================

Detects cases where the result of a resource allocation is used as a
``bool``.

Expressions like ``new`` and ``new[]`` and functions like ``malloc``, ``fopen``,
etc., dynamically allocate memory or resources and return pointers to the newly
created objects. However, using these pointers directly as boolean values,
either through implicit or explicit conversion, a hidden problem emerges. The
crux of the issue lies in the fact that any non-null pointer implicitly converts
to ``true``, masking potential errors and making the code difficult to
comprehend. Similar issues exist with ``open``, ``socket``, ``accept``, etc.,
those functions returns handle to resources in form of integer that can be
silently convert to boolean. Worse yet, it may trigger memory leaks, as
dynamically allocated resources remain inaccessible, never to be released by the
program.

Example:

.. code-block:: c++

  #include <iostream>

  bool processResource(bool resourceFlag) {
      if (resourceFlag) {
          std::cout << "Resource processing successful!" << std::endl;
          return true;
      } else {
          std::cout << "Resource processing failed!" << std::endl;
          return false;
      }
  }

  int main() {
      // Implicit conversion of int* to bool
      processResource(new int());
      return 0;
  }

In this example, we pass the result of the ``new int()`` expression directly to
the ``processResource`` function as its argument. Since the pointer returned by
``new`` is non-null, it is implicitly converted to ``true``, leading to
incorrect behavior in the ``processResource`` function.

Predefined list of pointer-returning allocator functions: ``malloc``,
``calloc``, ``realloc``, ``aligned_alloc``, ``allocate``, ``fopen``, ``fdopen``,
``freopen``, ``popen``, ``tmpfile``, ``::opendir``, ``::fdopendir``, ``::mmap``,
``::reallocf``, ``::strdup``, ``::wcsdup``, ``::strndup``, ``::realpath``,
``::tempnam``, ``::canonicalize_file_name``, ``::dbopen``, ``::fmemopen``,
``::open_memstream``, ``::open_wmemstream``, ``::get_current_dir_name``,
``memalloc``, ``memcalloc``, ``memrealloc``, ``::mpool_open``,
``::posix_memalign``, ``::memalign``, ``::valloc``, ``::pvalloc"``.

Predefined list of integer-returning allocator functions: ``::open``,
``::openat``, ``::creat``, ``::dup``, ``::dup2``, ``::dup3``, ``::socket``,
``::accept``, ``::pipe``, ``::pipe2``, ``::mkfifo``, ``::mkfifoat``,
``::mkstemp``, ``::mkostemp``, ``::mkstemps``, ``::mkostemps``.

The options `PointerReturningAllocators` and `IntegerReturningAllocators` allow
for an extension of those lists.

Check does not offer any auto-fixes.

Options
-------

.. option:: PointerReturningAllocators

   List of additional custom functions considered as allocators. Can be
   specified using a semicolon-separated list of (fully qualified) function
   names or regular expressions matched against the called function names.
   Default value: empty string.

.. option:: IntegerReturningAllocators

   List of additional custom functions considered as allocators. Can be
   specified using a semicolon-separated list of (fully qualified) function
   names or regular expressions matched against the called function names.
   Default value: empty string.
