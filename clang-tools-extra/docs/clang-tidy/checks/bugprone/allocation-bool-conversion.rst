.. title:: clang-tidy - bugprone-allocation-bool-conversion

bugprone-allocation-bool-conversion
===================================

Detects cases where the result of a resource allocation is used as a
``bool``.

Functions like ``new``, ``malloc``, ``fopen``, etc., dynamically allocate memory
or resources and return pointers to the newly created objects. However, using
these pointers directly as boolean values, either through implicit or explicit
conversion, a hidden problem emerges. The crux of the issue lies in the fact
that any non-null pointer implicitly converts to ``true``, masking potential
errors and making the code difficult to comprehend. Worse yet, it may trigger
memory leaks, as dynamically allocated resources remain inaccessible, never to
be released by the program.

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

Check does not offer any auto-fixes.

Options
-------

.. option:: AllocationFunctions

   Custom functions considered as allocators can be specified using a
   semicolon-separated list of (fully qualified) function names or regular
   expressions matched against the called function names. Default value:
   `malloc;calloc;realloc;strdup;fopen;fdopen;freopen;opendir;fdopendir;popen;
   mmap;allocate`.
