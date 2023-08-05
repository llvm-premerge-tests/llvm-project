.. title:: clang-tidy - bugprone-new-bool-conversion

bugprone-new-bool-conversion
============================

Detects cases where the result of a new expression is used as a ``bool``.

In C++, the new expression dynamically allocates memory on the heap and returns
a pointer to the newly created object. However, when developers inadvertently
use this pointer directly as a boolean value, either through implicit or
explicit conversion, a hidden problem emerges. The crux of the issue lies in
the fact that any non-null pointer implicitly converts to ``true``, masking
potential errors and making the code difficult to comprehend. Worse yet, it may
trigger memory leaks, as dynamically allocated objects remain inaccessible,
never to be released by the program.

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

Check comes with no configuration options and does not offer any auto-fixes.
