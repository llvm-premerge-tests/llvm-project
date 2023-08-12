.. _implementation-defined-behavior:

===============================
Implementation-defined behavior
===============================

Contains the implementation details of the implementation-defined behavior
in libc++. The order of the entries matches the entries in the
`draft of the Standard <http://eel.is/c++draft/impldefindex>`_.

.. note:
   This page is far from complete.

a terminal capable of displaying Unicode
----------------------------------------

* First it determines whether the stream's ``rdbuf()`` has an underlying
  ``FILE*``. This is ``true`` in the following cases:

  * The stream is ``std::cout``, ``std::cerr``, or ``std::clog``.

  * A ``std::basic_filebuf<CharT, Traits>`` derived from ``std::filebuf``.

* The way to determine whether this ``FILE*`` is the same as specified
  for `void vprint_unicode(FILE* stream, string_view fmt, format_args args);
  <http://eel.is/c++draft/print.fun#7>`_.

