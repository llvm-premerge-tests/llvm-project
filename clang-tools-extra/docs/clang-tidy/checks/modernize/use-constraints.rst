.. title:: clang-tidy - modernize-use-constraints

modernize-use-constraints
=========================

Replace ``enable_if`` with C++20 requires clauses.

``enable_if`` is a SFINAE mechanism for selecting the desired function or class
template based on type traits or other requirements. ``enable_if`` changes the
meta-arity of the template, and has other `adverse side effects <https://open-std.org/JTC1/SC22/WG21/docs/papers/2016/p0225r0.html>`_
in the code. C++20 introduces concepts and constraints as a cleaner language
provided solution to achieve the same outcome.

This check replaces common ``enable_if`` patterns that can be replaced
by C++20 requires clauses. Uses that cannot be replaced automatically
will emit a diagnostic without a replacement

.. code-block:: c++

  template <typename T>
  std::enable_if_t<T::some_trait, int> only_if_t_has_the_trait() { ... }

  template <typename T, std::enable_if_t<T::some_trait, int> = 0>
  void another_version() { ... }
