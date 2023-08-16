// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

template <typename _Iterator> class normal_iterator {};

template <typename From, typename To> struct is_convertible {};

template <typename From, typename To>
inline constexpr bool is_convertible_v = is_convertible<From, To>::value; // #1

// expected-error@#1 {{no member named 'value' in 'is_convertible<bool, bool>'}}

template <typename From, typename To>
concept convertible_to = is_convertible_v<From, To>; // expected-note 0+{{}}
template <typename IteratorL, typename IteratorR>
  requires requires(IteratorL lhs, IteratorR rhs) { // expected-note 0+{{}}
    { lhs == rhs } -> convertible_to<bool>; // #2
  }
constexpr bool compare(normal_iterator<IteratorL> lhs, normal_iterator<IteratorR> rhs) {
  return false;
}

// We don't know exactly the substituted type for `lhs == rhs`, thus a placeholder 'expr-type' is emitted.
// expected-note@#2 {{'convertible_to<expr-type, bool>'}}

// Consume remaining notes/errors.
// expected-note@* 0+{{}}
// expected-error@* 0+{{}}
class Object;

void function() {
  normal_iterator<Object *> begin, end;
  compare(begin, end);
}
